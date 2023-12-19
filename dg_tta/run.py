import sys
import re
from pathlib import Path
import importlib
import shutil
import json
import argparse
from copy import deepcopy
import json
from datetime import datetime
from contextlib import nullcontext

import torch
from torch._dynamo import OptimizedModule
import torch.nn.functional as F

from tqdm import trange, tqdm

if importlib.util.find_spec("wandb"):
    import wandb

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.run.run_training import run_training_entry as nnunet_run_training_main

import randomname

import dg_tta
from dg_tta.__build__ import inject_dg_trainers_into_nnunet
from dg_tta.utils import disable_internal_augmentation, check_dga_root_is_set
from dg_tta.gin import gin_aug
from dg_tta.tta.torch_utils import (
    get_batch,
    map_label,
    dice_coeff,
    soft_dice_loss,
    fix_all,
    release_all,
    release_norms,
    register_forward_pre_hook_at_beginning,
    register_forward_hook_at_beginning,
    hookify,
    generate_label_mapping,
    get_map_idxs,
    get_imgs,
)
from dg_tta.tta.augmentation_utils import get_disp_field, get_rand_affine
from dg_tta.tta.config_log_utils import (
    wandb_run,
    load_current_modifier_functions,
    get_global_idx,
    get_tta_folders,
    wandb_run_is_available,
    suppress_stdout,
    plot_run_results,
)


PROJECT_NAME = "nnunet_tta"
INTENSITY_AUG_FUNCTION_DICT = {"disabled": lambda img: img, "GIN": gin_aug}


def get_data_iterator(
    config, predictor, tta_data_filepaths, dataset_raw_path, tta_dataset_bucket
):
    assert tta_dataset_bucket in ["imagesTs", "imagesTr"]

    list_of_lists = [
        [_path]
        for _path in tta_data_filepaths
        if Path(_path).parts[-2] == tta_dataset_bucket
    ]

    label_folder = "labelsTs" if tta_dataset_bucket == "imagesTs" else "labelsTr"
    output_folder = (
        "tta_outputTs" if tta_dataset_bucket == "imagesTs" else "tta_outputTr"
    )

    (
        list_of_lists_or_source_folder,
        output_filename_truncated,
        seg_from_prev_stage_files,
    ) = predictor._manage_input_and_output_lists(
        list_of_lists, output_folder, dataset_raw_path / label_folder
    )

    if len(list_of_lists_or_source_folder) == 0:
        return iter(())

    seg_from_prev_stage_files = [
        s if Path(s).is_file() else None for s in seg_from_prev_stage_files
    ]
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        config["num_processes"],
    )
    return data_iterator


def load_tta_data(config, dataset_raw_path, predictor):
    with suppress_stdout():
        ts_iterator = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTs",
        )
        tr_iterator = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTr",
        )

    data = list(ts_iterator) + list(tr_iterator)

    return data


def load_network(weights_file, device):
    pretrained_weights_filepath = Path(*Path(weights_file).parts[:-2])
    fold = Path(weights_file).parts[-2].replace("fold_", "")
    use_folds = [int(fold)] if fold.isnumeric() else fold
    checkpoint_name = Path(weights_file).parts[-1]
    configuration = Path(weights_file).parts[-3].split("__")[-1]

    perform_everything_on_gpu = True
    verbose = False

    predictor = nnUNetPredictor(
        perform_everything_on_gpu=perform_everything_on_gpu,
        device=device,
        verbose_preprocessing=verbose,
    )

    predictor.initialize_from_trained_model_folder(
        pretrained_weights_filepath, use_folds, checkpoint_name
    )

    parameters = predictor.list_of_parameters
    plans_manager = predictor.plans_manager
    network = predictor.network
    patch_size = plans_manager.get_configuration(configuration).patch_size

    return predictor, patch_size, network, parameters


def run_inference(config, tta_data, model, predictor, all_tta_parameter_paths):
    save_probabilities = False
    num_processes_segmentation_export = config["num_processes"]

    tta_parameters = []
    for _path in all_tta_parameter_paths:
        tta_parameters.extend(torch.load(_path))

    predictor.network = deepcopy(model)
    predictor.list_of_parameters = tta_parameters
    predictor.predict_from_data_iterator(
        tta_data, save_probabilities, num_processes_segmentation_export
    )


def get_model_from_network(network, modifier_fn_module, parameters=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    # Register hook that modifies the input prior to custom augmentation
    modify_tta_input_fn = modifier_fn_module.ModifierFunctions.modify_tta_input_fn
    register_forward_pre_hook_at_beginning(
        model, hookify(modify_tta_input_fn, "forward_pre_hook")
    )

    # Register hook that modifies the output of the model
    modfify_tta_model_output_fn = (
        modifier_fn_module.ModifierFunctions.modfify_tta_model_output_fn
    )
    register_forward_hook_at_beginning(
        model, hookify(modfify_tta_model_output_fn, "forward_hook")
    )

    return model


running_stats_buffer = {}


def buffer_running_stats(m):
    _id = id(m)
    if (
        hasattr(m, "running_mean")
        and hasattr(m, "running_var")
        and not _id in running_stats_buffer
    ):
        if m.running_mean is not None and m.running_var is not None:
            running_stats_buffer[_id] = [m.running_mean.data, m.running_var.data]


def apply_running_stats(m):
    _id = id(m)
    if (
        hasattr(m, "running_mean")
        and hasattr(m, "running_var")
        and _id in running_stats_buffer
    ):
        m.running_mean.data.copy_(other=running_stats_buffer[_id][0])
        m.running_var.data.copy_(
            other=running_stats_buffer[_id][1]
        )  # Copy into .data to prevent backprop errors
        del running_stats_buffer[_id]


def get_parameters_save_path(save_path, sample_id, ensemble_idx):
    tta_parameters_save_path = (
        save_path / f"{sample_id}__ensemble_idx_{ensemble_idx}_tta_parameters.pt"
    )
    return tta_parameters_save_path


def calc_branch(
    branch_id,
    config,
    model,
    intensity_aug_func,
    identity_grid,
    patch_size,
    batch_size,
    label_mapping,
    optimized_labels,
    modifier_fn_module,
    imgs,
    device,
):
    assert branch_id in ["branch_a", "branch_b"]

    grad_context = (
        nullcontext if config["have_grad_in"] in ["branch_a", "both"] else torch.no_grad
    )

    modify_tta_output_after_mapping_fn = (
        modifier_fn_module.ModifierFunctions.modify_tta_output_after_mapping_fn
    )

    with grad_context():
        zero_grid = 0.0 * identity_grid

        imgs_aug = imgs

        if config["do_intensity_aug_in"] in [branch_id, "both"]:
            imgs_aug = intensity_aug_func(imgs_aug)
        else:
            imgs_aug = imgs_aug

        grid = zero_grid
        grid_inverse = zero_grid

        if config["spatial_aug_type"] == "affine" and config["do_spatial_aug_in"] in [
            branch_id,
            "both",
        ]:
            R, R_inverse = get_rand_affine(batch_size, flip=False)
            R, R_inverse = R.to(device), R_inverse.to(device)
            grid = grid + (
                F.affine_grid(R, [batch_size, 1] + patch_size, align_corners=False)
                - identity_grid
            )
            grid_inverse = grid_inverse + (
                F.affine_grid(
                    R_inverse, [batch_size, 1] + patch_size, align_corners=False
                )
                - identity_grid
            )

        if config["spatial_aug_type"] == "deformable" and config[
            "do_spatial_aug_in"
        ] in [branch_id, "both"]:
            grid_deformable, grid_deformable_inverse = get_disp_field(
                batch_size,
                patch_size,
                factor=0.5,
                interpolation_factor=5,
                device=device,
            )
            grid = grid + grid_deformable
            grid_inverse = grid_inverse + grid_deformable_inverse

        if config["do_spatial_aug_in"] in [branch_id, "both"]:
            grid = grid + identity_grid
            imgs_aug = F.grid_sample(
                imgs_aug, grid, padding_mode="border", align_corners=False
            )

        if branch_id == "branch_a":
            model.apply(buffer_running_stats)
        elif branch_id == "branch_b":
            model.apply(apply_running_stats)
        else:
            raise ValueError()

        branch_target = model(imgs_aug)
        branch_target = map_label(
            branch_target,
            get_map_idxs(label_mapping, optimized_labels, input_type="pretrain_labels"),
            input_format="logits",
        )
        branch_target = modify_tta_output_after_mapping_fn(branch_target)

        if isinstance(branch_target, tuple):
            branch_target = branch_target[0]

        if config["do_spatial_aug_in"] in [branch_id, "both"]:
            grid_inverse = grid_inverse + identity_grid
            branch_target = F.grid_sample(
                branch_target, grid_inverse, align_corners=False
            )
        else:
            branch_target = branch_target

        return branch_target


def tta_main(
    run_name,
    config,
    tta_data_dir,
    save_base_path,
    label_mapping,
    modifier_fn_module,
    device,
    debug=False,
):
    START_CLASS = 1  # Do not use background for consistency loss

    # Load model
    pretrained_weights_filepath = config["pretrained_weights_filepath"]
    predictor, patch_size, network, parameters = load_network(
        pretrained_weights_filepath, device
    )

    # Load TTA data
    tta_data = load_tta_data(config, tta_data_dir, predictor)

    num_samples = len(tta_data)
    tta_across_all_samples = config["tta_across_all_samples"]

    ensemble_count = config["ensemble_count"]
    B = config["batch_size"]
    patches_to_be_accumulated = config["patches_to_be_accumulated"]
    tta_eval_patches = config["tta_eval_patches"]
    num_epochs = config["epochs"]
    start_tta_at_epoch = config["start_tta_at_epoch"]

    optimized_labels = config["optimized_labels"]

    save_path = Path(save_base_path) / run_name
    save_path.mkdir(exist_ok=True, parents=False)

    with open(save_path / "tta_plan.json", "w") as f:
        json.dump({k: v for k, v in config.items()}, f, indent=4)

    sitk_io = SimpleITKIO()

    identity_grid = F.affine_grid(
        torch.eye(4, device=device).repeat(B, 1, 1)[:, :3],
        [B, 1] + patch_size,
        align_corners=False,
    )

    if tta_across_all_samples:
        sample_range = [0]
    else:
        sample_range = trange(num_samples, desc="Samples")

    disable_internal_augmentation()

    for smp_idx in sample_range:
        if tta_across_all_samples:
            tta_tens_list = [e["data"] for e in tta_data]
            sample_id = "all_samples"
            sub_dir_tta = save_path / "tta_output"
        else:
            tta_tens_list = [tta_data[smp_idx]["data"]]
            sample_id = tta_data[smp_idx]["ofile"]
            sub_dir_tta = save_path / Path(sample_id).parent

        sub_dir_tta.mkdir(exist_ok=True)

        for ensemble_idx in trange(ensemble_count, desc="Ensembles"):
            tta_parameters_save_path = get_parameters_save_path(
                sub_dir_tta, sample_id, ensemble_idx
            )
            if tta_parameters_save_path.is_file():
                tqdm.write(
                    f"TTA parameters file already exists. Skipping '{tta_parameters_save_path}'"
                )
                continue

            tta_losses = torch.zeros(num_epochs)
            eval_dices = torch.zeros(num_epochs)

            intensity_aug_func = INTENSITY_AUG_FUNCTION_DICT[
                config["intensity_aug_function"]
            ]

            model = get_model_from_network(network, modifier_fn_module, parameters)
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

            tbar = trange(num_epochs, desc="Epoch")

            model.apply(fix_all)
            for epoch in tbar:
                model.train()
                global_idx = get_global_idx(
                    [
                        (smp_idx, num_samples),
                        (ensemble_idx, ensemble_count),
                        (epoch, num_epochs),
                    ]
                )
                if wandb_run_is_available():
                    wandb.log({"ref_epoch_idx": epoch}, global_idx)
                step_losses = []

                if epoch == start_tta_at_epoch:
                    tqdm.write("Starting train")
                    model.apply(fix_all)
                    if config["params_with_grad"] == "all":
                        model.apply(release_all)
                    elif config["params_with_grad"] == "norms":
                        model.apply(release_norms)
                    elif config["params_with_grad"] == "encoder":
                        model.encoder.apply(release_all)
                    else:
                        raise ValueError()

                    grad_params = {
                        id(p): p.numel() for p in model.parameters() if p.requires_grad
                    }
                    tqdm.write(
                        f"Released #{sum(list(grad_params.values()))/1e6:.2f} million trainable params"
                    )

                for _ in range(patches_to_be_accumulated):
                    with torch.no_grad():
                        imgs, _ = get_batch(
                            tta_tens_list,
                            torch.randperm(len(tta_tens_list))[:B],
                            patch_size,
                            fixed_patch_idx=None,
                            device=device,
                        )

                    imgs = torch.cat(imgs, dim=0)

                    target_a = calc_branch(
                        "branch_a",
                        config,
                        model,
                        intensity_aug_func,
                        identity_grid,
                        patch_size,
                        B,
                        label_mapping,
                        optimized_labels,
                        modifier_fn_module,
                        imgs,
                        device,
                    )
                    target_b = calc_branch(
                        "branch_b",
                        config,
                        model,
                        intensity_aug_func,
                        identity_grid,
                        patch_size,
                        B,
                        label_mapping,
                        optimized_labels,
                        modifier_fn_module,
                        imgs,
                        device,
                    )

                    # Apply consistency loss
                    common_content_mask = (
                        target_a.sum(1, keepdim=True) > 0.0
                    ).float() * (target_b.sum(1, keepdim=True) > 0.0).float()
                    sm_a = target_a.softmax(1) * common_content_mask
                    sm_b = target_b.softmax(1) * common_content_mask

                    loss = 1 - soft_dice_loss(sm_a, sm_b)[:, START_CLASS:].mean()

                    loss_accum = loss / patches_to_be_accumulated
                    step_losses.append(loss.detach().cpu())

                    if epoch >= start_tta_at_epoch:
                        loss_accum.backward()

                if epoch >= start_tta_at_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

                tta_losses[epoch] = torch.stack(step_losses).mean().item()

                with torch.inference_mode():
                    model.eval()
                    for _ in range(tta_eval_patches):
                        imgs, labels = get_batch(
                            tta_tens_list,
                            torch.randperm(len(tta_tens_list))[:B],
                            patch_size,
                            fixed_patch_idx="center",  # This is just for evaluation purposes
                            device=device,
                        )

                        imgs = torch.cat(imgs, dim=0)

                        none_labels = [l is None for l in labels]
                        filtered_imgs = imgs[~torch.as_tensor(none_labels)]
                        filtered_labels = [
                            l for flag, l in zip(none_labels, labels) if not flag
                        ]

                        if len(filtered_imgs) == 0:
                            eval_dices[epoch] = float("nan")
                            continue

                        else:
                            filtered_labels = torch.cat(filtered_labels, dim=0)
                            output_eval = model(filtered_imgs)
                            if isinstance(output_eval, tuple):
                                output_eval = output_eval[0]

                            output_eval = map_label(
                                output_eval,
                                get_map_idxs(
                                    label_mapping,
                                    optimized_labels,
                                    input_type="pretrain_labels",
                                ),
                                input_format="logits",
                            )
                            target_argmax = output_eval.argmax(1)

                            filtered_labels = map_label(
                                filtered_labels,
                                get_map_idxs(
                                    label_mapping,
                                    optimized_labels,
                                    input_type="tta_labels",
                                ),
                                input_format="argmaxed",
                            ).long()
                            d_tgt_val = dice_coeff(
                                target_argmax, filtered_labels, len(optimized_labels)
                            )

                            eval_dices[epoch] += (
                                1 / tta_eval_patches * d_tgt_val.mean().item()
                            )

                    if debug:
                        break

                tbar.set_description(
                    f"Epochs, loss={tta_losses[epoch]:.3f}, Pseudo-Dice={eval_dices[epoch]*100:.1f}%"
                )
                if wandb_run_is_available():
                    wandb.log(
                        {
                            f"losses/loss__{sample_id}__ensemble_idx_{ensemble_idx}": tta_losses[
                                epoch
                            ]
                        },
                        step=global_idx,
                    )
                    wandb.log(
                        {
                            f"scores/eval_dice__{sample_id}__ensemble_idx_{ensemble_idx}": eval_dices[
                                epoch
                            ]
                        },
                        step=global_idx,
                    )

            tta_parameters = [model.state_dict()]
            torch.save(tta_parameters, tta_parameters_save_path)

            if not wandb_run_is_available():
                plot_run_results(
                    sub_dir_tta, sample_id, ensemble_idx, tta_losses, eval_dices
                )

            if debug:
                break
        # End of ensemble loop

    print("Starting prediction")
    all_prediction_save_paths = []

    for smp_idx in trange(num_samples, desc="Samples"):
        if tta_across_all_samples:
            param_sample_id = "all_samples"
            sub_dir_tta = save_path / "tta_output"
        else:
            param_sample_id = tta_data[smp_idx]["ofile"]
            sub_dir_tta = save_path / Path(param_sample_id).parent

        ensemble_parameter_paths = []
        tta_sample = tta_data[smp_idx]
        tta_sample["data"] = get_imgs(tta_sample["data"].unsqueeze(0)).squeeze(0)

        # Update internal save path for nnUNet
        ofile = tta_data[smp_idx]["ofile"]
        new_ofile = str(save_path / ofile)
        tta_data[smp_idx]["ofile"] = new_ofile

        prediction_save_path = Path(new_ofile + ".nii.gz")
        prediction_save_path.parent.mkdir(exist_ok=True)

        for ensemble_idx in range(config["ensemble_count"]):
            ensemble_parameter_paths.append(
                get_parameters_save_path(sub_dir_tta, param_sample_id, ensemble_idx)
            )

        disable_internal_augmentation()
        model = get_model_from_network(network, modifier_fn_module)

        run_inference(config, [tta_sample], model, predictor, ensemble_parameter_paths)

        predicted_output_array, data_properties = sitk_io.read_seg(prediction_save_path)
        predicted_output = map_label(
            torch.as_tensor(predicted_output_array),
            get_map_idxs(label_mapping, optimized_labels, input_type="pretrain_labels"),
            input_format="argmaxed",
        ).squeeze(0)

        sitk_io.write_seg(
            predicted_output.numpy(), prediction_save_path, properties=data_properties
        )
        all_prediction_save_paths.append(prediction_save_path)

    # End of sample loop

    tqdm.write("Evaluating predictions...")

    for pred_path in all_prediction_save_paths:
        pred_label_name = Path(pred_path).name
        if "outputTs" in Path(pred_path).parent.parts[-1]:
            path_mapped_target = save_path / "mapped_target_labelsTs" / pred_label_name
            path_orig_target = tta_data_dir / "labelsTs" / pred_label_name
        elif "outputTr" in Path(pred_path).parent.parts[-1]:
            path_mapped_target = save_path / "mapped_target_labelsTr" / pred_label_name
            path_orig_target = tta_data_dir / "labelsTr" / pred_label_name
        else:
            raise ValueError()

        if not path_orig_target.is_file():
            # No target available
            continue

        path_mapped_target.parent.mkdir(exist_ok=True)
        shutil.copy(path_orig_target, path_mapped_target)

        seg, sitk_stuff = sitk_io.read_seg(path_mapped_target)
        seg = torch.as_tensor(seg)
        mapped_seg = map_label(
            seg,
            get_map_idxs(label_mapping, optimized_labels, input_type="tta_labels"),
            input_format="argmaxed",
        ).squeeze(0)
        sitk_io.write_seg(mapped_seg.squeeze(0).numpy(), path_mapped_target, sitk_stuff)

    for bucket in ["Ts", "Tr"]:
        all_mapped_targets_path = save_path / f"mapped_target_labels{bucket}"
        all_pred_targets_path = save_path / f"tta_output{bucket}"

        if not all_mapped_targets_path.is_dir() or not all_pred_targets_path.is_dir():
            continue

        # Run postprocessing
        postprocess_results_fn = (
            modifier_fn_module.ModifierFunctions.postprocess_results_fn
        )
        postprocess_results_fn(all_pred_targets_path)

        summary_path = f"{save_path}/summary_{bucket}.json"
        compute_metrics_on_folder_simple(
            folder_ref=all_mapped_targets_path,
            folder_pred=all_pred_targets_path,
            labels=list(range(len(optimized_labels))),
            output_file=summary_path,
            num_processes=config["num_processes"],
            chill=True,
        )

        with open(summary_path, "r") as f:
            summary_json = json.load(f)
            final_mean_dice = summary_json["foreground_mean"]["Dice"]

        if wandb_run_is_available():
            wandb.log({f"scores/tta_dice_mean_{bucket}": final_mean_dice})


class DGTTAProgram:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="DG-TTA for nnUNetv2",
            usage="""dgtta <command> [<args>]

        Commands are:
        inject_trainers Inject DG trainers into nnUNet module
        pretrain        Pretrain on a dataset with DG trainers
        prepare_tta     Prepare test-time adaptation
        run_tta         Run test-time adaptation
        """,
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def inject_trainers(self):
        parser = argparse.ArgumentParser(
            description="Inject DG-TTA trainers into nnUNet module code"
        )
        parser.add_argument(
            "--num_epochs", type=int, default=1000, help="Number of epochs to train"
        )
        args = parser.parse_args(sys.argv[2:])
        inject_dg_trainers_into_nnunet(args.num_epochs)

    def pretrain(self):
        print("Dispatching into nnUNetv2_train.")
        sys.argv = sys.argv[2:]
        sys.argv.insert(0, "nnUNetv2_train")
        nnunet_run_training_main()

    def prepare_tta(self):
        parser = argparse.ArgumentParser(
            description="Prepare DG-TTA", usage="""dgtta prepare_tta [-h]"""
        )

        parser.add_argument(
            "pretrained_dataset_id",
            help="""
                            Task ID for pretrained model.
                            Can be numeric or one of ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND']""",
        )
        parser.add_argument("tta_dataset_id", help="Task ID for TTA")
        parser.add_argument(
            "--pretrainer",
            help="Trainer to use for pretraining",
            default=None,
        )
        parser.add_argument(
            "--pretrainer_config",
            help="Fold ID of nnUNet model to use for pretraining",
            default="3d_fullres",
        )
        parser.add_argument(
            "--pretrainer_fold",
            help="Fold ID of nnUNet model to use for pretraining",
            default="0",
        )
        parser.add_argument(
            "--tta_dataset_bucket",
            help="""Can be one of ['imagesTr', 'imagesTs', 'imagesTrAndTs']""",
            default="imagesTs",
        )

        args = parser.parse_args(sys.argv[2:])
        pretrained_dataset_id = (
            int(args.pretrained_dataset_id)
            if args.pretrained_dataset_id.isnumeric()
            else args.pretrained_dataset_id
        )
        pretrainer_fold = (
            int(args.pretrainer_fold)
            if args.pretrainer_fold.isnumeric()
            else args.pretrainer_fold
        )

        dg_tta.tta.config_log_utils.prepare_tta(
            pretrained_dataset_id,
            int(args.tta_dataset_id),
            pretrainer=args.pretrainer,
            pretrainer_config=args.pretrainer_config,
            pretrainer_fold=pretrainer_fold,
            tta_dataset_bucket=args.tta_dataset_bucket,
        )

    def run_tta(self):
        parser = argparse.ArgumentParser(description="Run DG-TTA")
        parser.add_argument(
            "pretrained_dataset_id",
            help="""
                            Task ID for pretrained model.
                            Can be numeric or one of ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND']""",
        )
        parser.add_argument("tta_dataset_id", help="Task ID for TTA")
        parser.add_argument(
            "--pretrainer",
            help="Trainer to use for pretraining",
            default=None,
        )
        parser.add_argument(
            "--pretrainer_config",
            help="Fold ID of nnUNet model to use for pretraining",
            default="3d_fullres",
        )
        parser.add_argument(
            "--pretrainer_fold",
            help="Fold ID of nnUNet model to use for pretraining",
            default="0",
        )
        parser.add_argument("--device", help="Device to be used", default="cuda")

        args = parser.parse_args(sys.argv[2:])
        pretrained_dataset_id = (
            int(args.pretrained_dataset_id)
            if args.pretrained_dataset_id.isnumeric()
            else args.pretrained_dataset_id
        )

        (
            tta_data_dir,
            plan_dir,
            results_dir,
            pretrained_dataset_name,
            tta_dataset_name,
        ) = get_tta_folders(
            pretrained_dataset_id,
            int(args.tta_dataset_id),
            args.pretrainer,
            args.pretrainer_config,
            args.pretrainer_fold,
        )

        now_str = datetime.now().strftime("%Y%m%d__%H_%M_%S")
        numbers = [
            int(re.search(r"[0-9]+$", str(_path))[0]) for _path in results_dir.iterdir()
        ]
        if len(numbers) == 0:
            run_no = 0
        else:
            run_no = torch.as_tensor(numbers).max().item() + 1

        run_name = f"{now_str}_{randomname.get_name()}-{run_no}"

        with open(Path(plan_dir / "tta_plan.json"), "r") as f:
            config = json.load(f)

        with open(
            Path(plan_dir) / f"{pretrained_dataset_name}_label_mapping.json", "r"
        ) as f:
            pretrained_label_mapping = json.load(f)

        with open(Path(plan_dir) / f"{tta_dataset_name}_label_mapping.json", "r") as f:
            tta_dataset_label_mapping = json.load(f)

        label_mapping = generate_label_mapping(
            pretrained_label_mapping, tta_dataset_label_mapping
        )
        modifier_fn_module = load_current_modifier_functions(plan_dir)
        device = torch.device(args.device)

        kwargs = dict(
            run_name=run_name,
            config=config,
            tta_data_dir=tta_data_dir,
            save_base_path=results_dir,
            label_mapping=label_mapping,
            modifier_fn_module=modifier_fn_module,
            device=device,
        )

        if wandb_run_is_available():
            wandb_run("DG-TTA", tta_main, **kwargs)
            sys.exit(0)

        tta_main(**kwargs)


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ["--help", "-h"]:
        check_dga_root_is_set(soft_check=True)
    else:
        check_dga_root_is_set()
    DGTTAProgram()


if __name__ == "__main__":
    main()
