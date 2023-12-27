import re
from itertools import tee
from pathlib import Path
import importlib
import shutil
import json
import json

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from tqdm import trange, tqdm

if importlib.util.find_spec("wandb"):
    import wandb

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from dg_tta.utils import disable_internal_augmentation
from dg_tta.gin import gin_aug
from dg_tta.tta.torch_utils import (
    get_batch,
    map_label,
    dice_coeff,
    soft_dice_loss,
    fix_all,
    release_all,
    release_norms,
    get_map_idxs,
    get_imgs,
)
from dg_tta.tta.augmentation_utils import get_disp_field, get_rand_affine
from dg_tta.tta.config_log_utils import (
    get_global_idx,
    wandb_run_is_available,
    plot_run_results,
    get_parameters_save_path,
)
from dg_tta.tta.model_utils import (
    get_model_from_network,
    buffer_running_stats,
    apply_running_stats,
)
from dg_tta.tta.nnunet_utils import (
    run_inference,
    load_network,
    load_tta_data,
)

INTENSITY_AUG_FUNCTION_DICT = {"disabled": lambda img: img, "GIN": gin_aug}


def repair_ofilename_and_add_fileextension(config, sample):
    # Fixes removing too much of the filename when using underscores and numbers in names
    ofile = sample['ofile']
    for _path in config['tta_data_filepaths']:
        _path = Path(_path)
        ofile_prefix = ofile_name = '/'.join(ofile.split('/')[:-1])
        ofile_name = ofile.split('/')[-1]

        if ofile_name in str(_path):
            suffixes = ''.join(_path.suffixes)
            repaired_name = re.match(r"(.*)_[0-9]{4}"+suffixes, _path.name).groups(0)[0]
            sample['ofile'] = ofile_prefix + "/" + repaired_name
            sample['file_extension'] = suffixes


def get_sample_specs(config, smp_idx, tta_data, save_path, across_all_samples=False):
    if across_all_samples:
        tta_sample = None
        for e in tta_data:
            repair_ofilename_and_add_fileextension(config, e)
        tta_tens_list = [e["data"] for e in tta_data]
        sample_id = "all_samples"
        sample_extension = None
        sub_dir_tta = save_path / "tta_output"
    else:
        tta_sample = next(tta_data)
        # tta_sample = tta_data[smp_idx] # In case you don't use a generator
        repair_ofilename_and_add_fileextension(config, tta_sample)
        tta_tens_list = [tta_sample["data"]]
        sample_id = tta_sample["ofile"]
        sample_extension = tta_sample["file_extension"]
        sub_dir_tta = save_path / Path(sample_id).parent

    return tta_sample, tta_tens_list, sample_id, sample_extension, sub_dir_tta


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
    tta_across_all_samples = config["tta_across_all_samples"]

    tqdm.write("\n# Loading data")
    tta_data, num_samples = load_tta_data(
        config, tta_data_dir, predictor, tta_across_all_samples
    )

    if tta_across_all_samples:
        # TTA data is a list
        inference_data = tta_data
    else:
        # TTA data is a generator, so we need to copy it for inference
        tta_data, inference_data = tee(tta_data)

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

    tqdm.write("\n# Starting TTA")
    for smp_idx in sample_range:
        _, tta_tens_list, sample_id, sample_extension, sub_dir_tta = get_sample_specs(
            config, smp_idx, tta_data, save_path, tta_across_all_samples
        )
        tqdm.write(f"\nSample {sample_id}")

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
                                1 / tta_eval_patches * d_tgt_val.nanmean().item()
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

    print("\n\n# Starting inference")
    all_prediction_save_paths = []

    for smp_idx in trange(num_samples, desc="Samples"):
        ensemble_parameter_paths = []

        tta_sample, tta_tens_list, param_sample_id, sample_extension, sub_dir_tta = get_sample_specs(
            config, smp_idx, inference_data, save_path, across_all_samples=False
        )
        tqdm.write(f"\nSample {param_sample_id}\n")
        tta_sample["data"] = get_imgs(tta_sample["data"].unsqueeze(0)).squeeze(0)

        # Update internal save path for nnUNet
        ofile = tta_sample["ofile"]
        new_ofile = str(save_path / ofile)
        tta_sample["ofile"] = new_ofile

        prediction_save_path = Path(new_ofile + sample_extension)
        prediction_save_path.parent.mkdir(exist_ok=True)

        for ensemble_idx in range(config["ensemble_count"]):
            ensemble_parameter_paths.append(
                get_parameters_save_path(sub_dir_tta, param_sample_id, ensemble_idx)
            )

        disable_internal_augmentation()
        model = get_model_from_network(network, modifier_fn_module)

        predicted_output_array = run_inference(tta_sample, model, predictor, ensemble_parameter_paths)
        data_properties = tta_sample['data_properties']

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

    tqdm.write("\n\nEvaluating predictions")

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
