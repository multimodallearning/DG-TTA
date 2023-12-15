import sys
import os
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
from collections import defaultdict

import torch
from torch._dynamo import OptimizedModule
import torch.nn.functional as F

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

import randomname
if importlib.util.find_spec('wandb'):
    import wandb

from tqdm import trange,tqdm
import matplotlib.pyplot as plt

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

import dg_tta
from dg_tta.__build__ import inject_dg_trainers_into_nnunet
from dg_tta.utils import disable_internal_augmentation
from dg_tta.gin import gin_aug
from dg_tta.tta.torch_utils import load_batch_train, map_label, soft_dice_loss, fix_all, release_all, release_norms, get_module_data, set_module_data, register_forward_pre_hook_at_beginning, register_forward_hook_at_beginning, hookify, generate_label_mapping, get_map_idxs
from dg_tta.tta.augmentation_utils import get_disp_field, get_rand_affine
from dg_tta.tta.config_log_utils import wandb_run, load_current_modifier_functions, get_global_idx, get_tta_folders



PROJECT_NAME = 'nnunet_tta'
INTENSITY_AUG_FUNCTION_DICT = {
    'disabled': lambda img: img,
    'GIN': gin_aug
}



def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label - 1).fill_(0)

    for label_num in range(1, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2.0 * intersection) / (
            1e-8 + torch.mean(iflat) + torch.mean(tflat)
        )
    return dice



def load_tta_data(config, task_raw_path, predictor, fixed_sample_idx):
    # TODO: enable loading of differently sized images
    task_raw_path = Path(task_raw_path)

    folder_with_segs_from_prev_stage = task_raw_path / "labelsTs" # TODO make this a setting
    overwrite = True
    part_id = 0
    num_parts = 1
    save_probabilities = False
    num_processes_preprocessing = 1 # TODO make this a setting
    output_folder_or_list_of_truncated_output_files = ""

    list_of_lists_or_source_folder = Path(task_raw_path / "imagesTs")

    with open(task_raw_path / "dataset.json", 'r') as f:
        dataset_json = json.load(f)

    if fixed_sample_idx is not None:
        list_of_lists_or_source_folder = [[e] for e in list_of_lists_or_source_folder.glob(f"./*{dataset_json['file_ending']}")][fixed_sample_idx:fixed_sample_idx+1]
    else:
        list_of_lists_or_source_folder = str(list_of_lists_or_source_folder)

    (
        list_of_lists_or_source_folder,
        output_filename_truncated,
        seg_from_prev_stage_files,
    ) = predictor._manage_input_and_output_lists(
        list_of_lists_or_source_folder,
        output_folder_or_list_of_truncated_output_files,
        folder_with_segs_from_prev_stage,
        overwrite,
        part_id,
        num_parts,
        save_probabilities,
    )
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[0:1] # TODO remove
    nnUNetPredictor._internal_get_data_iterator_from_lists_of_filenames
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        num_processes_preprocessing
    )

    data = list(data_iterator)
    all_imgs = torch.stack([elem['data'][0:1] for elem in data])
    all_segs = torch.stack([elem['data'][1:] for elem in data])
    # Readd background channel (will contain zeros after argmax())
    all_segs = torch.cat([(all_segs.sum(1, keepdim=True)<1.).float(), all_segs], dim=1).argmax(1, keepdim=True)

    imgs_segs = torch.cat([all_imgs, all_segs], dim=1)

    # Update images of data dictionary
    for d_idx in range(len(data)):
        data[d_idx]['data'] = imgs_segs[d_idx:d_idx+1,0]
        # Segs are deleted here

    return imgs_segs, data



def load_network(weights_file):
    pretrained_weights_filepathpath = Path(*Path(weights_file).parts[:-2])
    fold = Path(weights_file).parts[-2].replace('fold_', '')
    use_folds = [int(fold)] if fold.isnumeric() else fold
    checkpoint_name = "checkpoint_final.pth"
    configuration = Path(weights_file).parts[-3].split('__')[-1]

    device = torch.device("cuda") # TODO make this configurable
    perform_everything_on_gpu = True
    verbose = False

    predictor = nnUNetPredictor(perform_everything_on_gpu=perform_everything_on_gpu, device=device, verbose_preprocessing=verbose)
    predictor.initialize_from_trained_model_folder(pretrained_weights_filepathpath, use_folds, checkpoint_name)

    parameters = predictor.list_of_parameters
    plans_manager = predictor.plans_manager
    network = predictor.network
    patch_size = plans_manager.get_configuration(configuration).patch_size

    return predictor, patch_size, network, parameters



def run_inference(tta_data, model, predictor, all_tta_parameter_paths):

    save_probabilities = False
    num_processes_segmentation_export = 1

    tta_parameters = []
    for _path in all_tta_parameter_paths:
        tta_parameters.extend(torch.load(_path))

    predictor.network = deepcopy(model)
    predictor.list_of_parameters = tta_parameters
    predictor.predict_from_data_iterator(tta_data, save_probabilities, num_processes_segmentation_export)



def prepare_mind_layers(model):
    # TODO rename this function
    # Prepare MIND model
    first_layer = get_module_data(model, 'encoder.stages.0.0.convs.0.conv')
    new_first_layer = torch.nn.Conv3d(12, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

    with torch.no_grad():
        new_first_layer.weight.copy_(first_layer.weight)
        new_first_layer.bias.copy_(first_layer.bias)

    set_module_data(model, 'encoder.stages.0.0.convs.0.conv', new_first_layer)
    set_module_data(model, 'encoder.stages.0.0.convs.0.all_modules.0', new_first_layer)

    return model



def get_model_from_network(network, parameters=None, modifier_fn_module=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    # Register hook that modifies the input prior to custom augmentation
    modify_tta_input_fn = modifier_fn_module.ModifierFunctions.modify_tta_input_fn
    register_forward_pre_hook_at_beginning(model, hookify(modify_tta_input_fn, 'forward_pre_hook'))

    # Register hook that modifies the output of the model
    modfify_tta_model_output_fn = modifier_fn_module.ModifierFunctions.modfify_tta_model_output_fn
    register_forward_hook_at_beginning(model, hookify(modfify_tta_model_output_fn, 'forward_hook'))

    return model

running_stats_buffer = {}

def buffer_running_stats(m):
    _id = id(m)
    if hasattr(m, 'running_mean') and hasattr(m, 'running_var') and not _id in running_stats_buffer:
        if m.running_mean is not None and m.running_var is not None:
            running_stats_buffer[_id] = [m.running_mean.data, m.running_var.data]

def apply_running_stats(m):
    _id = id(m)
    if hasattr(m, 'running_mean') and hasattr(m, 'running_var') and _id in running_stats_buffer:
        m.running_mean.data.copy_(other=running_stats_buffer[_id][0])
        m.running_var.data.copy_(other=running_stats_buffer[_id][1]) # Copy into .data to prevent backprop errors
        del running_stats_buffer[_id]



def get_parameters_save_path(save_path, ofile, ensemble_idx, train_on_all_test_samples):
    if train_on_all_test_samples:
        ofile = 'all_samples'
    tta_parameters_save_path = save_path / f"{ofile}__ensemble_idx_{ensemble_idx}_tta_parameters.pt"
    return tta_parameters_save_path



def tta_main(config, tta_data_dir, save_path, optimized_labels, train_test_label_mapping, modifier_fn_module, run_name=None, debug=False):
    # Load model
    pretrained_weights_filepath = config['pretrained_weights_filepath']
    predictor, patch_size, network, parameters = load_network(pretrained_weights_filepath)

    # Load TTA data
    tta_imgs_segs, tta_data = load_tta_data(config, tta_data_dir, predictor, config['fixed_sample_idx'])

    num_samples = tta_imgs_segs.shape[0]
    tta_across_all_samples=config['tta_across_all_samples']

    ensemble_count = config['ensemble_count']
    B = config['batch_size']
    patches_to_be_accumulated = config['patches_to_be_accumulated']
    tta_eval_patches = config['tta_eval_patches']
    num_epochs = config['epochs']
    start_tta_at_epoch = config['start_tta_at_epoch']

    device='cuda'

    if run_name is not None:
        save_path = Path(save_path) / run_name
    else:
        save_path = Path(save_path) / ("RUN__" + "__".join([f"{k}_{v}" for k,v in config.items()]))

    save_path.mkdir(exist_ok=True, parents=False)

    with open(save_path / 'tta_plan.json', 'w') as f:
        json.dump({k:v for k,v in config.items()}, f, indent=4)

    sitk_io = SimpleITKIO()

    zero_grid = torch.zeros([B] + patch_size + [3], device=device)
    identity_grid = F.affine_grid(torch.eye(4, device=device).repeat(B,1,1)[:,:3], [B, 1] + patch_size, align_corners=False)

    if tta_across_all_samples:
        sample_range = [0]
    else:
        sample_range = trange(num_samples, desc='sample')

    disable_internal_augmentation() # TODO find a better way do enable-disable internal trainer augmentation

    for smp_idx in sample_range:

        if tta_across_all_samples:
            tta_sample = tta_imgs_segs
            ofile = 'all_samples'
        else:
            tta_sample = tta_imgs_segs[smp_idx:smp_idx+1]
            ofile = tta_data[smp_idx]['ofile']

            if config['fixed_sample_idx'] is not None and smp_idx != config['fixed_sample_idx']:
                # Only train a specific sample
                continue

        tta_sample = tta_sample.to(device=device)

        ensemble_parameter_paths = []

        for ensemble_idx in trange(ensemble_count, desc='ensemble'):

            tta_parameters_save_path = get_parameters_save_path(save_path, ofile, ensemble_idx, tta_across_all_samples)
            if tta_parameters_save_path.is_file():
                tqdm.write(f"TTA parameters file already exists: {tta_parameters_save_path}")
                continue

            train_losses = torch.zeros(num_epochs)
            eval_dices = torch.zeros(num_epochs)

            intensity_aug_func = INTENSITY_AUG_FUNCTION_DICT[config['intensity_aug_function']]

            model = get_model_from_network(network, parameters, modifier_fn_module)
            model = model.to(device=device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

            tbar = trange(num_epochs, desc='epoch')
            START_CLASS = 1

            model.apply(fix_all)
            for epoch in tbar:
                model.train()
                global_idx = get_global_idx([(smp_idx,num_samples),(ensemble_idx, ensemble_count),(epoch, num_epochs)])
                if wandb.run is not None: wandb.log({"ref_epoch_idx": epoch}, global_idx)
                step_losses = []

                if epoch == start_tta_at_epoch:
                    tqdm.write("Starting train")
                    model.apply(fix_all)
                    if config['params_with_grad'] == 'all':
                        model.apply(release_all)
                    elif config['params_with_grad'] == 'norms':
                        model.apply(release_norms)
                    elif config['params_with_grad'] == 'encoder':
                        model.encoder.apply(release_all)
                    else:
                        raise ValueError()

                    grad_params = {id(p): p.numel() for p in model.parameters() if p.requires_grad}
                    tqdm.write(f"Released #{sum(list(grad_params.values()))/1e6:.2f} million trainable params")

                for _ in range(patches_to_be_accumulated):

                    with torch.no_grad():
                        imgs, labels = load_batch_train(
                            tta_sample, torch.randperm(tta_sample.shape[0])[:B], patch_size, affine_rand=0,
                            fixed_patch_idx=None
                        )
                    # TODO: split this into two function calls
                    # Augment branch a
                    grad_context_a = nullcontext if config['have_grad_in'] in ['branch_a', 'both'] else torch.no_grad
                    with grad_context_a():

                        imgs_aug_a = imgs

                        if config['do_intensity_aug_in'] in ['branch_a', 'both']:
                            imgs_aug_a = intensity_aug_func(imgs_aug_a)
                        else:
                            imgs_aug_a = imgs_aug_a

                        grid_a = zero_grid
                        grid_a_inverse = zero_grid

                        if config['spatial_aug_type'] == 'affine' and config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            R_a, R_a_inverse = get_rand_affine(B, flip=False)
                            R_a, R_a_inverse = R_a.to(device=device), R_a_inverse.to(device=device)
                            grid_a = grid_a + (F.affine_grid(R_a, [B, 1] + patch_size, align_corners=False) - identity_grid)
                            grid_a_inverse = grid_a_inverse + (F.affine_grid(R_a_inverse, [B, 1] + patch_size, align_corners=False) - identity_grid)

                        if config['spatial_aug_type'] == 'deformable' and config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            grid_a_deformable, grid_a_deformable_inverse = get_disp_field(B, patch_size, factor=0.5, interpolation_factor=5, device=device)
                            grid_a = grid_a + grid_a_deformable
                            grid_a_inverse = grid_a_inverse + grid_a_deformable_inverse

                        if config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            grid_a = grid_a + identity_grid
                            imgs_aug_a = F.grid_sample(imgs_aug_a, grid_a, padding_mode='border', align_corners=False)

                        model.apply(buffer_running_stats)
                        target_a = model(imgs_aug_a)
                        target_a = map_label(target_a, get_map_idxs(train_test_label_mapping, optimized_labels, input_type='train_labels'), input_format='logits')

                        if isinstance(target_a, tuple):
                            target_a = target_a[0]

                        if config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            grid_a_inverse = grid_a_inverse + identity_grid
                            target_a = F.grid_sample(target_a, grid_a_inverse, align_corners=False
                                # , padding_mode='border' # no padding mode border here?
                            )
                        else:
                            target_a = target_a

                    # Augment branch b
                    grad_context_b = nullcontext if config['have_grad_in'] in ['branch_b', 'both'] else torch.no_grad

                    with grad_context_b():
                        imgs_aug_b = imgs

                        if config['do_intensity_aug_in'] in ['branch_b', 'both']:
                            imgs_aug_b = intensity_aug_func(imgs_aug_b)
                        else:
                            imgs_aug_b = imgs_aug_b

                        grid_b = zero_grid
                        grid_b_inverse = zero_grid

                        if config['spatial_aug_type'] == 'affine' and config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            R_b, R_b_inverse = get_rand_affine(B, flip=False)
                            R_b, R_b_inverse = R_b.to(device=device), R_b_inverse.to(device=device)
                            grid_b = grid_b + (F.affine_grid(R_b, [B, 1] + patch_size, align_corners=False) - identity_grid)
                            grid_b_inverse = grid_b_inverse + (F.affine_grid(R_b_inverse, [B, 1] + patch_size, align_corners=False) - identity_grid)

                        if config['spatial_aug_type'] == 'deformable' and config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            grid_b_deformable, grid_b_deformable_inverse = get_disp_field(B, patch_size, factor=0.5, interpolation_factor=5, device=device)
                            grid_b = grid_b + grid_b_deformable
                            grid_b_inverse = grid_b_inverse + grid_b_deformable_inverse

                        if config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            grid_b = grid_b + identity_grid
                            imgs_aug_b = F.grid_sample(imgs_aug_b, grid_b, padding_mode='border', align_corners=False)

                        model.apply(apply_running_stats)
                        target_b = model(imgs_aug_b)
                        target_b = map_label(target_b, get_map_idxs(train_test_label_mapping, optimized_labels, input_type='train_labels'), input_format='logits')

                        if isinstance(target_b, tuple):
                            target_b = target_b[0]

                        if config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            grid_b_inverse = grid_b_inverse + identity_grid
                            target_b = F.grid_sample(target_b, grid_b_inverse, align_corners=False
                                # , padding_mode='border' # no padding mode border here?
                            )
                        else:
                            target_b = target_b

                    # Apply consistency loss
                    common_content_mask = (target_a.sum(1, keepdim=True) > 0.).float() * (target_b.sum(1, keepdim=True) > 0.).float()
                    sm_a = target_a.softmax(1) * common_content_mask
                    sm_b = target_b.softmax(1) * common_content_mask

                    loss = 1 - soft_dice_loss(sm_a, sm_b)[:,START_CLASS:].mean()

                    loss_accum = loss / patches_to_be_accumulated
                    step_losses.append(loss.detach().cpu())

                    if epoch >= start_tta_at_epoch:
                        loss_accum.backward()

                if epoch >= start_tta_at_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

                train_losses[epoch] = torch.stack(step_losses).mean().item()

                with torch.inference_mode():
                    model.eval()
                    for _ in range(tta_eval_patches):
                        imgs, labels = load_batch_train(
                            tta_sample, torch.randperm(tta_sample.shape[0])[:B], patch_size, affine_rand=0,
                            fixed_patch_idx='center', # This is just for evaluation purposes
                        )

                        output_eval = model(imgs)
                        if isinstance(output_eval, tuple):
                            output_eval = output_eval[0]

                        output_eval = map_label(output_eval, get_map_idxs(train_test_label_mapping, optimized_labels, input_type='train_labels'), input_format='logits')
                        target_argmax = output_eval.argmax(1)

                        labels = map_label(labels, get_map_idxs(train_test_label_mapping, optimized_labels, input_type='test_labels'), input_format='argmaxed').long()
                        d_tgt_val = dice_coeff(
                            target_argmax, labels, len(optimized_labels)
                        )

                        eval_dices[epoch] += 1/tta_eval_patches * d_tgt_val.mean().item()

                    if debug: break

                tbar.set_description(f"epoch, loss = {train_losses[epoch]:.3f}, dice = {eval_dices[epoch]:.2f}")
                if wandb.run is not None: wandb.log({f'losses/loss__{ofile}__ensemble_idx_{ensemble_idx}': train_losses[epoch]}, step=global_idx)
                if wandb.run is not None: wandb.log({f'scores/eval_dice__{ofile}__ensemble_idx_{ensemble_idx}': eval_dices[epoch]}, step=global_idx)

            # Print graphic per ensemble
            # TODO: Externalises / improve the plotting
            plt.close()
            plt.cla()
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(2., 2.))
            # ax2 = ax1.twinx()
            ax1.set_ylim(0., 1.)

            ax1.plot(train_losses, label='loss')
            # ax2.plot(alt_train_losses, label='alt_loss', color='red')
            ax1.plot(eval_dices, label='eval_dices')
            ax1.legend()
            # ax2.legend()
            plt.tight_layout()

            tta_parameters = [model.state_dict()]
            tta_plot_save_path = save_path / f"{ofile}__ensemble_idx_{ensemble_idx}_tta_results.png"
            plt.savefig(tta_plot_save_path)
            plt.close()

            torch.save(tta_parameters, tta_parameters_save_path)

            if debug: break
        # End of ensemble loop


    print("Starting prediction")
    all_prediction_save_paths = []

    for smp_idx in trange(num_samples, desc='sample'):
        ensemble_parameter_paths = []

        tta_sample = tta_imgs_segs[smp_idx:smp_idx+1]
        ofile = tta_data[smp_idx]['ofile']
        new_ofile = str(save_path / ofile)
        tta_data[smp_idx]['ofile'] = new_ofile
        # ofilename = ofile + ".nii.gz"

        for ensemble_idx in range(config['ensemble_count']):
            ensemble_parameter_paths.append(get_parameters_save_path(save_path, ofile, ensemble_idx, tta_across_all_samples))

        if config['fixed_sample_idx'] is not None and smp_idx != config['fixed_sample_idx']:
            # Only train a specific sample
            continue

        # Save prediction
        prediction_save_path = new_ofile + ".nii.gz"

        disable_internal_augmentation()
        model = get_model_from_network(network)

        run_inference(tta_data[smp_idx:smp_idx+1], model, predictor, ensemble_parameter_paths)

        predicted_output_array, data_properties = sitk_io.read_seg(prediction_save_path)
        predicted_output = map_label(
            torch.as_tensor(predicted_output_array),
            get_map_idxs(train_test_label_mapping, optimized_labels, input_type='train_labels'),
            input_format='argmaxed'
        ).squeeze(0)

        sitk_io.write_seg(predicted_output.numpy(), prediction_save_path, properties=data_properties)
        all_prediction_save_paths.append(prediction_save_path)

    # End of sample loop
    gt_labels_path = tta_data_dir / "labelsTs"

    tqdm.write('Evaluating predictions...')

    path_for_mapped_targets = save_path / 'mapped_targets'
    path_for_mapped_targets.mkdir(exist_ok=True, parents=False)

    image_reader_writer = SimpleITKIO()

    for pred_path in all_prediction_save_paths:
        filename = Path(pred_path).name
        src_path = Path(gt_labels_path) / filename
        copied_label_path = path_for_mapped_targets / filename
        shutil.copy(src_path, copied_label_path)

        seg, sitk_stuff = image_reader_writer.read_seg(copied_label_path)
        seg = torch.as_tensor(seg)
        mapped_seg = map_label(seg, get_map_idxs(train_test_label_mapping, optimized_labels, input_type='test_labels'), input_format='argmaxed').squeeze(0)
        image_reader_writer.write_seg(mapped_seg.squeeze(0).numpy(), copied_label_path, sitk_stuff)

    with open(save_path / 'optimized_labels.json', 'w') as f:
        json.dump({v:k for k,v in enumerate(optimized_labels)}, f, indent=4)

    # Run postprocessing
    postprocess_results_fn = modifier_fn_module.ModifierFunctions.postprocess_results_fn
    postprocess_results_fn(save_path)

    compute_metrics_on_folder_simple(
        folder_ref=path_for_mapped_targets, folder_pred=save_path,
        labels=list(range(len(optimized_labels))),
        num_processes=1, chill=True)

    with open(save_path / 'summary.json', 'r') as f:
        summary_json = json.load(f)
        final_mean_dice = summary_json["foreground_mean"]["Dice"]

    if wandb.run is not None: wandb.log({f'scores/tta_dice_mean': final_mean_dice})



class DGTTAProgram():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='DG-TTA for nnUNetv2',
            usage='''dgtta <command> [<args>]

        Commands are:
        pretrain        Pretrain an nnUNet model with DG trainers
        prepare_tta     Prepare test-time adaptation
        run_tta         Run test-time adaptation
        ''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def pretrain(self):
        parser = argparse.ArgumentParser(
            description='Run pretraining with DG-TTA trainers')
        parser.add_argument('--amend', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        raise NotImplementedError()
        print(f'Running git commit, amend={args.ammend}')

    def prepare_tta(self):
        parser = argparse.ArgumentParser(description='Prepare DG-TTA',
                                         usage='''dgtta prepare_tta [-h]''')
        # TODO change term task to dataset
        parser.add_argument('pretrained_task_id', help='''
                            Task ID for pretrained model.
                            Can be numeric or one of ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND']''')
        parser.add_argument('tta_task_id', help='Task ID for TTA')
        parser.add_argument('--pretrainer', help='Trainer to use for pretraining', default='nnUNetTrainer_GIN_MIND')
        parser.add_argument('--pretrainer_config', help='Fold ID of nnUNet model to use for pretraining', default='3d_fullres')
        parser.add_argument('--pretrainer_fold', help='Fold ID of nnUNet model to use for pretraining', default='0')

        args = parser.parse_args(sys.argv[2:])
        pretrained_task_id = int(args.pretrained_task_id) \
            if args.pretrained_task_id.isnumeric() else args.pretrained_task_id
        pretrainer_fold = int(args.pretrainer_fold) \
            if args.pretrainer_fold.isnumeric() else args.pretrainer_fold

        dg_tta.tta.config_log_utils.prepare_tta(pretrained_task_id, int(args.tta_task_id),
                                                pretrainer=args.pretrainer,
                                                pretrainer_config=args.pretrainer_config,
                                                pretrainer_fold=pretrainer_fold)

    def run_tta(self):
        parser = argparse.ArgumentParser(description='Run DG-TTA')
        parser.add_argument('pretrained_task_id', help='''
                            Task ID for pretrained model.
                            Can be numeric or one of ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND']''')
        parser.add_argument('tta_task_id', help='Task ID for TTA')
        parser.add_argument('--pretrainer', help='Trainer to use for pretraining', default='nnUNetTrainer_GIN_MIND')
        parser.add_argument('--pretrainer_config', help='Fold ID of nnUNet model to use for pretraining', default='3d_fullres')
        parser.add_argument('--pretrainer_fold', help='Fold ID of nnUNet model to use for pretraining', default='0')

        args = parser.parse_args(sys.argv[2:])
        pretrained_task_id = int(args.pretrained_task_id) \
            if args.pretrained_task_id.isnumeric() else args.pretrained_task_id

        tta_data_dir, plan_dir, results_dir, pretrained_task_name, tta_task_name = \
            get_tta_folders(pretrained_task_id, int(args.tta_task_id), \
                            args.pretrainer, args.pretrainer_config, args.pretrainer_fold)

        now_str = datetime.now().strftime("%Y%m%d__%H_%M_%S")
        numbers = [int(re.search(r'[0-9]+$', str(_path))[0]) for _path in results_dir.iterdir()]
        if len(numbers) == 0:
            run_no = 0
        else:
            run_no = torch.as_tensor(numbers).max().item() + 1

        run_name = f"{now_str}_{randomname.get_name()}-{run_no}"

        with open(Path(plan_dir / 'tta_plan.json'), 'r') as f:
            config = json.load(f)

        with open(Path(plan_dir) / 'optimized_labels.json', 'r') as f:
            optimized_labels = json.load(f)

        with open(Path(plan_dir) / f"{pretrained_task_name}_label_mapping.json", 'r') as f:
            pretrained_label_mapping = json.load(f)

        with open(Path(plan_dir) / f"{tta_task_name}_label_mapping.json", 'r') as f:
            tta_task_label_mapping = json.load(f)

        label_mapping = generate_label_mapping(pretrained_label_mapping, tta_task_label_mapping)

        modifier_fn_module = load_current_modifier_functions(plan_dir)
        tta_main(config, tta_data_dir, results_dir, optimized_labels, label_mapping, modifier_fn_module, run_name=run_name, debug=False)



def main():
    assert Path(os.environ.get('DG_TTA_ROOT', '_')).is_dir(), \
        "Please define an existing root directory for DG-TTA by setting DG_TTA_ROOT."
    inject_dg_trainers_into_nnunet()
    DGTTAProgram()



if __name__ == "__main__":
    main()