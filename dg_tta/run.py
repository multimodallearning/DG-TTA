
# %%

NNUNET_BASE_DIR = ''
import os
import os
from copy import deepcopy
from torch._dynamo import OptimizedModule
import json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
from datetime import datetime
import importlib
if importlib.util.find_spec('wandb'):
    import wandb
import shutil
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnunetv2.inference.predict_from_raw_data import (
#     _manage_input_and_output_lists,
#     _internal_get_data_iterator_from_lists_of_filenames,
# )
import json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from pathlib import Path
from torch._dynamo import OptimizedModule
from pathlib import Path

from tqdm import trange,tqdm
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
import nibabel as nib

import numpy as np
from contextlib import nullcontext
from pathlib import Path
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from collections import defaultdict
from dg_tta.tta.torch_utils import load_batch_train
from dg_tta.tta.torch_utils import soft_dice_loss
from dg_tta.tta.torch_utils import fix_all, release_all, release_norms
from dg_tta.tta.augmentation_utils import get_disp_field
from dg_tta.mind import MIND3D
from dg_tta.tta.torch_utils import get_module_data, set_module_data
from dg_tta.tta.augmentation_utils import disable_internal_augmentation, check_internal_augmentation_disabled, get_rand_affine
from dg_tta.tta.augmentation_utils import gin_aug, GinMINDAug
import dg_tta
from dg_tta.tta.config_log_utils import wandb_run
import shutil
import argparse


# %%
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


# %%
def load_tta_data(task_raw_path, configuration_manager, plans_manager, dataset_json, fixed_sample_idx):
    # TODO: enable loading of differently sized images
    task_raw_path = Path(task_raw_path)

    folder_with_segs_from_prev_stage = task_raw_path / "labelsTs"
    overwrite = True
    part_id = 0
    num_parts = 1
    save_probabilities = False
    num_processes_preprocessing = 0
    device = torch.device("cuda")
    verbose = False
    output_folder_or_list_of_truncated_output_files = ""
    FILE_ENDING = ".nii.gz"

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
    ) = nnUNetPredictor._manage_input_and_output_lists(
        list_of_lists_or_source_folder,
        output_folder_or_list_of_truncated_output_files,
        dataset_json,
        folder_with_segs_from_prev_stage,
        overwrite,
        part_id,
        num_parts,
        save_probabilities,
    )

    data_iterator = nnUNetPredictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        configuration_manager,
        plans_manager,
        dataset_json,
        num_processes_preprocessing,
        pin_memory=device.type == "cpu",
        verbose=verbose,
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



def load_model(model_training_output_path, fold):

    use_folds = [fold] if isinstance(fold, int) else fold # We only trained one fold
    checkpoint_name = "checkpoint_final.pth"

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        nnUNetPredictor.initialize_from_trained_model_folder(model_training_output_path, use_folds, checkpoint_name)

    patch_size = plans_manager.get_configuration('3d_fullres').patch_size # TODO: make configuration a setting

    return network, parameters, patch_size, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json



def run_inference(tta_data, network, all_tta_parameter_paths,
                      plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
                      device='cpu'):

    tile_step_size = 0.5
    use_gaussian = True
    use_mirroring = False
    save_probabilities = False
    verbose = False
    perform_everything_on_gpu = True
    num_processes_segmentation_export = 1

    tta_parameters = []
    for _path in all_tta_parameter_paths:
        tta_parameters.extend(torch.load(_path))

    network = deepcopy(network)
    return nnUNetPredictor.predict_from_data_iterator(tta_data, network, tta_parameters, plans_manager, configuration_manager, dataset_json,
                                inference_allowed_mirroring_axes, tile_step_size, use_gaussian, use_mirroring,
                                perform_everything_on_gpu, verbose, save_probabilities,
                                num_processes_segmentation_export, device)


# %% [markdown]
# # Pre/Postprocessing methods

# %%


# %%
# TODO find a way to have these functions in a separate definition file
def ts_amos_pre_forward_hook_fn(input_data):
    if isinstance(input_data, tuple):
        input_data = input_data[0]
    input_data = input_data.flip(-1)
    return input_data

def ts_amos_forward_hook_fn(output_data):
    if isinstance(output_data, tuple):
        output_data = output_data[0]
    output_data = output_data.flip(-1)
    return output_data

def ts_mmwhs_pre_forward_hook_fn(input_data):
    if isinstance(input_data, tuple):
        input_data = input_data[0]
    input_data = input_data.transpose(-2,-3).flip([-2,-1]) # works! MMWHS to TS transform
    return input_data

def ts_mmwhs_forward_hook_fn(output_data):
    if isinstance(output_data, tuple):
        output_data = output_data[0]
    output_data = output_data.flip([-2,-1]).transpose(-2,-3) # works! TS to MMWHS transform
    return output_data

def ts_myo_spine_pre_forward_hook_fn(input_data):
    if isinstance(input_data, tuple):
        input_data = input_data[0]
    input_data = input_data.flip([-2,-3])
    return input_data

def ts_myo_spine_forward_hook_fn(output_data):
    if isinstance(output_data, tuple):
        output_data = output_data[0]
    output_data = output_data.flip([-2,-3])
    return output_data

def ts_myo_spine_postprocessing(output_folder):
    DILATION_SIZE = 9

    output_folder = Path(output_folder)
    all_target_paths = (output_folder / "mapped_targets").glob("*nii.gz")
    all_prediction_paths = (output_folder).glob("*nii.gz")

    kernel = torch.ones(1,1,DILATION_SIZE,DILATION_SIZE,DILATION_SIZE)

    for target_path, prediction_path in zip(all_target_paths, all_prediction_paths):
        target_data = torch.as_tensor(nib.load(target_path).get_fdata())
        prediction_nii = nib.load(prediction_path)
        prediction_data = torch.as_tensor(prediction_nii.get_fdata())

        dilated_target = torch.nn.functional.conv3d(target_data[None,None,:],kernel.to(target_data.dtype), padding=DILATION_SIZE//2).squeeze()
        dilated_target_mask = (dilated_target > 0).float()

        masked_prediction = prediction_data * dilated_target_mask
        nib.save(nib.Nifti1Image(masked_prediction.int().numpy(),
                                 affine=prediction_nii.affine), prediction_path)



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


# %%
def generate_label_mapping(source_label_dict, target_label_dict):
    # TODO permit user definition an alternative mapping dict per task
    assert all([isinstance(k, str) for k in source_label_dict.keys()])
    assert all([isinstance(k, str) for k in target_label_dict.keys()])
    assert set(source_label_dict.keys()).intersection(target_label_dict.keys()), "There are no intersecting label names in given dicts."
    mapped_label = []

    mapping_dict = dict.fromkeys(list(source_label_dict.keys()) + list(target_label_dict.keys()))

    for key in mapping_dict:
        if key in source_label_dict and key in target_label_dict:
            mapping_dict[key] = (source_label_dict[key], target_label_dict[key])

    return {k:v for k,v in mapping_dict.items() if v is not None}

def get_map_idxs(label_mapping: dict, evaluated_labels: list, input_type):
    assert input_type in ['train_labels', 'test_labels']
    assert evaluated_labels[0] == 'background'

    # Generate idxs from label_mapping dict
    map_idxs_list = []
    for reduced_idx, eval_label in enumerate(evaluated_labels):
        src_idx, target_idx = label_mapping[eval_label]
        # map_idxs_list = [tts_dict[k] for k,v in amos_bcv_dict.items()]
        append_idx = src_idx if input_type == 'train_labels' else target_idx
        map_idxs_list.append(append_idx)

    map_idxs = torch.as_tensor(map_idxs_list)

    return map_idxs

def map_label(label, map_idxs, input_format):
    assert input_format in ['logits', 'argmaxed']

    if input_format == 'logits':
        # We have a non argmaxed map, suppose that C dimension is label dimension
        mapped_label = label
        # Swap B,C and subselect
        mapped_label = mapped_label.transpose(0,1)[map_idxs].transpose(0,1)
    else:
        mapped_label = torch.zeros_like(label)
        for lbl_idx, map_idx in enumerate(map_idxs):
            mapped_label[label == map_idx] = lbl_idx

    return mapped_label

# %% [markdown]
# # Define settings

# %% [markdown]
# ## Dataset dicts (changed names to get automatic mapping right)

# %%
# TODO remove these hardcoded dicts
amos_bcv_dict = {
    "background": 0,
    "spleen": 1,
    "right_kidney": 2,
    "left_kidney": 3,
    "gallbladder": 4,
    "esophagus": 5,
    "liver": 6,
    "stomach": 7,
    "aorta": 8,
    "inferior_vena_cava": 9,
    "pancreas": 10,
    "right_adrenal_gland": 11,
    "left_adrenal_gland": 12
}

tts_dict = {
    "background": 0,
    "spleen": 1,
    "right_kidney": 2,
    "left_kidney": 3,
    "gallbladder": 4,
    "esophagus": 42,
    "liver": 5,
    "stomach": 6,
    "aorta": 7,
    "inferior_vena_cava": 8,
    "pancreas": 10,
    "right_adrenal_gland": 11,
    "left_adrenal_gland": 12,

    "aorta": 7,
    "inferior_vena_cava": 8,
    "portal_vein_and_splenic_vein": 9,
    "left_myocardium": 44,
    "left_atrium": 45,
    "left_ventricle": 46,
    "right_atrium": 47,
    "right_ventricle": 48,
    "pulmonary_artery": 49,

    "L1": 22,
    "L2": 21,
    "L3": 20,
    "L4": 19,
    "L5": 18,
}

mmwhs_dict = {
    "background": 0,
    "left_myocardium": 1,
    "left_atrium": 2,
    "left_ventricle": 3,
    "right_atrium": 4,
    "right_ventricle": 5,
    "aorta": 6,
    "pulmonary_artery": 7
}

myo_spine_dict = dict(
    background=0,
    L1=1,
    L2=2,
    L3=3,
    L4=4,
    L5=5,
)

dataset_labels_dict = dict(
    BCV=amos_bcv_dict,
    AMOS=amos_bcv_dict,
    TotalSegmentator=tts_dict,
    MMWHS=mmwhs_dict,
    MMWHS_CT=mmwhs_dict,
    MyoSegmenTUM_spine=myo_spine_dict
)

# %%

# TODO remove these hardcoded definitions
EVALUATED_LABELS_DICT = {
    'AMOS->BCV': ["background", "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava", "pancreas"],
    'BCV->AMOS': ["background", "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava", "pancreas"],
    'TotalSegmentator->AMOS': ["background", "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava", "pancreas"],
    'TotalSegmentator->BCV': ["background", "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava", "pancreas"],
    'TotalSegmentator->MMWHS': [
        'background',
        # 'aorta',
        'left_myocardium', 'left_atrium', 'left_ventricle', 'right_atrium', 'right_ventricle',
        # 'pulmonary_artery'
    ],
    'MMWHS_CT->MMWHS': [
        'background',
        # 'aorta',
        'left_myocardium', 'left_atrium', 'left_ventricle', 'right_atrium', 'right_ventricle',
        # 'pulmonary_artery'
    ],
    'TotalSegmentator->MyoSegmenTUM_spine': ["background", "L1", "L2", "L3", "L4", "L5"],
}

# TODO make config standard and exposable
CONFIG_DICT = dict(
    train_data='BCV',
    fold='0',
    data_postload_fn=None,
    trainer='GIN+MIND',
    tta_data='AMOS',
    intensity_aug_function='GIN',
    train_on_all_test_samples=False,
    params_with_grad='all', # all, norms, encoder
    lr=1e-5,
    fixed_sample_idx=None,
    ensemble_count=3,
    epochs=12,

    have_grad_in='branch_a', # ['branch_a', 'branch_b', 'both']
    do_intensity_aug_in='none', # ['branch_a', 'branch_b', 'both', 'none']
    do_spatial_aug_in='both', # ['branch_a', 'branch_b', 'both', 'none']
    spatial_aug_type='affine', # ['affine', 'deformable']

    wandb_mode='online',
)

# Remove hardcoded values
intensity_aug_function_dict = {
    'NNUNET': lambda img: img,
    'GIN': gin_aug,
    'MIND': MIND3D(),
    'GIN+MIND': GinMINDAug(),
}

trainer_dict = {
    'NNUNET': 'nnUNetTrainer__nnUNetPlans__3d_fullres',
    'NNUNET_BN': 'nnUNetTrainerBN__nnUNetPlans__3d_fullres',
    'GIN': 'nnUNetTrainer_GIN__nnUNetPlans__3d_fullres',
    'GIN_INSANE': 'nnUNetTrainer_GIN_INSANE__nnUNetPlans__3d_fullres',
    'MIND': 'nnUNetTrainer_MIND__nnUNetPlans__3d_fullres',
    'GIN+MIND': 'nnUNetTrainer_GIN_MIND__nnUNetPlans__3d_fullres',
    'InsaneDA': 'nnUNetTrainer_insaneDA__nnUNetPlans__3d_fullres',
}

base_data_dict = dict(
    AMOS = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset803_AMOS_w_gallbladder",
    BCV = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset804_BCV_w_gallbladder",
    TotalSegmentator = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset505_TS104",
    MMWHS = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset656_MMWHS_RESAMPLE_ONLY",
    MMWHS_CT = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset657_MMWHS_CT_RESAMPLE_ONLY",
    MyoSegmenTUM_spine = NNUNET_BASE_DIR + "nnUNetV2_raw/Dataset810_MyoSegmenTUM_spine"
)

postprocessing_function_dict = defaultdict(lambda: lambda output_dir: output_dir)
postprocessing_function_dict['TotalSegmentator->MyoSegmenTUM_spine'] = ts_myo_spine_postprocessing

additional_model_pre_forward_hook_dict = defaultdict(lambda: lambda data: data)
additional_model_pre_forward_hook_dict['TotalSegmentator->AMOS'] = ts_amos_pre_forward_hook_fn
additional_model_pre_forward_hook_dict['TotalSegmentator->MMWHS'] = ts_mmwhs_pre_forward_hook_fn
additional_model_pre_forward_hook_dict['TotalSegmentator->MyoSegmenTUM_spine'] = ts_myo_spine_pre_forward_hook_fn

additional_model_forward_hook_dict = defaultdict(lambda: lambda data: data)
additional_model_forward_hook_dict['TotalSegmentator->AMOS'] = ts_amos_forward_hook_fn
additional_model_forward_hook_dict['TotalSegmentator->MMWHS'] = ts_mmwhs_forward_hook_fn
additional_model_forward_hook_dict['TotalSegmentator->MyoSegmenTUM_spine'] = ts_myo_spine_forward_hook_fn

train_data_model_dict = dict(
    BCV = NNUNET_BASE_DIR + "nnUNetV2_results/Dataset804_BCV_w_gallbladder",
    TotalSegmentator = NNUNET_BASE_DIR + "nnUNetV2_results/Dataset505_TS104",
    MMWHS_CT = NNUNET_BASE_DIR + "nnUNetV2_results/Dataset657_MMWHS_CT_RESAMPLE_ONLY",
)



def get_model_from_network(config, network, parameters=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    model._forward_pre_hooks.clear()

    # Register hook if found, otherwise lambda e:e is used
    model.register_forward_pre_hook(lambda _, input: additional_model_pre_forward_hook_dict[config['train_tta_data_map']](input))

    check_internal_augmentation_disabled()

    if 'mind' in config['trainer'].lower():
        assert 'mind' in config['intensity_aug_function'].lower()

        def hook(module, input):
            input = input[0]
            return intensity_aug_function_dict[config['intensity_aug_function']](input)

        model.register_forward_pre_hook(hook)

    elif 'mind' in config['intensity_aug_function'].lower():
        # Prepare mind layers for a normal model
        prepare_mind_layers(model)

        def hook(module, input):
            input = input[0]
            return intensity_aug_function_dict[config['intensity_aug_function']](input)

        model.register_forward_pre_hook(hook)

    model.register_forward_hook(
        lambda model, input, output: \
            additional_model_forward_hook_dict[config['train_tta_data_map']](output))

    return model

def get_global_idx(list_of_tuple_idx_max):
    # Get global index e.g. 2250 for ensemble_idx=2, epoch_idx=250 @ max_epochs<1000
    global_idx = 0
    next_multiplier = 1

    # Smallest identifier tuple last!
    for idx, max_of_idx in reversed(list_of_tuple_idx_max):
        global_idx = global_idx + next_multiplier * idx
        next_multiplier = next_multiplier * 10**len(str(int(max_of_idx)))
    return global_idx

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



def tta_main(config, save_path, evaluated_labels, train_test_label_mapping, run_name=None, debug=False):

    # Load model
    base_models_path = Path(train_data_model_dict[config['train_data']])
    model_training_output_path = base_models_path / trainer_dict[config['trainer']]
    network, parameters, patch_size, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json = load_model(model_training_output_path, config['fold'])

    # Load TTA data
    target_data_path = base_data_dict[config['tta_data']]
    tta_imgs_segs, tta_data = load_tta_data(target_data_path, configuration_manager, plans_manager, dataset_json, config['fixed_sample_idx'])


    ensemble_count = config['ensemble_count']
    num_samples = tta_imgs_segs.shape[0]

    # TODO make these parameters configurable
    B = 1
    ACCUM = 16
    train_start_epoch = 1
    num_epochs = config['epochs']
    num_eval_patches = 1

    device='cuda'

    if run_name is not None:
        save_path = Path(save_path) / run_name
    else:
        save_path = Path(save_path) / ("RUN__" + "__".join([f"{k}_{v}" for k,v in config.items()]))

    save_path.mkdir(exist_ok=True, parents=False)

    with open(save_path / 'config.json', 'w') as f:
        json.dump({k:v for k,v in config.items()}, f, indent=4)

    sitk_io = SimpleITKIO()

    zero_grid = torch.zeros([B] + patch_size + [3], device=device)
    identity_grid = F.affine_grid(torch.eye(4, device=device).repeat(B,1,1)[:,:3], [B, 1] + patch_size)

    if config['train_on_all_test_samples']:
        sample_range = [0]
    else:
        sample_range = trange(num_samples, desc='sample')

    disable_internal_augmentation() # TODO find a better way do enable-disable internal trainer augmentation

    for smp_idx in sample_range:

        if config['train_on_all_test_samples']:
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

            tta_parameters_save_path = get_parameters_save_path(save_path, ofile, ensemble_idx, config['train_on_all_test_samples'])
            if tta_parameters_save_path.is_file():
                tqdm.write(f"TTA parameters file already exists: {tta_parameters_save_path}")
                continue

            train_losses = torch.zeros(num_epochs)
            eval_dices = torch.zeros(num_epochs)

            if 'mind' in config['trainer'].lower():
                intensity_aug_func = lambda imgs: imgs

            elif 'mind' in config['intensity_aug_function'].lower():
                intensity_aug_func = lambda imgs: imgs
            else:
                intensity_aug_func = intensity_aug_function_dict[config['intensity_aug_function']]

            model = get_model_from_network(config, network, parameters)
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

                if epoch == train_start_epoch:
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

                for _ in range(ACCUM):

                    with torch.no_grad():
                        imgs, labels, _ = load_batch_train(
                            tta_sample, torch.randperm(tta_sample.shape[0])[:B], patch_size, affine_rand=0, use_rf=False,
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
                            grid_a = grid_a + (F.affine_grid(R_a, [B, 1] + patch_size) - identity_grid)
                            grid_a_inverse = grid_a_inverse + (F.affine_grid(R_a_inverse, [B, 1] + patch_size) - identity_grid)

                        if config['spatial_aug_type'] == 'deformable' and config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            grid_a_deformable, grid_a_deformable_inverse = get_disp_field(B, patch_size, factor=0.5, interpolation_factor=5, device=device)
                            grid_a = grid_a + grid_a_deformable
                            grid_a_inverse = grid_a_inverse + grid_a_deformable_inverse

                        if config['do_spatial_aug_in'] in ['branch_a', 'both']:
                            grid_a = grid_a + identity_grid
                            imgs_aug_a = F.grid_sample(imgs_aug_a, grid_a, padding_mode='border', align_corners=False)

                        model.apply(buffer_running_stats)
                        target_a = model(imgs_aug_a)
                        target_a = map_label(target_a, get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='train_labels'), input_format='logits')

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
                            grid_b = grid_b + (F.affine_grid(R_b, [B, 1] + patch_size) - identity_grid)
                            grid_b_inverse = grid_b_inverse + (F.affine_grid(R_b_inverse, [B, 1] + patch_size) - identity_grid)

                        if config['spatial_aug_type'] == 'deformable' and config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            grid_b_deformable, grid_b_deformable_inverse = get_disp_field(B, patch_size, factor=0.5, interpolation_factor=5, device=device)
                            grid_b = grid_b + grid_b_deformable
                            grid_b_inverse = grid_b_inverse + grid_b_deformable_inverse

                        if config['do_spatial_aug_in'] in ['branch_b', 'both']:
                            grid_b = grid_b + identity_grid
                            imgs_aug_b = F.grid_sample(imgs_aug_b, grid_b, padding_mode='border', align_corners=False)

                        model.apply(apply_running_stats)
                        target_b = model(imgs_aug_b)
                        target_b = map_label(target_b, get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='train_labels'), input_format='logits')

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

                    loss_accum = loss / ACCUM
                    step_losses.append(loss.detach().cpu())

                    if epoch >= train_start_epoch:
                        loss_accum.backward()

                if epoch >= train_start_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

                train_losses[epoch] = torch.stack(step_losses).mean().item()

                with torch.inference_mode():
                    model.eval()
                    for _ in range(num_eval_patches):
                        imgs, labels, _ = load_batch_train(
                            tta_sample, torch.randperm(tta_sample.shape[0])[:B], patch_size, affine_rand=0,
                            fixed_patch_idx='center', # This is just for evaluation purposes
                        )

                        output_eval = model(imgs)
                        if isinstance(output_eval, tuple):
                            output_eval = output_eval[0]

                        output_eval = map_label(output_eval, get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='train_labels'), input_format='logits')
                        target_argmax = output_eval.argmax(1)

                        labels = map_label(labels, get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='test_labels'), input_format='argmaxed').long()
                        d_tgt_val = dice_coeff(
                            target_argmax, labels, len(evaluated_labels)
                        )
                        # # Omit adrenal glands
                        # d_tgt_val = d_tgt_val[1:BCV_AMOS_NUM_CLASSES]

                        eval_dices[epoch] += 1/num_eval_patches * d_tgt_val.mean().item()

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
        ofilename = ofile + ".nii.gz"

        for ensemble_idx in range(config['ensemble_count']):
            ensemble_parameter_paths.append(get_parameters_save_path(save_path, ofile, ensemble_idx, config['train_on_all_test_samples']))

        if config['fixed_sample_idx'] is not None and smp_idx != config['fixed_sample_idx']:
            # Only train a specific sample
            continue

        # Save prediction
        prediction_save_path = save_path / ofilename

        disable_internal_augmentation()
        model = get_model_from_network(config, network, None)

        predicted_output = run_inference(tta_data[smp_idx:smp_idx+1], model, ensemble_parameter_paths,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
            device=torch.device(device))

        predicted_output = map_label(torch.as_tensor(predicted_output[0]), get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='train_labels'), input_format='argmaxed').squeeze(0)
        sitk_io.write_seg(predicted_output.numpy(), prediction_save_path, properties=tta_data[smp_idx]['data_properites'])

        all_prediction_save_paths.append(prediction_save_path)

    # End of sample loop
    gt_labels_path = Path(base_data_dict[config['tta_data']]) / "labelsTs"

    tqdm.write('Evaluating predictions...')

    path_for_mapped_targets = save_path / 'mapped_targets'
    path_for_mapped_targets.mkdir(exist_ok=True, parents=False)

    image_reader_writer = SimpleITKIO()

    for pred_path in all_prediction_save_paths:
        filename = pred_path.name
        src_path = Path(gt_labels_path) / filename
        copied_label_path = path_for_mapped_targets / filename
        shutil.copy(src_path, copied_label_path)

        seg, sitk_stuff = image_reader_writer.read_seg(copied_label_path)
        seg = torch.as_tensor(seg)
        mapped_seg = map_label(seg, get_map_idxs(train_test_label_mapping, evaluated_labels, input_type='test_labels'), input_format='argmaxed').squeeze(0)
        image_reader_writer.write_seg(mapped_seg.squeeze(0).numpy(), copied_label_path, sitk_stuff)

    with open(save_path / 'evaluated_labels.json', 'w') as f:
        json.dump({v:k for k,v in enumerate(evaluated_labels)}, f, indent=4)

    # Run postprocessing
    postprocessing_function_dict[config['train_tta_data_map']](save_path)

    compute_metrics_on_folder_simple(
        folder_ref=path_for_mapped_targets, folder_pred=save_path,
        labels=list(range(len(evaluated_labels))),
        num_processes=0, chill=True)

    with open(save_path / 'summary.json', 'r') as f:
        summary_json = json.load(f)
        final_mean_dice = summary_json["foreground_mean"]["Dice"]

    if wandb.run is not None: wandb.log({f'scores/tta_dice_mean': final_mean_dice})

TTA_OUTPUT_DIR = NNUNET_BASE_DIR + "nnUNetV2_TTA_results"
PROJECT_NAME = 'nnunet_tta'


# %% [markdown]
# # Run a single run

# %%
# TODO remove those debugging lines
if False:
    CONFIG_DICT = update_data_mapping_config(CONFIG_DICT)

    debug_path = Path(NNUNET_BASE_DIR + "nnUNetV2_TTA_results") / 'debug'
    if debug_path.is_dir():
        shutil.rmtree(debug_path)

    tta_main(CONFIG_DICT, NNUNET_BASE_DIR + "nnUNetV2_TTA_results",
                get_evaluated_labels(CONFIG_DICT),
                get_train_test_label_mapping(CONFIG_DICT),
                'debug', debug=False)
    sys.exit(0)



# %%
# TODO externalize this
sweep_config_dict = dict(
    method='grid',
    metric=dict(goal='maximize', name='scores/tta_dice_mean'),
    parameters=dict(
        have_grad_in=dict(
            values=['branch_a', 'both']
        ),
        do_intensity_aug_in=dict(
            values=['branch_a', 'branch_b', 'both', 'none']

        ),
        do_spatial_aug_in=dict(
            values=['branch_a', 'branch_b', 'both', 'none']

        ),
        spatial_aug_type=dict(
            values=['deformable', 'affine']
        ),
        # trainer=dict(
        #     # values=['NNUNET', 'GIN', 'NNUNET_BN']
        # ),
    )
)

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
        parser.add_argument('--pretrained-config', help='Fold ID of nnUNet model to use for pretraining', default='3d_fullres')
        parser.add_argument('--pretrained-fold', help='Fold ID of nnUNet model to use for pretraining', default='0')

        args = parser.parse_args(sys.argv[2:])
        pretrained_task_id = int(args.pretrained_task_id) \
            if args.pretrained_task_id.isnumeric() else args.pretrained_task_id
        pretrained_fold = int(args.pretrained_fold) \
            if args.pretrained_fold.isnumeric() else args.pretrained_fold

        dg_tta.tta.config_log_utils.prepare_tta(pretrained_task_id, int(args.tta_task_id),
                                                pretrainer=args.pretrainer,
                                                pretrained_config=args.pretrained_config,
                                                pretrained_fold=pretrained_fold)

    def run_tta(self):
        parser = argparse.ArgumentParser(description='Run DG-TTA')
        parser.add_argument('repository')
        args = parser.parse_args(sys.argv[2:])
        raise NotImplementedError()
        print(f'Running git fetch, repository={args.repository}')

def main():
    assert Path(os.environ.get('DG_TTA_ROOT', '_')).is_dir(), \
    "Please define an existing root directory for DG-TTA by setting DG_TTA_ROOT."

    # parser = ArgumentParser()
    # parser.add_argument('--pretrained_task_id', type=str)
    # parser.add_argument('--tta_task_id', type=str)
    # parser.add_argument('--prepare', action='store_true')
    # # parser.add_argument('--debug', action='store_true')

    # args = parser.parse_args()
    # print(get_train_test_label_mapping(CONFIG_DICT))

    DGTTAProgram()
    tta_main()

if __name__ == "__main__":
    main()