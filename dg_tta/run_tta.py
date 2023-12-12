
# %%

NNUNET_BASE_DIR = '/home/weihsbach/storage/staff/christianweihsbach/nnunet/'
import os
os.environ['nnUNet_raw'] = NNUNET_BASE_DIR + "nnUNetV2_raw"
os.environ['nnUNet_preprocessed'] = NNUNET_BASE_DIR + "nnUNetV2_preprocessed"
os.environ['nnUNet_results'] = NNUNET_BASE_DIR + "nnUNetV2_results" # TODO check nnunet paths

import torch
from nnunetv2.training.nnUNetTrainer.variants.mdl_group_variants.nnUNetTrainer_GIN import gin_hook
from datetime import datetime
import wandb
import shutil
from nnunetv2.inference.predict_from_raw_data import (
    manage_input_and_output_lists,
    get_data_iterator_from_lists_of_filenames,
)
import json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from pathlib import Path
from torch._dynamo import OptimizedModule
from pathlib import Path
import wandb

from tqdm import trange,tqdm
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
import nibabel as nib

from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from collections import defaultdict
os.environ.update(get_vars('*'))

# %%
# def localNorm(train_img):
#     K = 11
#     kw = (K - 1) // 2
#     kernel_norm = F.avg_pool2d(torch.ones_like(train_img), K, stride=1, padding=kw)
#     img_mean = F.avg_pool2d(train_img, K, stride=1, padding=kw) / kernel_norm

#     img_var = (
#         F.avg_pool2d(train_img**2, K, stride=1, padding=kw) / kernel_norm
#         - img_mean**2
#     )
#     img_norm = (train_img - img_mean) / torch.sqrt(img_var + 0.1)
#     return img_norm


# def localNorm3d(train_img):
#     K = 11
#     kw = (K - 1) // 2

#     kernel_norm = F.avg_pool3d(torch.ones_like(train_img), K, stride=1, padding=kw)
#     img_mean = F.avg_pool3d(train_img, K, stride=1, padding=kw) / kernel_norm

#     img_var = (
#         F.avg_pool3d(train_img**2, K, stride=1, padding=kw) / kernel_norm
#         - img_mean**2
#     )
#     img_norm = (train_img - img_mean) / torch.sqrt(img_var + 0.1)
#     return img_norm


# %%
# def get_centers(probs):
#     B,C,D,H,W = probs.shape
#     prob_sum = probs.sum((-3,-2,-1))
#     d_centers = (probs * torch.arange(D, device=probs.device).view(1,1,D,1,1)).sum((-3,-2,-1))
#     h_centers = (probs * torch.arange(H, device=probs.device).view(1,1,1,H,1)).sum((-3,-2,-1))
#     w_centers = (probs * torch.arange(W, device=probs.device).view(1,1,1,1,W)).sum((-3,-2,-1))

#     centers = torch.stack([d_centers, h_centers, w_centers], dim=-1)
#     centers = centers / prob_sum.view(B,C,1)
#     return centers

# TODO clean functions

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

def get_rf_field(num_batch, size_3d, interpolation_factor=4, num_fields=4, alternating_fields=True, device='device'):

    rf_field = F.interpolate(
        F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(
                    torch.randn(
                        num_batch, num_fields,
                        size_3d[0]//interpolation_factor,
                        size_3d[1]//interpolation_factor,
                        size_3d[2]//interpolation_factor, device=device),
                    interpolation_factor,
                    stride=1,
                    padding=interpolation_factor//2,
                ),
                interpolation_factor,
                stride=1,
                padding=interpolation_factor//2,
            ),
            interpolation_factor,
            stride=1,
            padding=interpolation_factor//2,
        ),
        size=size_3d,
        mode="trilinear",
    )
    rf_field -= rf_field.mean((-3,-2,-1), keepdim=True)
    rf_field /= 1e-3 + rf_field.view(num_batch * num_fields, -1).std(1).view(
        num_batch, num_fields, 1, 1, 1
    )
    if alternating_fields:
        rf_field *= 2.5
        rf_field[:, ::2] += 1

    return rf_field

def load_batch_train(train_data, batch_idx, patch_size, affine_rand=0.05, use_rf=False, fixed_patch_idx=None):
    assert fixed_patch_idx in range(8) or fixed_patch_idx == None or fixed_patch_idx == 'center'

    num_batch = len(batch_idx)
    C = max(train_data.shape[1] - 1, 1)
    train_img = torch.zeros(num_batch, C, patch_size[0], patch_size[1], patch_size[2]).to(train_data.device)

    train_img1 = None
    train_label = torch.zeros(num_batch, patch_size[0], patch_size[1], patch_size[2]).to(train_data.device).long()

    for b in range(num_batch):
        with torch.no_grad():
            # Get patches
            data = train_data[batch_idx[b]]
            if fixed_patch_idx is None:
                rand_patch1 = torch.randint(max(data.shape[1] - patch_size[0], 0), (1,))
                rand_patch2 = torch.randint(max(data.shape[2] - patch_size[1], 0), (1,))
                rand_patch3 = torch.randint(max(data.shape[3] - patch_size[2], 0), (1,))
            elif fixed_patch_idx == 'center':
                rand_patch1 = max((data.shape[1]-patch_size[0])//2, 0)
                rand_patch2 = max((data.shape[2]-patch_size[1])//2, 0)
                rand_patch3 = max((data.shape[3]-patch_size[2])//2, 0)
            else:
                p_idxs = f"{fixed_patch_idx:03b}"
                p_idxs = [int(idx) for idx in [*p_idxs]]
                rand_patch1 = p_idxs[0] * patch_size[0]
                rand_patch2 = p_idxs[1] * patch_size[1]
                rand_patch3 = p_idxs[2] * patch_size[2]
                # print(rand_patch1, rand_patch2, rand_patch3)

            out_shape = (
                1,
                1,
                max(data.shape[1], patch_size[0]),
                max(data.shape[2], patch_size[1]),
                max(data.shape[3], patch_size[2]),
            )
            grid = F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(train_data.device)
                + affine_rand * torch.randn(1, 3, 4).to(train_data.device),
                out_shape, align_corners=False
            )
            patch_grid = grid[
                :,
                rand_patch1 : rand_patch1 + patch_size[0],
                rand_patch2 : rand_patch2 + patch_size[1],
                rand_patch3 : rand_patch3 + patch_size[2],
            ]
            if train_data.shape[1] > 1:
                train_label[b] = (
                    F.grid_sample(
                        data[-1:].unsqueeze(0).to(train_data.device), patch_grid, mode="nearest", align_corners=False
                    )
                    .squeeze()
                    .long()
                )
            train_img[b] = F.grid_sample(
                data[:-1].unsqueeze(0).to(train_data.device), patch_grid, align_corners=False
            ).squeeze()

    train_label = train_label.clamp_min_(0)

    all_train_img_augs = []
    if use_rf:
        with torch.no_grad():
            for _ in range(2):
                rf_field = get_rf_field(num_batch, (patch_size, patch_size, patch_size))
                train_img_aug = (
                    train_img * rf_field[:, : 2 * C : 2] + rf_field[:, 1 : 2 * C : 2]
                )
                all_train_img_augs.append(train_img_aug)

    return train_img, train_label, all_train_img_augs


# %%
def soft_dice(fixed,moving):
    B,C,D,H,W = fixed.shape
    # TODO Add d parameter

    nominator = (4. * fixed*moving).reshape(B,-1,D*H*W).mean(2)
    denominator = ((fixed + moving)**2).reshape(B,-1,D*H*W).mean(2)

    if denominator.sum() == 0.:
        dice = (nominator * 0.) + 1.
    else:
        dice  = nominator / denominator # Do not add an eps here, it disturbs the consistency

    return dice




# %%
# TODO move into torch utils
def fix_all(m):
    for p in m.parameters():
        p.requires_grad_(False)

def release_all(m):
    for p in m.parameters():
        p.requires_grad_(True)

def release_norms(m):
    if 'instancenorm' in m.__class__.__name__.lower() or 'batchnorm' in m.__class__.__name__.lower():
        print("Released", m.__class__.__name__)
        for p in m.parameters():
            p.requires_grad_(True)


# %%
def calc_consistent_diffeomorphic_field(disp_field, inverse_disp_field, time_steps=1, ensure_inverse_consistency=True, iter_steps_override=None):
    # https://github.com/multimodallearning/convexAdam/blob/76a595914eb21ea17795e6cd19503ab447f0ea6b/l2r_2021_convexAdam_task1_docker.py#L166
    # https://github.com/cwmok/LapIRN/blob/d8f96770a704b1f190955cc26297c7b01a270b0a/Code/miccai2020_model_stage.py#L761

    # Vincent Arsigny, Olivier Commowick, Xavier Pennec, Nicholas Ayache: A Log-Euclidean Framework for Statistics on Diffeomorphisms
    B,C,D,H,W = disp_field.size()
    dimension_correction = torch.tensor([D,H,W], device=disp_field.device).view(1,3,1,1,1)
    dt = 1./time_steps

    with torch.no_grad():
        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,D,H,W), align_corners=True).permute(0,4,1,2,3).to(disp_field)
        if ensure_inverse_consistency:
            out_disp_field = (disp_field/dimension_correction/(2**time_steps)*dt).clone()
            out_inverse_disp_field = (inverse_disp_field/dimension_correction/(2**time_steps)*dt).clone()

            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = out_disp_field.clone()
                inverse_ds = out_inverse_disp_field.clone()
                out_disp_field = \
                    +0.5 * ds \
                    -0.5 * F.grid_sample(inverse_ds, (identity + ds).permute(0,2,3,4,1), padding_mode='border', align_corners=True)

                out_inverse_disp_field = \
                    +0.5 * inverse_ds \
                    -0.5 * F.grid_sample(ds, (identity + inverse_ds).permute(0,2,3,4,1), padding_mode='border', align_corners=True)
            out_disp_field = out_disp_field * 2**time_steps * dimension_correction
            out_inverse_disp_field = out_inverse_disp_field * 2**time_steps * dimension_correction

        else:
            # https://github.com/cwmok/LapIRN/blob/d8f96770a704b1f190955cc26297c7b01a270b0a/Code/miccai2020_model_stage.py#L761

            ds_dt = disp_field/dimension_correction/(2**time_steps) # velocity = ds/dt
            inverse_ds_dt = inverse_disp_field/dimension_correction/(2**time_steps)
            ds = ds_dt*dt
            inverse_ds = inverse_ds_dt*dt

            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = ds + F.grid_sample(ds, (identity + ds).permute(0,2,3,4,1), mode='bilinear', padding_mode="zeros", align_corners=True)
                inverse_ds = inverse_ds + F.grid_sample(inverse_ds, (identity + inverse_ds).permute(0,2,3,4,1), mode='bilinear', padding_mode="zeros", align_corners=True)

            out_disp_field = ds * dimension_correction
            out_inverse_disp_field = inverse_ds * dimension_correction

    return out_disp_field, out_inverse_disp_field



def get_disp_field(batch_num, size_3d, factor=0.1, interpolation_factor=5, device='cpu'):
    field = get_rf_field(batch_num,size_3d, alternating_fields=False, num_fields=3, interpolation_factor=interpolation_factor, device=device)
    STEPS = 5
    disp_field, inverse_disp_field = calc_consistent_diffeomorphic_field(
        field*factor,
        torch.zeros_like(field),
        STEPS,
        ensure_inverse_consistency=True
    )
    return disp_field.permute(0,2,3,4,1), inverse_disp_field.permute(0,2,3,4,1)


# %% [markdown]
# # Define AUG functions

# %%

def get_rand_affine(batch_size, strength=0.05, flip=False):
    affine = torch.cat(
        (
            torch.randn(batch_size, 3, 4) * strength + torch.eye(3, 4).unsqueeze(0),
            torch.tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(batch_size, 1, 1),
        ),
        1,
    )

    if flip:
        flip_affine = torch.diag(torch.cat([(2*(torch.rand(3) > 0.5).float()-1), torch.tensor([1.])]))
        affine = affine @ flip_affine
    return affine[:,:3], affine.inverse()[:,:3]

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()

    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)


# TODO move into mind
def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img



def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist



class MIND3D(torch.nn.Module):
    def __init__(self, delta=1, sigma=1, randn_weighting=0.05) -> None:
        super().__init__()
        self.delta = delta
        self.sigma = sigma
        self.out_channels = 12
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.tensor([[0, 1, 1],
                                        [1, 1, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [2, 1, 1],
                                        [1, 2, 1]], dtype=torch.float)

        # squared distances
        dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :].long()
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :].long()
        mshift1 = torch.zeros((12, 1, 3, 3, 3))
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros((12, 1, 3, 3, 3))
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        self.rpad = nn.ReplicationPad3d(delta)
        self.mshift1 = mshift1
        self.mshift2 = mshift2
        self.randn_weighting = randn_weighting

    def forward(self, img):
        # compute patch-ssd
        device = img.device

        edge_selection = (
            F.conv3d(self.rpad(img), self.mshift1.to(device), dilation=self.delta)
            - F.conv3d(self.rpad(img), self.mshift2.to(device), dilation=self.delta)
        )

        edge_selection = edge_selection + self.randn_weighting * torch.randn_like(edge_selection)
        ssd = smooth(edge_selection ** 2, self.sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind


def gin_aug(input):
    enable_internal_augmentation()
    input = gin_hook(None, input)
    disable_internal_augmentation()
    return input

class GinMINDAug():
    def __init__(self):
        super().__init__()
        self.mind_fn = MIND3D()

    def __call__(self, input):
        return self.mind_fn(gin_aug(input))


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
    ) = manage_input_and_output_lists(
        list_of_lists_or_source_folder,
        output_folder_or_list_of_truncated_output_files,
        dataset_json,
        folder_with_segs_from_prev_stage,
        overwrite,
        part_id,
        num_parts,
        save_probabilities,
    )

    data_iterator = get_data_iterator_from_lists_of_filenames(
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



# %%
import os
from copy import deepcopy
from torch._dynamo import OptimizedModule
import json

from nnunetv2.inference.predict_from_raw_data import load_trained_model_for_inference
from nnunet_inference import predict_from_data_iterator

def load_model(model_training_output_path, fold):

    use_folds = [fold] if isinstance(fold, int) else fold # We only trained one fold
    checkpoint_name = "checkpoint_final.pth"

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_trained_model_for_inference(model_training_output_path, use_folds, checkpoint_name)

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
    return predict_from_data_iterator(tta_data, network, tta_parameters, plans_manager, configuration_manager, dataset_json,
                                inference_allowed_mirroring_axes, tile_step_size, use_gaussian, use_mirroring,
                                perform_everything_on_gpu, verbose, save_probabilities,
                                num_processes_segmentation_export, device)


# %% [markdown]
# # Pre/Postprocessing methods

# %%
import functools
def get_named_layers_leaves(module):
    """ Returns all leaf layers of a pytorch module and a keychain as identifier.
        e.g.
        ...
        ('features.0.5', nn.ReLU())
        ...
        ('classifier.0', nn.BatchNorm2D())
        ('classifier.1', nn.Linear())
    """

    return [(keychain, sub_mod) for keychain, sub_mod in list(module.named_modules()) if not next(sub_mod.children(), None)]



MOD_GET_FN = lambda self, key: self[int(key)] if isinstance(self, nn.Sequential) \
                                              else getattr(self, key)

def get_module_data(module, keychain):
    """Retrieves any data inside a pytorch module for a given keychain.
       Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """

    return functools.reduce(MOD_GET_FN, keychain.split('.'), module)



def set_module_data(module, keychain, data):
    """Replaces any data inside a pytorch module for a given keychain.
       Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """
    key_list = keychain.split('.')
    root = functools.reduce(MOD_GET_FN, key_list[:-1], module)
    leaf = key_list[-1]
    if isinstance(root, nn.Sequential):
        root[int(leaf)] = data
    else:
        setattr(root, leaf, data)


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

def update_data_mapping_config(config_dict):
    # TODO: find a better solution for this string thing
    from_to_str =  f"{config_dict['train_data']}->{config_dict['tta_data']}"
    config_dict['train_tta_data_map'] = from_to_str
    return config_dict

def get_evaluated_labels(config_dict):
    return EVALUATED_LABELS_DICT[config_dict['train_tta_data_map']]

def get_train_test_label_mapping(config_dict):
    train_test_label_mapping = generate_label_mapping(
        dataset_labels_dict[config_dict['train_data']],
        dataset_labels_dict[config_dict['tta_data']]
    )
    return train_test_label_mapping

print(get_train_test_label_mapping(CONFIG_DICT))

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

# %% [markdown]
# # TTA routine

# %%

def enable_internal_augmentation():
    if '--tr_disable_internal_augmentation' in sys.argv:
        sys.argv.remove('--tr_disable_internal_augmentation')

def disable_internal_augmentation():
    if not '--tr_disable_internal_augmentation' in sys.argv:
        sys.argv.append('--tr_disable_internal_augmentation')

def check_internal_augmentation_disabled():
    assert '--tr_disable_internal_augmentation' in sys.argv

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

def tta_routine(config, save_path, evaluated_labels, train_test_label_mapping, run_name=None, debug=False):

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

    disable_internal_augmentation()

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

                    loss = 1 - soft_dice(sm_a, sm_b)[:,START_CLASS:].mean()

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

    import shutil
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

# %% [markdown]
# # Debugging

# %%

# %%
TTA_OUTPUT_DIR = NNUNET_BASE_DIR + "nnUNetV2_TTA_results"
PROJECT_NAME = 'nnunet_tta'
now_str = datetime.now().strftime("%Y%m%d__%H_%M_%S")

# TODO find a solution for auto-naming
def wandb_run(config_dict):
    config_dict = update_data_mapping_config(config_dict)
    evaluated_labels = get_evaluated_labels(config_dict)
    train_test_label_mapping = get_train_test_label_mapping(config_dict)

    with wandb.init(
        mode=config_dict['wandb_mode'], config=config_dict, project=PROJECT_NAME) as run:
        config = wandb.config
        run.name = f"{now_str}_{run.name}"
        tta_routine(config, TTA_OUTPUT_DIR, evaluated_labels, train_test_label_mapping, run.name, debug=False)
    wandb.finish()
    torch.cuda.empty_cache()


# %% [markdown]
# # Run a single run

# %%
# TODO remove those debugging lines
if False:
    CONFIG_DICT = update_data_mapping_config(CONFIG_DICT)

    debug_path = Path(NNUNET_BASE_DIR + "nnUNetV2_TTA_results") / 'debug'
    if debug_path.is_dir():
        shutil.rmtree(debug_path)

    tta_routine(CONFIG_DICT, NNUNET_BASE_DIR + "nnUNetV2_TTA_results",
                get_evaluated_labels(CONFIG_DICT),
                get_train_test_label_mapping(CONFIG_DICT),
                'debug', debug=False)
    sys.exit(0)

# %%
# wandb_run(CONFIG_DICT)


# %% [markdown]
# # Run sweep

# %%
# TODO  Move this to log utils
def wandb_sweep_run(config_dict):
    with wandb.init(settings=wandb.Settings(start_method="thread"),
        mode=config_dict['wandb_mode']) as run:
        config = wandb.config
        run.name = f"{now_str}_{run.name}"
        tta_routine(config, TTA_OUTPUT_DIR, run.name, debug=False)
    wandb.finish()
    torch.cuda.empty_cache()

def closure_wandb_sweep_run():
    return wandb_sweep_run(CONFIG_DICT)


# %%
import copy
from enum import Enum

def clean_sweep_dict(config_dict, sweep_config_dict):
    # Integrate all config entries into sweep_dict.parameters -> sweep overrides config
    cp_config_dict = copy.deepcopy(dict(config_dict))

    for del_key in sweep_config_dict['parameters'].keys():
        if del_key in cp_config_dict:
            del cp_config_dict[del_key]
    merged_sweep_config_dict = copy.deepcopy(sweep_config_dict)

    for key, value in cp_config_dict.items():
        merged_sweep_config_dict['parameters'][key] = dict(value=value)

    # Convert enum values in parameters to string. They will be identified by their numerical index otherwise
    for key, param_dict in merged_sweep_config_dict['parameters'].items():
        if 'value' in param_dict and isinstance(param_dict['value'], Enum):
            param_dict['value'] = str(param_dict['value'])
        if 'values' in param_dict:
            param_dict['values'] = [str(elem) if isinstance(elem, Enum) else elem for elem in param_dict['values']]

        merged_sweep_config_dict['parameters'][key] = param_dict
    return merged_sweep_config_dict

# %%
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

if False:
    merged_sweep_config_dict = clean_sweep_dict(config_dict, sweep_config_dict)
    sweep_id = wandb.sweep(merged_sweep_config_dict, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=closure_wandb_sweep_run)

# %% [markdown]
# # Configure manual sweep

# %%

run_dicts = [
    dict(
        trainer='GIN+MIND',
        lr=1e-6,
        intensity_aug_function='GIN+MIND'
    ),
    dict(
        trainer='NNUNET',
        intensity_aug_function='NNUNET'
    ),
    dict(
        trainer='NNUNET_BN',
        intensity_aug_function='NNUNET'
    ),
    dict(
        trainer='GIN',
        intensity_aug_function='GIN'
    ),
    dict(
        trainer='MIND',
        lr=1e-6,
        intensity_aug_function='MIND'
    ),
]

if True:
    for updatetee in run_dicts:
        updated_dict = deepcopy(CONFIG_DICT)
        updated_dict.update(updatetee)
        wandb_run(updated_dict)



if __name__ == "__main__":
    tta_main()