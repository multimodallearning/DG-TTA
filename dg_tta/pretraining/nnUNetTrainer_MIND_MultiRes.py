import torch
import torch.nn as nn

from typing import Union, Tuple, List

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from dg_tta.mind import mind_hook
from dg_tta.pretraining.discrete_downsampling import SimulateDiscreteLowResolutionTransform


class nnUNetTrainer_MIND_MultiRes(nnUNetTrainer):
    # Changed SimulateLowResolutionTransform parameters to have a more intense low-res setting
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        num_input_channels = 12
        network = get_network_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
        )
        mind_hook_handle = network.register_forward_pre_hook(mind_hook)

        return network

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:

        tr_transforms = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data, order_resampling_seg, border_val_seg, use_mask_for_norm, is_cascaded,
            foreground_labels, regions, ignore_label,
        )

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
        else:
            ignore_axes = None

        transform_list = tr_transforms.transforms
        for tr_idx, tr in enumerate(transform_list):
            if isinstance(tr, SimulateLowResolutionTransform):
                # Replace SimulateLowResolutionTransform with SimulateDiscreteLowResolutionTransform
                sdlr_tr = SimulateDiscreteLowResolutionTransform(
                    zoom_range=(1/6, 1/4, 1/2), # Using discrete zoom range
                    zoom_axes_invidually=False,
                    per_channel=False,
                    p_per_channel=1., # Increased to 1.
                    order_downsample=0, order_upsample=3, p_per_sample=.5,
                    ignore_axes=ignore_axes)

                transform_list[tr_idx] = sdlr_tr

        tr_transforms = Compose(transform_list)
        return tr_transforms