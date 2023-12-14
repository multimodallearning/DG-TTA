import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from dg_tta.gin import GINGroupConv


def gin_hook(module, input):
    if isinstance(input, list):
        input = input[0]

    use_gin = '--tr_disable_internal_augmentation' not in sys.argv
    cfg = dict(
        IN_CHANNELS=1,
        N_LAYER=4,
        INTERM_CHANNELS=2,
    )
    gin_group_conv = GINGroupConv(cfg)
    input = gin_group_conv(input, use_gin)
    return input

class nnUNetTrainer_GIN(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """RFA nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        network.register_forward_pre_hook(gin_hook)

        return network