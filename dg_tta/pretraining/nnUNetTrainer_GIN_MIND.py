import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# TODO clean trainers

from dg_tta.gin import GINGroupConv
from dg_tta.mind import MIND3D

def gin_mind_hook(module, input):
    input = input[0]

    use_gin = '--tr_disable_internal_augmentation' not in sys.argv
    cfg = dict(
        IN_CHANNELS=1,
        N_LAYER=4,
        INTERM_CHANNELS=2,
    )
    gin_group_conv = GINGroupConv(cfg)
    input = gin_group_conv(input, use_gin)

    input = MIND3D().forward(input)
    return input

class nnUNetTrainer_GIN_MIND(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """MINDSSC nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_input_channels = 12
        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        network.register_forward_pre_hook(gin_mind_hook)

        return network