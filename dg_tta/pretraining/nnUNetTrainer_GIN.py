import torch
import torch.nn as nn


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from dg_tta.gin import gin_hook
from dg_tta.utils import enable_internal_augmentation


class nnUNetTrainer_GIN(nnUNetTrainer):
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
        network = get_network_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
        )

        enable_internal_augmentation()
        gin_hook_handle = network.register_forward_pre_hook(gin_hook)

        return network
