import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from dg_tta.mind import MIND3D

def mind_hook(module, input):
    return MIND3D().forward(input[0])

class nnUNetTrainer_MIND(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """MINDSSC nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # self.descriptor = MIND3D()



        # if not self.was_initialized:
        #     self.initialize()

        # self.wrapper = MINDWrapper(self.network, MIND3D())
        # self.network = self.wrapper

        # self.network.register_forward_pre_hook(mind_hook)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_input_channels = 12
        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)
        network.register_forward_pre_hook(mind_hook)

        return network

    # def initialize(self):
    #     # Taken from nnunetTrainer and modified channel number
    #     if not self.was_initialized:
    #         self.num_input_channels = 12 # MIND channel number

    #         self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
    #                                                        self.configuration_manager,
    #                                                        self.num_input_channels,
    #                                                        enable_deep_supervision=True).to(self.device)
    #         # compile network for free speedup
    #         if ('nnUNet_compile' in os.environ.keys()) and (
    #                 os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
    #             self.print_to_log_file('Compiling network...')
    #             self.network = torch.compile(self.network)

    #         self.optimizer, self.lr_scheduler = self.configure_optimizers()
    #         # if ddp, wrap in DDP wrapper
    #         if self.is_ddp:
    #             self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
    #             self.network = DDP(self.network, device_ids=[self.local_rank])

    #         self.loss = self._build_loss()
    #         self.was_initialized = True
    #     else:
    #         raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
    #                            "That should not happen.")