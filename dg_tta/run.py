import sys
import re
from pathlib import Path
import json
import argparse
import json
from datetime import datetime

import torch
import torch.nn.functional as F

import randomname

from nnunetv2.run.run_training import run_training_entry as nnunet_run_training_main

import dg_tta
from dg_tta.__build__ import inject_dg_trainers_into_nnunet, check_trainers_injected
from dg_tta.utils import check_dga_root_is_set
from dg_tta.tta.torch_utils import generate_label_mapping

from dg_tta.tta.config_log_utils import (
    wandb_run,
    load_current_modifier_functions,
    get_tta_folders,
    wandb_run_is_available,
)
from dg_tta.tta.tta import tta_main

PROJECT_NAME = "nnunet_tta"


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
        check_trainers_injected()
        print("Dispatching into nnUNetv2_train.")
        sys.argv = sys.argv[2:]
        sys.argv.insert(0, "nnUNetv2_train")
        nnunet_run_training_main()

    def prepare_tta(self):
        check_trainers_injected()
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
        check_trainers_injected()
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
        pretrainer_fold = (
            int(args.pretrainer_fold)
            if args.pretrainer_fold.isnumeric()
            else args.pretrainer_fold
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
            pretrainer_fold,
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
