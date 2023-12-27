import os
import sys
import pathlib
from pathlib import Path
import importlib
import subprocess
import shutil
import inspect
import json
from contextlib import contextmanager

import matplotlib
from matplotlib import pyplot as plt

if importlib.util.find_spec("wandb"):
    import wandb
import torch

from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

import dg_tta

TEMPLATE_PLAN = dict(
    tta_across_all_samples=False,
    tta_eval_patches=1,
    batch_size=1,
    patches_to_be_accumulated=16,
    lr=1e-5,
    ensemble_count=3,
    epochs=12,
    start_tta_at_epoch=1,
    intensity_aug_function="GIN",  # ['GIN', 'disabled']
    spatial_aug_type="affine",  # ['affine', 'deformable']
    params_with_grad="all",  # all, norms, encoder
    have_grad_in="branch_a",  # ['branch_a', 'branch_b', 'both']
    do_intensity_aug_in="none",  # ['branch_a', 'branch_b', 'both', 'none']
    do_spatial_aug_in="both",  # ['branch_a', 'branch_b', 'both', 'none']
    num_processes=1,
    wandb_mode="disabled",
)


class ModifierFunctions:
    def __init__(self):
        pass

    @staticmethod
    def modify_tta_input_fn(image: torch.Tensor):
        assert image.ndim == 5  # B,1,D,H,W
        # This function will be called on the input that is fed to the model
        return image

    @staticmethod
    def modfify_tta_model_output_fn(pred_label: torch.Tensor):
        assert pred_label.ndim == 5  # B,C,D,H,W
        # This function will be called directly after model prediction
        return pred_label

    @staticmethod
    def modify_tta_output_after_mapping_fn(mapped_label: torch.Tensor):
        assert mapped_label.ndim == 5  # B,MAPPED_C,D,H,W
        # This function will be called after model prediction when labels are mapped
        # to the target label numbers/ids.
        return mapped_label

    @staticmethod
    def postprocess_results_fn(results_dir: pathlib.Path):
        pass
        # This function will be called on the final output directory.


def wandb_run(wandb_project_name, tta_fn, **kwargs):
    config = kwargs["config"]
    with wandb.init(
        project=wandb_project_name,
        name=kwargs["run_name"],
        mode=config["wandb_mode"],
        config=config,
    ):
        kwargs["config"] = wandb.config
        tta_fn(**kwargs)
    wandb.finish()
    torch.cuda.empty_cache()


def get_tta_folders(
    pretrained_dataset_id,
    tta_dataset_id,
    pretrainer,
    pretrainer_config,
    pretrainer_fold,
):
    root_dir = Path(os.environ["DG_TTA_ROOT"])

    pretrainer, pretrainer_config, pretrainer_fold = check_dataset_pretrain_config(
        pretrained_dataset_id, pretrainer, pretrainer_config, pretrainer_fold
    )

    # Get dataset names
    tta_dataset_name = maybe_convert_to_dataset_name(tta_dataset_id)

    if isinstance(pretrained_dataset_id, int):
        pretrained_dataset_name = maybe_convert_to_dataset_name(pretrained_dataset_id)
    else:
        pretrained_dataset_name = pretrained_dataset_id

    fold_folder = (
        f"fold_{pretrainer_fold}" if pretrainer_fold != "all" else pretrainer_fold
    )
    map_folder = f"Pretrained_{pretrained_dataset_name}_at_{tta_dataset_name}"
    pretrainer_folder = f"{pretrainer}__{pretrainer_config}"

    plan_dir = root_dir / "plans" / map_folder / pretrainer_folder / fold_folder
    results_dir = root_dir / "results" / map_folder / pretrainer_folder / fold_folder

    tta_data_dir = Path(nnUNet_raw, tta_dataset_name)

    return (
        tta_data_dir,
        plan_dir,
        results_dir,
        pretrained_dataset_name,
        tta_dataset_name,
    )


def check_dataset_pretrain_config(
    pretrained_dataset_id, pretrainer, pretrainer_config, pretrainer_fold
):
    assert pretrained_dataset_id in [
        "TS104_GIN",
        "TS104_MIND",
        "TS104_GIN_MIND",
    ] or isinstance(pretrained_dataset_id, int)

    if isinstance(pretrained_dataset_id, int):
        # Check fold specifier
        assert pretrainer is not None
        assert pretrainer_config is not None
        assert pretrainer_fold == "all" or isinstance(pretrainer_fold, int)
    else:
        if pretrained_dataset_id == "TS104_GIN":
            pretrainer = "nnUNetTrainer_GIN"
            pretrainer_config = "3d_fullres"
            pretrainer_fold = "0"

        elif pretrained_dataset_id == "TS104_MIND":
            pretrainer = "nnUNetTrainer_MIND"
            pretrainer_config = "3d_fullres"
            pretrainer_fold = "0"

        elif pretrained_dataset_id == "TS104_GIN_MIND":
            pretrainer = "nnUNetTrainer_GIN_MIND"
            pretrainer_config = "3d_fullres"
            pretrainer_fold = "0"

        else:
            raise ValueError()

    return pretrainer, pretrainer_config, pretrainer_fold


def prepare_tta(
    pretrained_dataset_id,
    tta_dataset_id,
    pretrainer=None,
    pretrainer_config=None,
    pretrainer_fold=None,
    tta_dataset_bucket="imagesTs",
):
    root_dir = Path(os.environ["DG_TTA_ROOT"])
    assert root_dir.is_dir()

    # Create directories
    (
        _,
        plan_dir,
        results_dir,
        pretrained_dataset_name,
        tta_dataset_name,
    ) = get_tta_folders(
        pretrained_dataset_id,
        tta_dataset_id,
        pretrainer,
        pretrainer_config,
        pretrainer_fold,
    )

    shutil.rmtree(plan_dir, ignore_errors=True)
    plan_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(pretrained_dataset_id, str):
        target_path, weights_file_path = download_pretrained_weights(
            pretrained_dataset_id
        )
        with open(target_path / "dataset.json", "r") as f:
            pretrained_classes = json.load(f)["labels"]

    else:
        # Get label mappings from nnUNet task
        raw_pretrained_dataset_dir = Path(nnUNet_raw, pretrained_dataset_name)
        with open(raw_pretrained_dataset_dir / "dataset.json", "r") as f:
            pretrained_classes = json.load(f)["labels"]

        # Get weights file
        fold_dir = (
            f"fold_{pretrainer_fold}" if pretrainer_fold != "all" else pretrainer_fold
        )
        results_pretrained_dataset_dir = Path(
            nnUNet_results,
            pretrained_dataset_name,
            f"{pretrainer}__nnUNetPlans__{pretrainer_config}",
            fold_dir,
        )
        weights_file_path = results_pretrained_dataset_dir / "checkpoint_final.pth"

        if not weights_file_path.is_file():
            raise FileNotFoundError(
                f"Could not find weights file at {weights_file_path}"
            )

    raw_tta_dataset_dir = Path(nnUNet_raw, tta_dataset_name)

    # Load tta_task classes
    with open(raw_tta_dataset_dir / "dataset.json", "r") as f:
        tta_dataset_classes = json.load(f)["labels"]

    # Dump label mappings
    with open(plan_dir / f"{pretrained_dataset_name}_label_mapping.json", "w") as f:
        json.dump(pretrained_classes, f, indent=4)
    with open(plan_dir / f"{tta_dataset_name}_label_mapping.json", "w") as f:
        json.dump(tta_dataset_classes, f, indent=4)

    # Create plan
    initial_plan = TEMPLATE_PLAN.copy()
    initial_plan["__pretrained_dataset_name__"] = pretrained_dataset_name
    initial_plan["__tta_dataset_name__"] = tta_dataset_name
    initial_plan["pretrained_weights_filepath"] = str(weights_file_path)

    # Retrive possible labels to be optimized (may require manual tweaking later)
    intersection_classes = list(
        set(pretrained_classes.keys()).intersection(set(tta_dataset_classes))
    )
    assert (
        "background" in intersection_classes
    ), "Background class must be present in both datasets!"
    intersection_classes.sort()
    intersection_classes.remove("background")
    intersection_classes.insert(0, "background")
    initial_plan["optimized_labels"] = intersection_classes

    # Retrive filepath of tta_task data
    tta_data_filepaths = get_data_filepaths(tta_dataset_name, tta_dataset_bucket)
    initial_plan["tta_data_filepaths"] = [str(fp) for fp in tta_data_filepaths]

    # Dump plan
    with open(plan_dir / "tta_plan.json", "w") as f:
        json.dump(initial_plan, f, indent=4)

    # Dump modifier functions
    modifier_src = inspect.getsource(ModifierFunctions)

    with open(plan_dir / "modifier_functions.py", "w") as f:
        f.write("import pathlib\n")
        f.write("import torch\n\n")
        f.write(modifier_src)

    copy_check_tta_input_notebook(plan_dir)

    print(
        f"\nPreparation done. You can edit the plan, modifier functions and optimized labels in {plan_dir} prior to running TTA."
    )


def get_resources_dir():
    return Path(dg_tta.__file__).parent / "__resources__"


def download_pretrained_weights(pretrained_dataset_id):
    pretrained_weights_dir = Path(os.environ["DG_TTA_ROOT"]) / "_pretrained_weights"

    if pretrained_dataset_id == "TS104_GIN":
        pretrainer_dir = "nnUNetTrainer_GIN__nnUNetPlans__3d_fullres"
        dl_link = "https://cloud.imi.uni-luebeck.de/s/ERK6Wic3D95qDKz/download"

    elif pretrained_dataset_id == "TS104_MIND":
        pretrainer_dir = "nnUNetTrainer_MIND__nnUNetPlans__3d_fullres"
        dl_link = "https://cloud.imi.uni-luebeck.de/s/LZByo9m3A5c6Dki/download"

    elif pretrained_dataset_id == "TS104_GIN_MIND":
        pretrainer_dir = "nnUNetTrainer_GIN_MIND__nnUNetPlans__3d_fullres"
        dl_link = "https://cloud.imi.uni-luebeck.de/s/dkGdfFGwbnzWya4/download"

    else:
        raise ValueError()

    dummy_results_path = get_resources_dir() / "dummy_results" / pretrainer_dir
    target_path = pretrained_weights_dir / pretrainer_dir
    target_path_weights = target_path / "fold_0" / "checkpoint_final.pth"

    target_path.mkdir(exist_ok=True)
    target_path_weights.parent.mkdir(exist_ok=True)

    # Copy dummy pretraining results (folder structure and nnUNet fils)
    shutil.copytree(dummy_results_path, target_path, dirs_exist_ok=True)

    if not target_path_weights.exists():
        subprocess.run(["wget", dl_link, "-O", target_path_weights])

    return target_path, target_path_weights


def get_global_idx(list_of_tuple_idx_max):
    # Get global index e.g. 2250 for ensemble_idx=2, epoch_idx=250 @ max_epochs<1000
    global_idx = 0
    next_multiplier = 1

    # Smallest identifier tuple last!
    for idx, max_of_idx in reversed(list_of_tuple_idx_max):
        global_idx = global_idx + next_multiplier * idx
        next_multiplier = next_multiplier * 10 ** len(str(int(max_of_idx)))
    return global_idx


def load_current_modifier_functions(plan_dir):
    mod_path = Path(plan_dir / "modifier_functions.py")
    spec = importlib.util.spec_from_file_location(
        "dg_tta.current_modifier_functions", mod_path
    )
    dyn_mod = importlib.util.module_from_spec(spec)
    sys.modules["dg_tta.current_modifier_functions"] = dyn_mod
    spec.loader.exec_module(dyn_mod)

    return dyn_mod


def get_data_filepaths(tta_dataset_name, tta_dataset_bucket):
    raw_tta_dataset_dir = Path(nnUNet_raw, tta_dataset_name)
    if tta_dataset_bucket == "imagesTr":
        source_folders = [raw_tta_dataset_dir / "imagesTr"]
    elif tta_dataset_bucket == "imagesTs":
        source_folders = [raw_tta_dataset_dir / "imagesTs"]
    elif tta_dataset_bucket == "imagesTrAndTs":
        source_folders = [
            raw_tta_dataset_dir / "imagesTr",
            raw_tta_dataset_dir / "imagesTs",
        ]

    file_list = []
    for src_fld in source_folders:
        if src_fld.is_dir():
            file_list.extend(filter(lambda x: x.is_file(), src_fld.iterdir()))

    return file_list


def wandb_run_is_available():
    return (
        importlib.util.find_spec("wandb") is not None
        and wandb.run is not None
        and not wandb.run.disabled
    )


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_dgtta_colormap():
    hi_1 = "#248888"
    hi_2 = "#79DCF0"
    hi_3 = "#e7475e"
    hi_4 = "#f0d879"
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [hi_3, hi_4, hi_2, hi_1]
    )


def plot_run_results(save_path, sample_id, ensemble_idx, tta_losses, eval_dices):
    fig, ax_one = plt.subplots()
    ax_two = ax_one.twinx()
    cmap = get_dgtta_colormap()
    c1, c2 = cmap(0.0), cmap(0.8)
    ax_one.plot(tta_losses, label="loss", c=c1)
    ax_one.set_yticks([tta_losses.min(), tta_losses.max()])
    ax_one.set_xlim(0, len(tta_losses) - 1)
    ax_one.set_ylabel("Soft-Dice Loss", c=c1)
    ax_one.tick_params(axis="y", colors=c1)
    ax_one.set_xlabel("TTA Epoch")
    ax_one.grid(axis="y", linestyle="--", linewidth=0.5)
    ax_one.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    ax_two.plot(eval_dices * 100, label="eval_dices", c=c2)
    ax_two.set_yticks([eval_dices.min() * 100, eval_dices.max() * 100])
    ax_two.set_ylabel("Pseudo Dice in %", c=c2)
    ax_two.tick_params(axis="y", colors=c2)
    ax_two.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    fig.suptitle(f"{sample_id} (ensemble_idx={ensemble_idx})")
    split_sample_id = sample_id.split('/')[-1]
    tta_plot_save_path = (
        save_path / f"{split_sample_id}__ensemble_idx_{ensemble_idx}_tta_results.png"
    )
    fig.savefig(tta_plot_save_path)
    fig.tight_layout()
    plt.close(fig)


def copy_check_tta_input_notebook(plan_dir):
    NB_FILENAME = "check_tta_input.ipynb"
    shutil.copyfile(
        get_resources_dir() / NB_FILENAME,
        plan_dir / NB_FILENAME,
    )


def get_parameters_save_path(save_path, sample_id, ensemble_idx):
    sample_id = sample_id.split('/')[-1]
    tta_parameters_save_path = (
        save_path / f"{sample_id}__ensemble_idx_{ensemble_idx}_tta_parameters.pt"
    )
    return tta_parameters_save_path