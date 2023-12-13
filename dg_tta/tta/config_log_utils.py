
import subprocess
import shutil
import json
import datetime
import wandb
import torch

from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

TEMPLATE_CONFIG = dict(
    pretrainer='nnUNetTrainer_GIN_MIND',
    pretrained_config='3d_fullres',
    pretrained_fold='0',
    intensity_aug_function='GIN',

    tta_across_all_samples=False,

    lr=1e-5,
    fixed_sample_idx=None,
    ensemble_count=3,
    epochs=12,

    params_with_grad='all', # all, norms, encoder
    have_grad_in='branch_a', # ['branch_a', 'branch_b', 'both']
    do_intensity_aug_in='none', # ['branch_a', 'branch_b', 'both', 'none']
    do_spatial_aug_in='both', # ['branch_a', 'branch_b', 'both', 'none']
    spatial_aug_type='affine', # ['affine', 'deformable']

    wandb_mode='disabled',
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



# TODO find a solution for auto-naming
def wandb_run(config_dict):
    # TODO refactor
    config_dict = update_data_mapping_config(config_dict)
    evaluated_labels = get_evaluated_labels(config_dict)
    train_test_label_mapping = get_train_test_label_mapping(config_dict)
    now_str = datetime.now().strftime("%Y%m%d__%H_%M_%S")

    with wandb.init(
        mode=config_dict['wandb_mode'], config=config_dict, project=PROJECT_NAME) as run:
        config = wandb.config
        run.name = f"{now_str}_{run.name}"
        tta_main(config, TTA_OUTPUT_DIR, evaluated_labels, train_test_label_mapping, run.name, debug=False)
    wandb.finish()
    torch.cuda.empty_cache()


from pathlib import Path
import os

def prepare_tta(pretrained_task_id, tta_task_id,
                pretrainer=None, pretrained_config=None, pretrained_fold=None):
    assert pretrained_task_id != tta_task_id
    assert (
        pretrained_task_id in ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND'] \
        or isinstance(pretrained_task_id, int)
    )
    assert isinstance(tta_task_id, int)

    if isinstance(pretrained_task_id, int):
        # Check fold specifier
        assert pretrainer is not None
        assert pretrained_config is not None
        assert pretrained_fold == 'all' or isinstance(pretrained_fold, int)

    # Get dataset names
    tta_task_name = maybe_convert_to_dataset_name(tta_task_id)
    if isinstance(pretrained_task_id, int):
        pretrained_task_name = maybe_convert_to_dataset_name(pretrained_task_id)
    else:
        pretrained_task_name = pretrained_task_id

    root_dir = Path(os.environ['DG_TTA_ROOT'])
    assert root_dir.is_dir()

    # Create directories
    map_folder = f"Pretrained_{pretrained_task_name}_at_{tta_task_name}"
    plan_dir = (root_dir / 'plans' / map_folder)
    results_dir = (root_dir / 'results' / map_folder)

    shutil.rmtree(plan_dir, ignore_errors=True)
    plan_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(pretrained_task_id, str):
        weights_file_path = download_pretrained_weights(pretrained_task_id)
        pretrained_classes = torch.load(weights_file_path, map_location='cpu')['classes']

    else:
        # Get label mappings from nnUNet task
        raw_pretrained_dataset_dir = Path(nnUNet_raw, pretrained_task_name)
        with open(raw_pretrained_dataset_dir / "dataset.json", 'r') as f:
            pretrained_classes = json.load(f)['labels']

        # Get weights file
        fold_dir = f"fold_{pretrained_fold}" if pretrained_fold != 'all' else pretrained_fold
        results_pretrained_dataset_dir = Path(nnUNet_results, pretrained_task_name,
                                              f"{pretrainer}__nnUNetPlans__{pretrained_config}",
                                              fold_dir)
        weights_file_path = results_pretrained_dataset_dir / "checkpoint_final.pth"

        if not weights_file_path.is_file():
            raise FileNotFoundError(f"Could not find weights file at {weights_file_path}")

    raw_tta_task_dataset_dir = Path(nnUNet_raw, tta_task_name)

    # Load tta_task classes
    with open(raw_tta_task_dataset_dir / "dataset.json", 'r') as f:
        tta_task_classes = json.load(f)['labels']

    # Dump label mappings
    with open(plan_dir / f"{pretrained_task_name}_label_mapping.json", 'w') as f:
        json.dump(pretrained_classes, f, indent=4)
    with open(plan_dir / f"{tta_task_name}_label_mapping.json", 'w') as f:
        json.dump(tta_task_classes, f, indent=4)

    # Dump initial config
    initial_config = TEMPLATE_CONFIG.copy()
    initial_config['pretrained_weights_file'] = str(weights_file_path)

    with open(plan_dir / "tta_plan.json", 'w') as f:
        json.dump(initial_config, f, indent=4)



def download_pretrained_weights(pretrained_task_id):
    pretrained_weights_dir = Path(os.environ['DG_TTA_ROOT']) / "_pretrained_weights"

    if pretrained_task_id == 'TS104_GIN':
        dl_link = "https://cloud.imi.uni-luebeck.de/s/jE9sSR9d8ycd3WL/download"
        target_path = pretrained_weights_dir / "weights_DG_TTA_TS104_GIN.pt"

    elif pretrained_task_id == 'TS104_MIND':
        dl_link = "https://cloud.imi.uni-luebeck.de/s/p742eMfA6FZzJ2p/download"
        target_path = pretrained_weights_dir / "weights_DG_TTA_TS104_MIND.pt"

    elif pretrained_task_id == 'TS104_GIN_MIND':
        dl_link = "https://cloud.imi.uni-luebeck.de/s/54PZxGe58KSSyke/download"
        target_path = pretrained_weights_dir / "weights_DG_TTA_TS104_GIN_MIND.pt"

    else:
        raise ValueError()

    target_path.parent.mkdir(exist_ok=True)

    if not target_path.exists():
        subprocess.run(["wget", dl_link, "-O", target_path])

    return target_path