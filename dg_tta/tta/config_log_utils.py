import json
import datetime
import wandb
import torch



INITIAL_CONFIG_DICT = {

}

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

def prepare_dg_tta(pretrained_task_id, tta_task_id):
    assert pretrained_task_id != tta_task_id
    assert (
        pretrained_task_id in ['DG_TTA_TS_GIN', 'DG_TTA_TS_MIND', 'DG_TTA_TS_GIN+MIND'] \
        or isinstance(pretrained_task_id, int)
    )
    assert isinstance(tta_task_id, int)

    root_dir = Path(os.environ['DG_TTA_ROOT'])
    assert root_dir.is_dir()

    # Create directories
    map_folder = f"Pretrained_{pretrained_task_id}_at_{tta_task_id}"
    plan_dir = (root_dir / 'plans' / map_folder)
    results_dir = (root_dir / 'results' / map_folder)

    plan_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    if isinstance(pretrained_task_id, str):
        # Download pretrained weights and link in folder
        raise NotImplementedError()
    else:
        # Copy label mappings from nnUNet task
        raise NotImplementedError()

    # Add pretrained weights file path to config

    # Dump initial config
    with open(plan_dir / "config.json", 'w') as f:
        json.dump(INITIAL_CONFIG_DICT, f, indent=4)
    # TODO continue here

    raise NotImplementedError()

import subprocess

def download_pretrained_weights(pretrained_task_id):
    pretrained_weights_dir = Path(os.environ['DG_TTA_ROOT']) / "pretrained_weights"
    if pretrained_task_id == 'DG_TTA_TS104_GIN':
        raise NotImplementedError()
    elif pretrained_task_id == 'DG_TTA_TS104_MIND':
        raise NotImplementedError()
    elif pretrained_task_id == 'DG_TTA_TS104_GIN_MIND':
        dl_link = "https://cloud.imi.uni-luebeck.de/s/eeS3Coq4nywEJFn"
        target = pretrained_weights_dir / "weights_DG_TTA_TS104_GIN_MIND.pt"
    else:
        raise ValueError()

    if not target.exists():
        subprocess.run(["wget", dl_link, "-O", target])