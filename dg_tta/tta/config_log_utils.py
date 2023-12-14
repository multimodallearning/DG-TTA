
import os
import sys
from pathlib import Path
import importlib
import subprocess
import shutil
import inspect
import json
import datetime
import wandb
import torch

from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

TEMPLATE_CONFIG = dict(
    tta_across_all_samples=False,

    lr=1e-5,
    fixed_sample_idx=None,
    ensemble_count=3,
    epochs=12,

    intensity_aug_function='GIN', # ['GIN', 'disabled']
    spatial_aug_type='affine', # ['affine', 'deformable']

    params_with_grad='all', # all, norms, encoder
    have_grad_in='branch_a', # ['branch_a', 'branch_b', 'both']
    do_intensity_aug_in='none', # ['branch_a', 'branch_b', 'both', 'none']
    do_spatial_aug_in='both', # ['branch_a', 'branch_b', 'both', 'none']

    wandb_mode='disabled',
)



class ModifierFunctions():
    def __init__(self):
        pass
    @staticmethod

    def modify_tta_input_fn(image: torch.Tensor):
        # This function will be called on the input that is fed to the model
        return image

    @staticmethod
    def modfify_tta_model_output_fn(pred_label: torch.Tensor):
        # This function will be called directly after model prediction
        return pred_label

    @staticmethod
    def modify_tta_output_after_mapping_fn(mapped_label: torch.Tensor):
        # This function will be caled after model prediction when labels are mapped
        # to the target label numbers/ids.
        return mapped_label



def update_data_mapping_config(config_dict):
    # TODO: find a better solution for this string thing
    from_to_str =  f"{config_dict['train_data']}->{config_dict['tta_data']}"
    config_dict['train_tta_data_map'] = from_to_str
    return config_dict



# def get_optimized_labels(config_dict):
#     return optimized_labels_DICT[config_dict['train_tta_data_map']]



# def get_train_test_label_mapping(config_dict):
#     train_test_label_mapping = generate_label_mapping(
#         dataset_labels_dict[config_dict['train_data']],
#         dataset_labels_dict[config_dict['tta_data']]
#     )
#     return train_test_label_mapping



# TODO find a solution for auto-naming
def wandb_run(wandb_project_name, run_name, output_dir, config_dict, label_mapping, optimized_labels, tta_fn):
    # TODO refactor
    config_dict = update_data_mapping_config(config_dict)
    # train_test_label_mapping = get_train_test_label_mapping(config_dict)

    with wandb.init(
        mode=config_dict['wandb_mode'], config=config_dict, project=wandb_project_name) as run:
        config = wandb.config
        now_str = datetime.now().strftime("%Y%m%d__%H_%M_%S")
        run.name = f"{now_str}_{run.name}"
        tta_fn(config, output_dir, optimized_labels, label_mapping, run.name, debug=False)
    wandb.finish()
    torch.cuda.empty_cache()



def get_tta_folders(pretrained_task_id, tta_task_id, pretrainer, pretrainer_config, pretrainer_fold):
    root_dir = Path(os.environ['DG_TTA_ROOT'])

    # Get dataset names
    tta_task_name = maybe_convert_to_dataset_name(tta_task_id)

    if isinstance(pretrained_task_id, int):
        pretrained_task_name = maybe_convert_to_dataset_name(pretrained_task_id)
    else:
        pretrained_task_name = pretrained_task_id

    fold_folder = f"fold_{pretrainer_fold}" if pretrainer_fold != 'all' else pretrainer_fold
    map_folder = f"Pretrained_{pretrained_task_name}_at_{tta_task_name}__{pretrainer}__{pretrainer_config}"
    plan_dir = (root_dir / 'plans' / map_folder / fold_folder)
    results_dir = (root_dir / 'results' / map_folder / fold_folder)

    tta_data_dir = Path(nnUNet_raw, tta_task_name)

    return tta_data_dir, plan_dir, results_dir, pretrained_task_name, tta_task_name

def prepare_tta(pretrained_task_id, tta_task_id,
                pretrainer=None, pretrainer_config=None, pretrainer_fold=None):
    assert pretrained_task_id != tta_task_id
    assert (
        pretrained_task_id in ['TS104_GIN', 'TS104_MIND', 'TS104_GIN_MIND'] \
        or isinstance(pretrained_task_id, int)
    )

    if isinstance(pretrained_task_id, int):
        # Check fold specifier
        assert pretrainer is not None
        assert pretrainer_config is not None
        assert pretrainer_fold == 'all' or isinstance(pretrainer_fold, int)
    else:
        if pretrained_task_id == 'TS104_GIN':
            pretrainer = 'nnUNetTrainer_GIN'
            pretrainer_config = '3d_fullres'
            pretrainer_fold = '0'

        elif pretrained_task_id == 'TS104_MIND':
            pretrainer = 'nnUNetTrainer_MIND'
            pretrainer_config = '3d_fullres'
            pretrainer_fold = '0'

        elif pretrained_task_id == 'TS104_GIN_MIND':
            pretrainer = 'nnUNetTrainer_GIN_MIND'
            pretrainer_config = '3d_fullres'
            pretrainer_fold = '0'

        else:
            raise ValueError()


    root_dir = Path(os.environ['DG_TTA_ROOT'])
    assert root_dir.is_dir()

    # Create directories
    _, plan_dir, results_dir, pretrained_task_name, tta_task_name = \
        get_tta_folders(pretrained_task_id, tta_task_id, pretrainer, pretrainer_config, pretrainer_fold)

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
        fold_dir = f"fold_{pretrainer_fold}" if pretrainer_fold != 'all' else pretrainer_fold
        results_pretrained_dataset_dir = Path(nnUNet_results, pretrained_task_name,
                                              f"{pretrainer}__nnUNetPlans__{pretrainer_config}",
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

    with open(plan_dir / "optimized_labels.json", 'w') as f:
        intersection_classes = list(set(pretrained_classes.keys()).intersection(set(tta_task_classes)))
        assert 'background' in intersection_classes, 'Background class must be present in both datasets!'
        intersection_classes.remove('background')
        intersection_classes.insert(0, 'background')
        json.dump(intersection_classes, f, indent=4)

    # Dump modifier functions
    modifier_src = inspect.getsource(ModifierFunctions)

    with open(plan_dir / "modifier_functions.py", 'w') as f:
        f.write("import torch\n\n")
        f.write(modifier_src)

    print(f"Preparation done. You can edit the plan, modifier functions and optimized labels in {plan_dir} prior to running TTA.")


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



def get_global_idx(list_of_tuple_idx_max):
    # Get global index e.g. 2250 for ensemble_idx=2, epoch_idx=250 @ max_epochs<1000
    global_idx = 0
    next_multiplier = 1

    # Smallest identifier tuple last!
    for idx, max_of_idx in reversed(list_of_tuple_idx_max):
        global_idx = global_idx + next_multiplier * idx
        next_multiplier = next_multiplier * 10**len(str(int(max_of_idx)))
    return global_idx



def load_current_modifier_functions(plan_dir):
    mod_path = Path(plan_dir / "modifier_functions.py")
    spec = importlib.util.spec_from_file_location("dg_tta.current_modifier_functions", mod_path)
    dyn_mod = importlib.util.module_from_spec(spec)
    sys.modules["dg_tta.current_modifier_functions"] = dyn_mod
    spec.loader.exec_module(dyn_mod)

    return dyn_mod