import os
from pathlib import Path


def enable_internal_augmentation():
    os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "true"


def disable_internal_augmentation():
    os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "false"


def check_internal_augmentation_disabled():
    assert os.environ.get("DG_TTA_INTERNAL_AUGMENTATION").lower() != "true"


def get_internal_augmentation_enabled():
    return os.environ.get("DG_TTA_INTERNAL_AUGMENTATION").lower() == "true"


def check_dga_root_is_set(soft_check=False):
    prompt = "Please define an existing root directory for DG-TTA by setting DG_TTA_ROOT."
    set_path = os.environ.get("DG_TTA_ROOT", "_")
    check = Path(set_path).is_dir()

    if soft_check and not check:
        print(prompt)
        return

    assert check, prompt


def set_environ_vars_from_paths_sh(sh_path):
    with open(sh_path, "r") as f:
        paths = f.readlines()
    vars = [p.replace('export', '').split('=') for p in paths]
    vars = [(v[0].strip(), v[1].strip().replace(r'"', r'').replace(r"'", r'')) for v in vars]

    for var, path in vars:
        os.environ[var] = path