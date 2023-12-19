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
    check = Path(
        os.environ.get("DG_TTA_ROOT", "_")
    ).is_dir()

    if soft_check and not check:
        print(prompt)
        return

    assert check, prompt
