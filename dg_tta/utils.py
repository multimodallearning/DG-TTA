import os


def enable_internal_augmentation():
    os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "true"


def disable_internal_augmentation():
    os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "false"


def check_internal_augmentation_disabled():
    assert os.environ.get("DG_TTA_INTERNAL_AUGMENTATION").lower() != "true"


def get_internal_augmentation_enabled():
    return os.environ.get("DG_TTA_INTERNAL_AUGMENTATION").lower() == "true"
