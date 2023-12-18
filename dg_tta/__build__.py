import shutil
from pathlib import Path
import nnunetv2
from dg_tta import pretraining

def inject_dg_trainers_into_nnunet():
    dg_trainer_paths = Path(pretraining.__file__).parent.glob(
        "nnUNetTrainer*.py"
    )
    target_dir = Path(nnunetv2.__path__[0], "training/nnUNetTrainer/variants/dg_tta/")
    target_dir.mkdir(exist_ok=True)
    (target_dir / "__init__.py").touch()  # Make directory a mdoule

    for tr in dg_trainer_paths:
        shutil.copy(tr, target_dir)
