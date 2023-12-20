import re
from pathlib import Path
import nnunetv2
from dg_tta import pretraining


def inject_dg_trainers_into_nnunet(num_epochs=1000):
    dg_trainer_paths = Path(pretraining.__file__).parent.glob("nnUNetTrainer*.py")
    target_dir = Path(nnunetv2.__path__[0], "training/nnUNetTrainer/variants/dg_tta/")
    target_dir.mkdir(exist_ok=True)
    (target_dir / "__init__.py").touch()  # Make directory a mdoule

    for tr in dg_trainer_paths:
        # open file
        with open(tr, "r") as f:
            tr_code = f.read()
        tr_code_with_set_epochs = re.sub(
            r"self\.num_epochs = \d+", f"self.num_epochs = {num_epochs}", tr_code
        )

        with open(target_dir / tr.name, "w") as f:
            f.write(tr_code_with_set_epochs)


def check_trainers_injected():
    target_dir = Path(nnunetv2.__path__[0], "training/nnUNetTrainer/variants/dg_tta/")
    assert (
        target_dir.exists()
    ), "DG trainers not injected into nnUNet module. Please inject trainers first."
