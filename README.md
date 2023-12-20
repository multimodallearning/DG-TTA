# DG-TTA: Out-of-domain medical image segmentation through Domain Generalization and Test-Time Adaptation

## Installation
We have a package available on pypi. Just run:

`pip3 install dg-tta`

Optionally, you can install `wandb` to log results to your dashboard.

### nnUNet dependency
The nnUNet framework will be installed automatically alongside DG-TTA. Please refer to https://github.com/MIC-DKFZ/nnUNet to prepare your datasets.
DG-TTA needs datasets to be prepared according to the v2 version of nnUNet.

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## Usage
Run `dgtta -h` from your commandline interface to get started.
There are four basic commands available:
1) `dgtta inject_trainers`: This will copy our specialized trainers with DG techniques and make them available in the nnUNet framework
2) `dgtta pretrain`: Use this command to pre-train a model on a (CT) dataset with one of our trainers.
3) `dgtta prepare_tta`: After pre-training, prepare TTA by specifying the source and target dataset
4) `dgtta run_tta`: After preparation, you can run TTA on a target (MRI) dataset and evaluate how well the model bridged the domain gap.
5) If you want to perform TTA without pre-training, you can skip step 2) and use our pre-trained models (pre-trained on the [TotalSegmentator dataset](https://github.com/wasserth/TotalSegmentator) dataset)

### Examples
Prepare a `paths.sh` file which exports the following variables:
```bash
#!/usr/bin/bash
export nnUNet_raw="/path/to/dir"
export nnUNet_preprocessed="/path/to/dir"
export nnUNet_results="/path/to/dir"
export DG_TTA_ROOT="/path/to/dir"
```

1) Use case: Get to know the tool
  * `source paths.sh && dgtta`

2) Use case: Pre-train a GIN_MIND model on dataset 802 in nnUNet
  * `source paths.sh && dgtta inject_trainers --num_epochs 150`
  * `source paths.sh && dgtta pretrain -tr nnUNetTrainer_GIN_MIND 802 3d_fullres 0`

3) Use case: Run TTA on dataset 678 for the pre-trained model of step 2)
  * Inject trainers: `source paths.sh && dgtta inject_trainers`
  * Prepare TTA: `source paths.sh && dgtta prepare_tta 802 678 --pretrainer nnUNetTrainer_GIN --pretrainer nnUNetTrainer_GIN_MIND --pretrainer_config 3d_fullres --pretrainer_fold 0 --tta_dataset_bucket imagesTrAndTs`
  * Now inspect and change the `plan.json` (see console output of preparation). E.g. remove some samples on which you do not want to perform TTA on, change the number of TTA epochs etc.
  * Also inspect the notebook inside the plans folder and visualize the dataset orientation. Modify functions of `modifier_functions.py` as explained in the notebook to get the input/output orientation of the TTA data right.
  * Run TTA: `source paths.sh && dgtta run_tta 802 678 --pretrainer nnUNetTrainer_GIN_MIND --pretrainer_config 3d_fullres --pretrainer_fold 0`
  * Find the results inside the DG_TTA_ROOT directory

4) Use case: Run TTA on dataset 678 with our pre-trained GIN model:
  * Inject trainers: `source paths.sh && dgtta inject_trainers`
  * Prepare TTA: `source paths.sh && dgtta prepare_tta TS104_GIN 678 --tta_dataset_bucket imagesTrAndTs`
  * Run TTA: `source paths.sh && dgtta run_tta TS104_GIN 678`

## Please refer to our work
If you used DG-TTA, please cite:

Weihsbach, C., Kruse, C. N., Bigalke, A., & Heinrich, M. P. (2023). DG-TTA: Out-of-domain medical image segmentation through Domain Generalization and Test-Time Adaptation. arXiv preprint arXiv:2312.06275.

https://arxiv.org/abs/2312.06275
