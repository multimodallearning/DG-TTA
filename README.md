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
5) If you want to perform TTA without pre-training, you can skip steps 1) and 2) and use our pre-trained models (pre-trained on the [TotalSegmentator dataset](https://github.com/wasserth/TotalSegmentator) dataset)

## Please refer to our work
If you used DG-TTA, please cite:

Weihsbach, C., Kruse, C. N., Bigalke, A., & Heinrich, M. P. (2023). DG-TTA: Out-of-domain medical image segmentation through Domain Generalization and Test-Time Adaptation. arXiv preprint arXiv:2312.06275.

https://arxiv.org/abs/2312.06275
