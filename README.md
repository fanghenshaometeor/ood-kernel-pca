# ood-kernel-pca
Kernel PCA for Out-of-Distribution Detection

## Pre-requisite
Prepare in-distribution and out-distribution data sets following the instructions in [this repo](https://github.com/deeplearning-wisc/knn-ood).
Then, modify the data paths in `utils_ood.py` and as yours.

Download the model files released by [this repo](https://github.com/deeplearning-wisc/knn-ood) and put them as
```
ood-kernel-pca
├── model
├── save
|   ├── CIFAR10
|   |   └── R18
|   |       ├── ce
|   |       |   └── checkpoint_100.pth.tar
|   |       └── supcon
|   |           └── checkpoint_500.pth.tar
|   └── ImageNet
|       └── R50
|           └── supcon
|               └── supcon.pth
├── ...
```

## Running
step.1. Run the `feat_extract.sh` to extract the penultimate features

step.2. 
- Run the `run_detection.sh` to obtain the detection results where only the KPCA-based reconstruction error serves as the detection score. 
- Run the `run_detection_fusion.sh` to obtain the detection results where the fusion strategy combining the KPCA-based reconstruction error and other detection scores (MSP,Energy,ReAct,BATS) is employed.