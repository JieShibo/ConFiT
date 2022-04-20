# ConFiT
Source code of "Alleviating Representational Shift for Continual Fine-tuning" (CVPR-W 2022)
### Requirements
+ Python 3.9.1
+ PyTorch 1.10.0
+ TorchVision 0.11.1
+ [Timm](https://rwightman.github.io/pytorch-image-models/) 0.4.12
+ [Avalanche](https://avalanche.continualai.org/) 0.0.1 
### Run
```sh 
# CIFAR100
python main.py --dataset cifar --max_epoch 10 --batch_size 128
# CUB200
python main.py --dataset cub --max_epoch 50 --batch_size 32
# CALTECH101
python main.py --dataset caltech --max_epoch 20 --batch_size 32
# FLOWERS102
python main.py --dataset flowers --max_epoch 10 --batch_size 32
```
