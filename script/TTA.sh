#!/bin/bash


CUDA_VISIBLE_DEVICES="1" python main.py \
  --gpu=0 \
  --load-path='pretrained_models/cifar10c/resnet50' \
  --save-path='pretrained_models/cifar10c/resnet50/TTA-gem-t' \
  --do-train=False \
  --dataset=cifar10c \
  --loss-names=gem-t \
  --TTAug \
  --TTA-bs=3 \
  --TTA-head=em \
  --shuffled=True \
  --eval=tta_ft \
  --model=DomainAdaptor \
  --backbone=resnet50 \
  --batch-size=64 \
  --exp-num=1 \
  --start-time=0 \
  --times=1 \
  --corruption fog \


