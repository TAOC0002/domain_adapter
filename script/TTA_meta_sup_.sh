#!/bin/bash


python main.py \
  --gpu=0 \
  --load-path=pretrained_models/resnet18_PACS \
  --save-path=test \
  --do-train=False \
  --dataset=PACS \
  --loss-names=gem-t \
  --TTAug \
  --TTA-bs=3 \
  --TTA-head=em \
  --shuffled=True \
  --eval=tta_meta_sup \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --exp-num=0 \
  --start-time=0 \
  --times=5 \
