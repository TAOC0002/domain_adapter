#!/bin/bash


python main.py \
  --gpu=0 \
  --load-path=Results/Ours/debug/PACS/resnets18_sup1_Max_Lambda_gem_t_lr2e-4_meta_lr5e-2_maxlr1e-3_thresh075 \
  --save-path=test \
  --do-train=False \
  --dataset=PACS \
  --data-root=../../Datasets/DA
  --loss-names=gem-t \
  --TTAug \
  --TTA-bs=3 \
  --TTA-head=em \
  --shuffled=True \
  --eval=tta_ft \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --with-max

