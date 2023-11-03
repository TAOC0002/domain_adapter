#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup1_Max_Lambda_gem_t_lr2e-4_maxlr_1e2_thresh075 \
  --gpu=0 \
  --do-train=True \
  --lr=2e-4 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-second-order=False \
  --TTA-head em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=30 \
  --exp-num 0 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-lr=5e-2 \
  --meta-step=1 \
  --thresh=0.75 \
  --mix-lambda=0.75 \
  --meta-lambd-lr=5e-2 \
  --with-max \
  --meta-max-lr=1e-3
  #--no-inner-lambda
  #--domain_bn_shift
  #--Transform