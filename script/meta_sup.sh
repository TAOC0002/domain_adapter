#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup_MaxH_Lambda_gem_t_lr2e-4_meta_lr5e-2_maxlr1e-3_thresh095\
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
  --num-epoch=60 \
  --exp-num -2 \
  --start-time=0 \
  --times=3 \
  --fc-weight=10.0 \
  --train=tta_meta_sup \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-lr=5e-2 \
  --meta-step=1 \
  --thresh=0.95 \
  --mix-lambda=0.75 \
  --meta-lambd-lr=5e-2 \
  --with-max \
  --meta-max-lr=1e-3 \
  --domain_bn_shift
  #--no-inner-lambda
  #--Transform