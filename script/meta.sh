#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_tta_meta_gem_t_lr1e3\
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-second-order=False \
  --TTA-head em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=80 \
  --exp-num -2 \
  --start-time=0 \
  --times=3 \
  --fc-weight=10.0 \
  --train=tta_meta \
  --eval=tta_meta \
  --s=1 \
  --loss-names=gem-t
  --loader=meta \
  --meta-step=1 \
  --meta-lr=1e-2
  #--meta-lambd-lr=5e-2 \
  #--mix-lambda=0.75 \
  #--thresh=0.75 \
  #--Transform

#python main.py \
#  --dataset='OfficeHome' \
#  --save-path='AdaBN/meta_norm_OfficeHome' \
#  --gpu=0 \
#  --do-train=True \
#  --meta-lr=0.1 \
#  --lr=1e-3 \
#  --data-root = '../../Datasets/DA' \
#  --replace \
#  --meta-step=1 \
#  --meta-second-order=False \
#  --TTA-head='norm' \
#  --model='DomainAdaptor' \
#  --backbone='resnet50' \
#  --batch-size=64 \
#  --num-epoch=30 \
#  \
#  --exp-num -2 \
#  --start-time=0 \
#  --times=5 \
#  --fc-weight=10.0 \
#  --train='tta_meta' \
#  --eval='tta_meta' \
#  --loader='meta' \
