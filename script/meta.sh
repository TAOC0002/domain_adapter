#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python main.py \
  --dataset=PACS \
  --save-path=pretrained_models/PACS/resnet18-bs64/best/trial3 \
  --gpu=0 \
  --do-train=True \
  --meta-lr=1e-2 \
  --meta-lambd-lr=1e-2 \
  --lr=1e-3 \
  --data-root=../data \
  --replace \
  --meta-step=2 \
  --meta-second-order=False \
  --TTA-head=em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=60 \
  --exp-num=-1 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --loader=meta \
  --mix-lambda=0.75 \
  --thresh=0.85 \
  --s=1 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup \
  --loss-names=gem-t \
  --domain_bn_shift
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
