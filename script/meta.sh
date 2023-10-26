#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_meta_gem_t_noTran \
  --gpu=0 \
  --do-train=True \
  --meta-lr=1e-2 \
  --meta-lambd-lr=1e-2 \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-step=1 \
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
  --loader=meta \
  --mix-lambda=0.9 \
  --thresh=0 \
  --s=1 \
  --train=tta_meta \
  --eval=tta_meta \
  --loss-names=gem-t

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_meta_gem_t_l5e2_noTran \
  --gpu=0 \
  --do-train=True \
  --meta-lr=1e-2 \
  --meta-lambd-lr=5e-2 \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-step=1 \
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
  --loader=meta \
  --mix-lambda=0.9 \
  --thresh=0 \
  --s=1 \
  --train=tta_meta \
  --eval=tta_meta \
  --loss-names=gem-t

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_meta_gem_t_l5e2_m_noTran \
  --gpu=0 \
  --do-train=True \
  --meta-lr=1e-2 \
  --meta-lambd-lr=5e-2 \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-step=1 \
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
  --loader=meta \
  --mix-lambda=0.9 \
  --thresh=0 \
  --s=1 \
  --train=tta_meta \
  --eval=tta_meta \
  --loss-names=gem-t \
  --bn-momentum
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
