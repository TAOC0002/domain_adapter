#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup1_woMax_woLambda_gem_t_lr1e3 \
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
  --num-epoch=100 \
  --exp-num -2 \
  --start-time=0 \
  --times=3 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t
  --loader=meta \
  --meta-step=1 \
  --thresh=0 \
  --meta-lr=1e-2
  #--meta-lambd-lr=5e-2 \
  #--mix-lambda=0.75 \
  #--thresh=0.75
  #--Transform