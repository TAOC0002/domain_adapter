#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup_gem_t_maxH75_100_olambda75_lr1e3_metalr5e2\
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
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-step=1 \
  --meta-lr=5e-2 \
  --mix-lambda=0.75 \
  --inner bias weight \
  --thresh=0.75 \
  --with-max
  #--Transform
  # --meta-lambd-lr=5e-2 \
