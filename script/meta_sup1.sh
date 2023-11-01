#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=pretrained_models/PACS/resnet18-bs64/AS/3 \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root=../data \
  --replace \
  --meta-second-order=False \
  --TTA-head em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=30 \
  --exp-num -1 \
  --start-time=0 \
  --times=0 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=normal \
  --meta-step=1 \
  --meta-lr=1e-2 \
  --thresh=0.75 \
  #--meta-lambd-lr=5e-2 \
  #--mix-lambda=0.75 \
  #--Transform