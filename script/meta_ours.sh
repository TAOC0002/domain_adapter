#!/bin/bash

declare -a losses=("em" "gem-t" "gem-aug")
declare -a nets=("resnet18" "resnet50")

for step in 1 5; do
  for s in 1 5; do
    for loss in ${losses[@]}; do
    for net in ${nets[@]}; do
      python main.py \
        --dataset=PACS  \
        --save-path=Results/Ours/debug/PACS/${net}_s${s}_step${step}_l${loss} \
        --gpu=0 \
        --do-train=True \
        --meta-lr=1e-3 \
        --lr=5e-4 \
        --data-root=../../Datasets/DA \
        --replace \
        --meta-step=$step \
        --meta-second-order=False \
        --TTA-head em \
        --model=DomainAdaptor \
        --backbone=resnet18 \
        --batch-size=64 \
        --num-epoch=60 \
        --exp-num 0 \
        --start-time=0 \
        --times=1 \
        --fc-weight=10.0 \
        --loader=meta \
        --sup_weight=1 \
        --main_weight 0 \
        --sl_weight 1 \
        --mix-lambda=0.9 \
        --thresh=0.7 \
        --s=$s \
        --train=tta_meta_sup \
        --eval=tta_meta_sup \
        --BN-start=0 \
        --loss-names=$loss \
        --domain_mixup
        #--domain_bn_shift \
        #--load-path=Results/ERM/resnet18_PACS
    done
  done
done
done