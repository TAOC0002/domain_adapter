#!/bin/bash

declare -a nets=("resnet18" "resnet50")

for step in 1 5; do
  for s in 1; do
    for net in ${nets[@]}; do
      python main.py \
        --dataset=PACS  \
        --save-path=Results/Ours/debug/PACS/${net}_s${s}_step${step} \
        --gpu=0 \
        --do-train=True \
        --meta-lr=5e-2 \
        --lr=1e-3 \
        --data-root=../../Datasets/DA \
        --replace \
        --meta-step=$step\
        --meta-second-order=False \
        --TTA-head=em \
        --model=DomainAdaptor \
        --backbone=$net \
        --batch-size=64 \
        --num-epoch=60 \
        --exp-num 0 \
        --start-time=0 \
        --times=1 \
        --fc-weight=10.0 \
        --train=tta_meta \
        --loader=meta \
        --sup_weight=1 \
        --mix-lambda=0.9 \
        --thresh=0.7 \
        --loss-names gem-aug norm\
        --BN-start=4 \
        --s=$s \
        --eval=tta_meta_sup \
        --TTAug
        #--load-path=Results/ERM/resnet18_PACS
    done
  done
done
