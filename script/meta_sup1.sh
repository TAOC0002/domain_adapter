#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup1_gem_t_olambd0.80_lr1e-3_metalr1e-2_shift2e-2_maxlast_sup0.9_maxlr1e-2_isum\
  --gpu=0 \
  --do-train=True \
  --inneropt=sgd \
  --optimizer=sgd\
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
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup1 \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-step=1\
  --meta-lr=1e-2 \
  --inner weight bias\
  --domain_bn_shift \
  --mix-lambda=0.8 \
  --domain_bn_shift_p=2e-2 \
  --with-max \
  --sup_thresh=0.8 \
  --max_bn_layer layer4.1.bn2 \
  --meta-max-lr=1e-2
  #  --meta-lambd-lr=1e-1 \