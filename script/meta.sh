#!/bin/bash

python main.py \
  --dataset=VISDA \
  --save-path=Results/VISDA/resnets18/sup_gem_t_lr1e-3_metalr5e-2_shift2e-2\
  --data-root=../../Datasets/DA \
  --backbone=resnet18 \
  --gpu=0 \
  --do-train=True \
  --inneropt=sgd \
  --optimizer=sgd\
  --lr=1e-3 \
  --replace \
  --meta-second-order=False \
  --TTA-head em \
  --model=DomainAdaptor \
  --batch-size=64 \
  --num-epoch=5 \
  --exp-num -2 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-step=1\
  --meta-lr=5e-2 \
  --domain_bn_shift_p=2e-2
  #--inner weight bias\
  #--domain_bn_shift \
  #--mix-lambda=0.8 \
  #--sup_thresh=0.9 \
  #--with-max \
  #--max_bn_layer layer4 \
  #--meta-max-lr=5e-2
  #--meta-lambd-lr=2e-1\
