#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup1_gem_t_olambd0.75_lr1e3_metalr5e2_bnshift\
  --gpu=0 \
  --do-train=True \
  --inneropt=sgd \
  --optimizer=sgd \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-second-order=False \
  --TTA-head em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=60 \
  --exp-num -2 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --eval=tta_meta_sup \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-step=1\
  --meta-lr=5e-2 \
  --inner weight bias\
  --domain_bn_shift \
  --mix-lambda=0.75
  #--meta-lambd-lr=5e-2\
  #--sup_thresh=0.9
#  --max_thresh=0.75 \
#  --meta_max_lr=1e-2 \
#  --with-max
  #--Transform
  #--no-inner-lambda \
  #--optimizer=adam \