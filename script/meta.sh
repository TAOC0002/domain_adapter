#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_sup_gem_t_olambd0.8_lr1e-3_metalr1e-1_shift4e-2\
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
  --num-epoch=65 \
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
  --meta-lr=1e-1 \
  --inner weight bias\
  --domain_bn_shift \
  --mix-lambda=0.8 \
  --domain_bn_shift_p=4e-2
  #--sup_thresh=0.9 \
  #--with-max \
  #--max_bn_layer layer4 \
  #--meta-max-lr=5e-2
  #--meta-lambd-lr=2e-1\
