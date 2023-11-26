#!/bin/bash

python main.py \
  --dataset=PACS \
  --save-path=Results/PACS/resnet18/sup_gem_t_lr1e-3_metalr1e-1_olambd0.75_maxlr1e-1_maxlay4.1.bn2_sup0.75_shift2e-2_J2\
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
  --num-epoch=50 \
  --exp-num -2 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=tta_meta_sup \
  --eval=tta_meta_sup1 \
  --s=1 \
  --loss-names=gem-t \
  --loader=meta \
  --meta-step=2\
  --meta-lr=1e-1 \
  --inner weight bias\
  --domain_bn_shift \
  --mix-lambda=0.75 \
  --domain_bn_shift_p=2e-2 \
  --sup_thresh=0.75 \
  --with-max \
  --max_bn_layer layer4.1.bn2
  #--meta-max-lr=1e-1
  # --meta-lambd-lr=1e-1\
