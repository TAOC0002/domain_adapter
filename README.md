# Meta-TTT

Adapted from the paper 《 [Meta-TTT: A Meta-learning Minimax Framework For Test-Time Training](https://arxiv.org/abs/2410.01709) 》

Test-time domain adaptation is a challenging task that aims to adapt a pre-trained model to limited, unlabeled target data during inference. Current methods that rely on self-supervision and entropy minimization underperform when the self-supervised learning (SSL) task does not align well with the primary objective. Additionally, minimizing entropy can lead to suboptimal solutions when there is limited diversity within minibatches. This paper introduces a meta-learning minimax framework for test-time training on batch normalization (BN) layers, ensuring that the SSL task aligns with the primary task while addressing minibatch overfitting. 

## Dataset structure

```
VLCS
├── CALTECH
│   ├── crossval
│   ├── full
│   ├── test
│   └── train
├── LABELME
│   ├── crossval
│   ├── full
| ...
OfficeHome
├── Art
│   ├── Alarm_Clock
│   ├── Backpack
│   ├── Batteries
│   ├── Bed
│   ├── Bike
│   ├── Bottle
| ...
```


## Run the code

```bash
bash scripts/scrip1.sh
```
you may pass the following arguments to adjust hyperparameters in the experiemnts:
1. --with-max: Replace entropy with minimax entropy optimization
2. --train=tta-meta-sup: Rrain source model with the proposed meta-learning framework for test-time-training 
3. --eval=tta-meta-sup: Adapt source model to test streams with self-supervised learning
4. --loader=meta: Select the dataloader suitable for the meta-learning framework
5. --meta-step=1: Steps to perfrom meta learning
6. --domain_bn_shift: Enable data augmentation per domain
7. --max_bn_layer: Batch norm layers to maximize the entropy of their shift parameters
8. --sup_thresh: Threshold to determine confident samples in self-supervised learning
