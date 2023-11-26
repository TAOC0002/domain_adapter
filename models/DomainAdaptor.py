import copy
import functools
import warnings

import torch

from framework.ERM import ERM
from framework.loss_and_acc import *
from framework.registry import EvalFuncs, Models
from models.AdaptorHeads import RotationHead, NormHead, NoneHead, JigsawHead, EntropyMinimizationHead
from models.AdaptorHelper import get_new_optimizers, convert_to_target
from utils.tensor_utils import to, AverageMeterDict, zero_and_update


warnings.filterwarnings("ignore")
np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "{:.4f},  ".format(x)))


class AdaMixBN(nn.BatchNorm2d):
    # AdaMixBn cannot be applied in an online manner.
    def __init__(self, in_ch, lambd=None, transform=True, mix=True, idx=0):
        super(AdaMixBN, self).__init__(in_ch)
        if lambd is not None:
            self.lambd = nn.Parameter(torch.ones(1, in_ch, 1, 1))
            #self.lambd.requires_grad = True
            torch.nn.init.normal(self.lambd, mean=lambd, std=1e-2)
            self.lambd_trainable = True
        else:
            self.lambd_trainable = False
        self.rectified_params = None
        self.transform = transform
        self.layer_idx = idx
        self.mix = mix

    def get_retified_gamma_beta(self, src_mu, src_var, cur_mu, cur_var):
        C = src_mu.shape[1]
        new_gamma = (cur_var + self.eps).sqrt() / (self.lambd * src_var + (1 - self.lambd) * cur_var + self.eps).sqrt() * self.weight.view(1, C, 1, 1)
        new_beta = self.lambd * (cur_mu - src_mu) / (cur_var + self.eps).sqrt() * new_gamma + self.bias.view(1, C, 1, 1)
        return new_gamma.view(-1), new_beta.view(-1)

    def get_lambd(self, x, src_mu, src_var, cur_mu, cur_var):
        instance_mu = x.mean((2, 3), keepdims=True)
        instance_std = x.std((2, 3), keepdims=True)

        it_dist = ((instance_mu - cur_mu) ** 2).mean(1, keepdims=True) + ((instance_std - cur_var.sqrt()) ** 2).mean(1, keepdims=True)
        is_dist = ((instance_mu - src_mu) ** 2).mean(1, keepdims=True) + ((instance_std - src_var.sqrt()) ** 2).mean(1, keepdims=True)
        st_dist = ((cur_mu - src_mu) ** 2).mean(1)[None] + ((cur_var.sqrt() - src_var.sqrt()) ** 2).mean(1)[None]

        src_lambd = 1 - (st_dist) / (st_dist + is_dist + it_dist)

        src_lambd = torch.clip(src_lambd, min=0, max=1)
        return src_lambd

    def get_mu_var(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)

        if not self.lambd_trainable:
            self.lambd = self.get_lambd(x, src_mu, src_var, cur_mu, cur_var).mean(0, keepdims=True)

        if self.transform:
            if self.rectified_params is None:
                new_gamma, new_beta = self.get_retified_gamma_beta(src_mu, src_var, cur_mu, cur_var)
                # self.test(x, lambd, src_mu, src_var, cur_mu, cur_var, new_gamma, new_beta)
                self.weight.data = new_gamma.data
                self.bias.data = new_beta.data
                self.rectified_params = new_gamma, new_beta
            return cur_mu, cur_var
        else:
            new_mu = self.lambd * src_mu + (1 - self.lambd) * cur_mu
            new_var = self.lambd * src_var + (1 - self.lambd) * cur_var + (1 - self.lambd)*self.lambd*torch.pow((src_mu-cur_mu), 2)
            return new_mu, new_var

    def forward(self, x):
        n, C, H, W = x.shape
        new_mu = x.mean((0, 2, 3), keepdims=True)
        new_var = x.var((0, 2, 3), keepdims=True)

        #if self.training:
        if self.mix:
            new_mu, new_var = self.get_mu_var(x)

        # Normalization with new statistics
        inv_std = 1 / (new_var + self.eps).sqrt()
        new_x = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
        return new_x
        #else:
        #    return super(AdaMixBN, self).forward(x)

    def reset(self):
        self.rectified_params = None

    def test_equivalence(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)
        lambd = 0.9

        new_gamma, new_beta = self.get_retified_gamma_beta(x, lambd, src_mu, src_var, cur_mu, cur_var)
        inv_std = 1 / (cur_var + self.eps).sqrt()
        x_1 = (x - cur_mu) * (inv_std * new_gamma.view(1, C, 1, 1)) + new_beta.view(1, C, 1, 1)

        new_mu = lambd * src_mu + (1 - lambd) * cur_mu
        new_var = lambd * src_var + (1 - lambd) * cur_var
        inv_std = 1 / (new_var + self.eps).sqrt()
        x_2 = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
        assert (x_2 - x_1).abs().mean() < 1e-5
        return x_1, x_2


@Models.register('DomainAdaptor')
class DomainAdaptor(ERM):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(DomainAdaptor, self).__init__(num_classes, pretrained, args)
        heads = {
            'em': EntropyMinimizationHead,
            'rot': RotationHead,
            'norm': NormHead,
            'none': NoneHead,
            'jigsaw': JigsawHead,
        }
        self.heads = [heads[head.lower()](num_classes, self.in_ch, args) for head in args.TTA_head]
        self.train_weights = args.main_weight
        self.ft_weights = args.sl_weight
        if args.AdaMixBN:
            self.bns = list(convert_to_target(self.backbone, functools.partial(AdaMixBN, transform=args.Transform, lambd=args.mix_lambda),
                                              verbose=False, res50=args.backbone == 'resnet50')[-1].values())

    def set_momentum(self, momentum):
        for bn in self.bns:
            bn.momentum = momentum

    def reset_shift_bn(self):
        nn.init.constant_(self.backbone.shift_weight.data, 1)
        nn.init.constant_(self.backbone.shift_bias.data, 0)
        #for npp, p in self.backbone.bn0.named_parameters():
        #    if npp in ['weight', 'bias']:
        #        with torch.no_grad():
        #            p.data = 0
    def StochasticBNShift(self, e=2e-2):
        # Stochastic bn shift
        with torch.no_grad():
            self.reset_shift_bn()
            mask = (torch.rand(self.backbone.shift_weight.shape)<e).float().cuda()
            self.backbone.shift_weight = torch.rand(self.backbone.shift_weight.shape).float().cuda() * mask + (1 - mask) * self.backbone.shift_weight
            mask = (torch.rand(self.backbone.shift_bias.shape) < e).float().cuda()
            self.backbone.shift_bias = torch.rand(self.backbone.shift_bias.shape).float().cuda() * mask + (
                        1 - mask) * self.backbone.shift_bias

    def step(self, x, label, train_mode='test', **kwargs):
        res = {}
        if train_mode == 'train':
            # res = self.head.do_train(self.backbone, x, label, model=self, **kwargs)
            res.update(self.heads[0].do_train(self.backbone, x, label, model=self, weight=self.train_weights[0], **kwargs))
        elif train_mode == 'test':
            # res = self.head.do_test(self.backbone, x, label, model=self, **kwargs)
            res.update(self.heads[0].do_test(self.backbone, x, label, model=self, weight=self.train_weights[0], **kwargs))
        elif train_mode == 'ft':
            for i, head in enumerate(self.heads):
                res = head.do_ft(self.backbone, x, label, model=self, weight=self.ft_weights[i], **kwargs)
                if res:
                    res.update(res)
        else:
            raise Exception("Unexpected mode : {}".format(train_mode))
        return res
    def head_to(self, device):
        for i in range(len(self.heads)):
            self.heads[i].to(device)

    def finetune(self, data, optimizers, loss_name, running_loss=None, running_corrects=None):
        if hasattr(self, 'bns'):
            [bn.reset() for bn in self.bns]

        with torch.enable_grad():
            res = None
            for i in range(self.head.ft_steps):
                o = self.step(**data, train_mode='ft', step=i, loss_name=loss_name)
                meta_train_loss = get_loss_and_acc(o, running_loss, running_corrects, prefix=f'ft_A{i}_')
                zero_and_update(optimizers, meta_train_loss)
                if i == 0:
                    res = o
            return res
    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def setup(self, online):
        return self.head.setup(self, online)


@EvalFuncs.register('tta_ft')
def test_time_adaption(model, eval_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    model_to_ft = copy.deepcopy(model)
    original_state_dict = model.state_dict()
    model.eval()
    online = args.online
    optimizers = model_to_ft.setup(online)
    loss_names = args.loss_names  # 'gem-t', 'gem-skd', 'gem-tta']

    with torch.no_grad():
        for i, data in enumerate(eval_data):
            data = to(data, device)
            # Normal Test

            get_loss_and_acc(model(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

            # test-time adaptation to a single batch
            for loss_name in loss_names:
                # recover to the original weight
                model_to_ft.load_state_dict(original_state_dict) if (not online) else ""

                # adapt to the current batch
                adapt_out = model_to_ft.finetune(data, optimizers, loss_name, running_loss, running_corrects)

                # get the adapted result
                cur_out = model_to_ft(**data, train_mode='test')

                get_loss_and_acc(cur_out, running_loss, running_corrects, prefix=f'{loss_name}_')
                if loss_name == loss_names[-1]:
                    get_loss_and_acc(cur_out, running_loss, running_corrects)  # the last one is recorded as the main result

    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    return acc['main'], (loss, acc)