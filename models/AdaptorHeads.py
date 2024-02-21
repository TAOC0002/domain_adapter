import torch
from torch import nn
from models.LAME import laplacian_optimization, kNN_affinity
from models.AdaptorHelper import get_new_optimizers

class Head(nn.Module):
    Replace = False
    ft_steps = 1

    def __init__(self, num_classes, in_ch, args):
        super(Head, self).__init__()
        self.args = args
        self.in_ch = in_ch
        self.num_classes = num_classes

    def forward(self, base_features, x, label, backbone, **kwargs):
        raise NotImplementedError()

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        lr = 0.05
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd')[0]
        ]

    def do_ft(self, backbone, x, label, **kwargs):
        return self.do_train(backbone, x, label, **kwargs)

    def do_test(self, backbone, x, label, **kwargs):
        return self.do_train(backbone, x, label, **kwargs)

    def do_train(self, backbone, x, label, **kwargs):
        logits = backbone(x)[-1]
        class_dict = {'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label}}
        return class_dict

class Losses():
    def __init__(self, s=1.0, sup_thresh=0, sup_weight=1):
        self.losses = {
            'em': self.em,
            'slr': self.slr,
            'norm': self.norm,
            'gem-t': self.GEM_T,
            'gem-skd': self.GEM_SKD,
            'gem-aug': self.GEM_Aug,
        }
        self.s = s
        self.sup_thresh = sup_thresh
        self.sup_weight = sup_weight

    def GEM_T(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        #T = self.s *logits.std(1, keepdim=True).detach() * 2
        if logits.shape[0] > 1:
            T = self.s *logits.std(0, keepdim=True).mean().detach()
            prob = (logits / T).softmax(1)
        else:
            prob = logits.softmax(1)
        #loss = - ((prob * prob.log()).sum(1) * (T ** 2)).mean()
        loss = - (prob * prob.log()).sum(1).mean()
        return loss

    def GEM_SKD(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        #T = self.s *logits.std(1, keepdim=True).detach() * 2
        T = self.s * logits.std(0, keepdim=True).mean().detach()
        original_prob = logits.softmax(1)
        prob = (logits / T).softmax(1)
        #loss = - ((original_prob.detach() * prob.log()).sum(1) * (T ** 2)).mean()
        loss = - (original_prob.detach() * prob.log()).sum(1).mean()
        return loss

    def GEM_Aug(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        #T =  self.s * logits.std(1, keepdim=True).detach() * 2
        T = self.s * logits.std(0, keepdim=True).mean().detach()
        aug_logits = kwargs['aug_logits']
        #loss = - ((aug_logits.softmax(1).detach() * (logits / T).softmax(1).log()).sum(1) * (T ** 2)).mean()
        loss = - (aug_logits.softmax(1).detach() * (logits / T).softmax(1).log()).sum(1).mean()
        return loss

    def em(self, logits, **kwargs):
        prob = (logits).softmax(1)
        loss = (- prob * prob.log()).sum(1)
        return loss.mean()

    def slr(self, logits, **kwargs):
        prob = (logits).softmax(1)
        return -(prob * (1 / (1 - prob + 1e-8)).log()).sum(1).mean()  # * 3 is enough = 82.7

    def norm(self, logits, **kwargs):
        return -logits.norm(dim=1).mean() * 2

    def get_loss(self, name, **kwargs):
        if 'em' in name and self.sup_thresh > 0:
            loss, sup_loss = None, None
            logits = kwargs['logits'].clone()
            conf = logits.softmax(1).max(1)[0]
            high = conf >= self.sup_thresh
            conf_logits, conf_label = logits[high], logits.argmax(1)[high]
            if len(conf_label) > 0:
                sup_loss = nn.functional.cross_entropy(conf_logits, conf_logits.softmax(1))
                #sup_loss = nn.functional.cross_entropy(conf_logits, conf_label)
            #res= {'sup': {'loss': sup_loss, 'weight': 1}}#kwargs['weight']}}
            w = len(conf_label)/len(logits)
            kwargs['logits'] = logits[conf < self.sup_thresh]
            if len(kwargs['logits']) > 0:
                loss = self.losses[name.lower()](**kwargs)
            res = {'sup': {'loss': sup_loss, 'weight': 2 * w}}
            res.update({name: {'loss': loss, 'weight': 2 * (1-w)}})
            #res.update({name: {'loss': loss, 'weight': kwargs['weight']}})
        else:
            res = {name: {'loss': self.losses[name.lower()](**kwargs), 'weight': kwargs['weight']}}
        return res
    
    def get_loss(self, name, **kwargs):
        if 'em' in name and self.sup_thresh > 0:
            loss, sup_loss = None, None
            logits = kwargs['logits']
            conf = logits.softmax(1).max(1)[0]
            high = conf >= self.sup_thresh
            conf_logits, conf_label = logits[high], logits.argmax(1)[high]
            if len(conf_label) > 0:
                sup_loss = nn.functional.cross_entropy(conf_logits, conf_label)
            res = {'sup': {'loss': sup_loss, 'weight': 2*len(conf_label)/ len(logits)}}
            #res= {'sup': {'loss': sup_loss, 'weight': kwargs['weight']}}
            kwargs['logits'] = logits[conf < self.sup_thresh]
            if len(kwargs['logits']) > 0:
                loss = self.losses[name.lower()](**kwargs)
            res.update({name: {'loss': loss, 'weight': 2*len(kwargs['logits'])/len(logits)}})
            #res.update({name: {'loss': loss, 'weight': kwargs['weight']}})
        else:
            res = {name: {'loss': self.losses[name.lower()](**kwargs), 'weight': kwargs['weight']}}

        return res

class EntropyMinimizationHead(Head):
    KEY = 'EM'
    ft_steps = 1

    def __init__(self, num_classes, in_ch, args):
        super(EntropyMinimizationHead, self).__init__(num_classes, in_ch, args)
        self.losses = Losses(s=args.s, sup_weight=args.sup_weight, sup_thresh=args.sup_thresh)
        self.loss_names = args.loss_names

    def get_cos_logits(self, feats, backbone):
        w = backbone.fc.weight  # c X C
        w, feats = nn.functional.normalize(w, dim=1), nn.functional.normalize(feats, dim=1)
        logits = (feats @ w.t())  # / 0.07
        return logits

    def label_rectify(self, feats, logits, thresh=0.95):
        # mask = self.get_confident_mask(logits, thresh=thresh)
        max_prob = logits.softmax(1).max(1)[0]
        normed_feats = feats / feats.norm(dim=1, keepdim=True)
        # N x N
        sim = (normed_feats @ normed_feats.t()) / 0.07
        # sim = feats @ feats.t()
        # select from high confident masks
        selected_sim = sim  # * max_prob[None]
        # N x n @ n x C = N x C
        rectified_feats = (selected_sim.softmax(1) @ feats)
        return rectified_feats + feats

    def do_lame(self, feats, logits):
        prob = logits.softmax(1)
        unary = - torch.log(prob + 1e-10)  # [N, K]

        feats = nn.functional.normalize(feats, p=2, dim=-1)  # [N, d]
        kernel = kNN_affinity(5)(feats)  # [N, N]

        kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        Y = laplacian_optimization(unary, kernel)
        return Y

    def do_ft(self, backbone, x, label, step=0, model=None, **kwargs):
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))

        if self.args.LAME:
            logits = self.do_lame(feats, logits)
            ret = {'LAME': {'acc_type': 'acc', 'pred': logits, 'target': label}}
        else:
             ret = {
                 'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            }
        if 'loss_name' in kwargs:
            loss_names = [kwargs['loss_name']]
        else:
            loss_names = self.loss_names

        for loss_name in loss_names:
            assert loss_name is not None

            if loss_name.lower() == 'gem-aug':
                with torch.no_grad():
                    aug_x = kwargs['tta']
                    n, N, C, H, W = aug_x.shape
                    aug_x = aug_x.reshape(n * N, C, H, W)
                    aug_logits = backbone(aug_x)[-1].view(n, N, -1).mean(1)
            else:
                aug_logits = None
            
            ret.update(self.losses.get_loss(loss_name, logits=logits, backbone=backbone, feats=feats,
                                           step=step, aug_logits=aug_logits, weight=kwargs['weight']))
        return ret


    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))

        if self.args.LAME:
            res = {'LAME': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': self.do_lame(feats, logits), 'target': label}}
        else:
            res = {
                'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label}
            }
        return res

    def do_test(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))

        if self.args.LAME:
            res = {'LAME': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': self.do_lame(feats, logits), 'target': label}}
        else:
            res = {
                'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label},
                'vis': {'feats': feats}
            }
        return res

    def setup(self, model, online):
        model.backbone.train()
        lr = self.args.lr
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(model, lr=lr, names=['bn'], opt_type='sgd', momentum=self.args.online)[0],
        ]

class RotationHead(Head):
    KEY = 'rotation'

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        lr = 0.05
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd')[0]
        ]

    def __init__(self, num_classes, in_ch, args):
        super(RotationHead, self).__init__(num_classes, in_ch, args)
        # self.shared = args.shared
        self.rotation_fc = nn.Linear(in_ch, 4, bias=False)
        # emb_dim = in_ch
        # self.rotation_fc = nn.Sequential(
        #     nn.Linear(in_ch, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, 4),
        # )

    def do_ft(self, backbone, x, label, **kwargs):
        logits = backbone(x)[-1]

        rotated_x, rotation_label = kwargs['rot_x'], kwargs['rot_label']
        l4 = backbone(rotated_x)[-2].mean((-1, -2))
        rotation_logits = self.rotation_fc(l4)

        return {
            'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            RotationHead.KEY: {'loss_type': 'ce', 'acc_type': 'acc', 'pred': rotation_logits, 'target': rotation_label,
                               'weight': kwargs['weight']}
        }

    def do_train(self, backbone, x, label, **kwargs):
        logits = backbone(x)[-1]

        rotated_x, rotation_label = kwargs['rot_x'], kwargs['rot_label']
        l4 = backbone(rotated_x)[-2].mean((-1, -2))

        rotation_logits = self.rotation_fc(l4)

        class_dict = {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label},
            RotationHead.KEY: {'loss_type': 'ce', 'acc_type': 'acc', 'pred': rotation_logits, 'target': rotation_label,
                               'weight': kwargs['weight']} #0.0}
        }
        return class_dict


class NormHead(Head):
    KEY = 'Norm'

    def __init__(self, num_classes, in_ch, args):
        super(NormHead, self).__init__(num_classes, in_ch, args)
        class MLP(nn.Module):
            def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
                super(MLP, self).__init__()
                self.norm_reduce = norm_reduce
                self.model = nn.Sequential(
                    nn.Linear(in_size, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_size),
                )

            def forward(self, x):
                out = self.model(x)
                if self.norm_reduce:
                    out = torch.norm(out)
                return out

        self.mlp = MLP(in_size=num_classes, norm_reduce=True)

    def do_ft(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        feats = base_features[-1]
        normed_loss = self.mlp(feats)
        return {
            'main': {'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
            NormHead.KEY: {'loss': normed_loss, 'weight': kwargs['weight']},
        }

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        feats = base_features[-1]
        normed_loss = self.mlp(feats)
        return {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
            NormHead.KEY: {'loss': normed_loss, 'weight': kwargs['weight']} #1
        }


class JigsawHead(Head):
    KEY = 'Jigsaw'

    def __init__(self, num_classes, in_ch, args):
        super(JigsawHead, self).__init__(num_classes, in_ch, args)
        jigsaw_classes = 32
        emb_dim = in_ch
        # self.jigsaw_classifier = nn.Linear(in_ch, jigsaw_classes)
        self.jigsaw_classifier = nn.Sequential(
            nn.Linear(in_ch, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, jigsaw_classes),
        )
        self.i = 0

    def do_ft(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits = base_features[-1]

        jig_features = backbone(kwargs['jigsaw_x'])[-2]
        jig_features = jig_features.mean((-1, -2))
        jig_logits = self.jigsaw_classifier(jig_features)
        return {
            'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            JigsawHead.KEY: {'acc_type': 'acc', 'pred': jig_logits, 'target': kwargs['jigsaw_label'], 'loss_type': 'ce',
                             'weight': kwargs['weight']},
        }

    def train(self, mode=True):
        super(JigsawHead, self).train(mode)
        self.i = 0

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits = base_features[-1]
        ret = {
                'main': {'acc_type': 'acc', 'pred': logits, 'target': label, 'loss_type': 'ce'},
            }
        # if self.i == 0 or random.random() > 0.9:
        #     self.i = 1
        if True:
            jig_features = backbone(kwargs['jigsaw_x'])
            jig_features = jig_features[-2].mean((-1, -2))
            jig_logits = self.jigsaw_classifier(jig_features)
            ret.update({
                    JigsawHead.KEY: {'acc_type': 'acc', 'pred': jig_logits, 'target': kwargs['jigsaw_label'],
                                     'loss_type': 'ce', 'weight': kwargs['weight']} #0.1},
                    # 'jig_cls': {'acc_type': 'acc', 'pred': jig_class_logits, 'target': label, 'loss_type': 'ce', 'weight':0.5},
                })
        return ret

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        # online best : 0.01
        # not online  : 0.02?
        lr = 0.01 # 0.0005 is better for MDN
        print(f"Learning rate : {lr} ")
        return get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd', momentum=online)[0]


class NoneHead(Head):
    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        return {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
        }

