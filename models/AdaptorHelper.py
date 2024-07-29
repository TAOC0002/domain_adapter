import torch
from torch import nn as nn


def module_has_string(has_strings, x):
    # No string required, all layers can be converted
    if len(has_strings) == 0:
        return True

    # Only modules with names contain one string in has_strings can be converted
    for string in has_strings:
        if string in x:
            return True
    return False


def has_string(has_strings, x):
    # No string required, all layers can be converted
    if len(has_strings) == 0:
        return False

    # Only modules with names contain one string in has_strings can be converted
    for string in has_strings:
        if string in x:
            return True
    return False


def collect_module_params(model, module_class, param_names=None, has_strings=[]):
    params = []
    names = []
    nnn = []
    for nm, m in model.named_modules():
        if has_string(nnn, nm) or 'downsample' in nm :
            continue
        #if any(l in nm for l in layer_name) or layer_name is None:
        if isinstance(m, module_class) and module_has_string(has_strings, nm):
            named_params = m.named_parameters()
            for np, p in named_params:
                if param_names is None:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                elif np in param_names:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def set_param_trainable(model, module_names, requires_grad, param_names=None):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    set_names = []
    for name in module_names:
        _params, _names = collect_module_params(model, classes[name], param_names)
        for p in _params:
            p.requires_grad = requires_grad
        set_names.extend(_names)
    return set_names

def remove_param_grad(model, module_names, param_names):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    set_names = []
    for name in module_names:
        _params, _names = collect_module_params(model, classes[name], param_names)
        for p in _params:
            p.grad = None
        set_names.extend(param_names)
    return set_names


def get_optimizer(opt_dic, lr, opt_type='sgd', momentum=True):
    if opt_type == 'sgd':
        if momentum:
            opt = torch.optim.SGD(opt_dic, lr=lr, momentum=0.9)
        else:
            opt = torch.optim.SGD(opt_dic, lr=lr)
    else:
        opt = torch.optim.Adam(opt_dic, lr=lr, betas=(0.9, 0.999))
    return opt


def get_new_optimizers(model, lr=1e-4, names=('bn', 'conv', 'fc'), param_names=None,
                       opt_type='sgd', layers=None, nonpara_name=[], lambd_lr=None, momentum=False):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    opt_dic = []
    if lambd_lr is None:
        lambd_lr = lr
    added = []
    total = 0
    for name in names:
        name = name.lower()
        params, _names = collect_module_params(model, module_class=classes[name], param_names=param_names)
        for param, n in zip(params, _names):
            if layers is not None:
                 if not any(l in n for l in layers): continue
            if n not in nonpara_name:
                if 'lambd' in n:
                    opt_dic.append({'params': param, 'lr': lambd_lr})
                else:
                    opt_dic.append({'params': param, 'lr': lr})
                added.append(n)
                total += param.size(dim=0)
        # print(added)
        # print(total)
    opt = get_optimizer(opt_dic, lr, opt_type, momentum)
    return opt, added

def get_mme_optimizers(model, names=('bn', 'conv', 'fc'), max_L=None, min_L=None, opt_type='sgd',
                       max_lr=1e-3, min_lr=1e-3, lambda_lr=1e-3, max_param=None, min_param=None, momentum=False):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    max_params = []
    max_names = []
    for name in names:
        name = name.lower()
        params, _names = collect_module_params(model, module_class=classes[name], param_names=max_param)
        for param, n in zip(params, _names):
            if max_L is not None:
                if not any(l in n for l in max_L): continue
            max_params.append(param)
            max_names.append(n)

    min_names = []
    dic = [{"params": max_params, "lr": max_lr}]
    lrs = [max_lr]
    min_lambd_params = []
    min_params = []
    for name in names:
        name = name.lower()
        params, _names = collect_module_params(model, module_class=classes[name], param_names=min_param)
        for param, n in zip(params, _names):
            if min_L is not None:
                if not any(l in n for l in min_L): continue
            if n not in max_names:
                if 'lambd' in n:
                    min_lambd_params.append(param)
                else:
                    min_params.append(param)
                min_names.append(n)
    if min_params:
        dic.append({"params": min_params, "lr": min_lr})
        lrs.append(min_lr)
    if min_lambd_params:
        dic.append({"params": min_lambd_params, "lr": lambda_lr})
        lrs.append(lambda_lr)
    if opt_type.lower() == 'sgd':
        if momentum:
            opt = torch.optim.SGD(dic, lr=1e-3, momentum=0.9)
        else:
            opt = torch.optim.SGD(dic, lr=1e-3)
    if opt_type.lower() == 'adam':
        opt = torch.optim.Adam(dic, lr=1e-3, betas=(0.9, 0.999))
    return opt, lrs

def convert_to_target(net, norm, verbose=True, res50=False):
    def convert_norm(old_norm, new_norm, num_features, idx):
        norm_layer = new_norm(num_features, idx=idx).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    converted_bns = {}
    for i, layer in enumerate(layers):
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx)
            converted_bns['L0-BN0-0'] = net.bn1
            idx += 1
        else:
            for j, bottleneck in enumerate(layer):
                bottleneck.bn1 = convert_norm(bottleneck.bn1, norm, bottleneck.bn1.num_features, idx)
                converted_bns['L{}-BN{}-{}'.format(i, j, 0)] = bottleneck.bn1
                idx += 1
                bottleneck.bn2 = convert_norm(bottleneck.bn2, norm, bottleneck.bn2.num_features, idx)
                converted_bns['L{}-BN{}-{}'.format(i, j, 1)] = bottleneck.bn2
                idx += 1
                if res50:
                    bottleneck.bn3 = convert_norm(bottleneck.bn3, norm, bottleneck.bn3.num_features, idx)
                    converted_bns['L{}-BN{}-{}'.format(i, j, 3)] = bottleneck.bn3
                    idx += 1
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = convert_norm(bottleneck.downsample[1], norm, bottleneck.downsample[1].num_features, idx)
                    converted_bns['L{}-BN{}-{}'.format(i, j, 2)] = bottleneck.downsample[1]
                    idx += 1
    return net, converted_bns
