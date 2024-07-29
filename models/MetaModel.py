from framework.loss_and_acc import *
from framework.meta_util import split_image_and_label, put_parameters, get_parameters
from framework.registry import EvalFuncs, TrainFuncs
from models.AdaptorHelper import get_new_optimizers, get_mme_optimizers
from utils.tensor_utils import to, AverageMeterDict
from dataloader.augmentations import MixUp
import higher
import pandas as pd
from utils.t_sne import plot_tsne
from sklearn.manifold import TSNE
import numpy as np
"""
ARM
"""
sub_test = 200
@TrainFuncs.register('tta_meta')
def tta_meta_train(meta_model, train_data, lr, epoch, args, engine, mode):
    #import higher
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    inner_opt_conv, _ = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr,
                                        lambd_lr=args.meta_lambd_lr, names=['bn'], momentum=args.meta_second_order)
    #inner_opt_conv = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')

    globalstep = len(train_data) * epoch
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        #optimizers.zero_grad()
        for data in split_data:
            optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_conv, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                for _ in range(args.meta_step):
                    unsup_loss, _ = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_')
                    diffopt.step(unsup_loss)
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_')
                losses[0].backward()
            optimizers.step()
            #loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            #acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            #print(loss_log + '\n' + acc_log)
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@EvalFuncs.register('tta_meta')
def tta_meta_test(meta_model, eval_data, lr, epoch, args, engine, mode):
    #import higher
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    meta_model.eval()
    inner_opt, _ = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, lambd_lr=args.meta_lambd_lr, names=['bn'], momentum=args.meta_second_order)
    #inner_opt = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    print(f'Inner optimizer: {type(inner_opt).__name__}')
    for data in eval_data:
        data = to(data, device)

        with torch.no_grad():  # Normal Test
            get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss, _ = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_')
                diffopt.step(unsup_loss)
            get_loss_and_acc(fnet(**data, train_mode='test'), running_loss, running_corrects)
        #loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
        #acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
        #print(loss_log+'\n'+acc_log+'\n')
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)

@TrainFuncs.register('tta_meta_sup')
def tta_meta_minimax(meta_model, train_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    inner_opt_max, params_max = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_max_lr, names=['bn'], layers=args.max_bn_layer,
                                       param_names=['bias'], momentum=args.meta_second_order)
    inner_opt_min, params_min = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, names=['bn'],  nonpara_name=params_max,
                                       lambd_lr=args.meta_lambd_lr, param_names=args.inner, momentum=args.meta_second_order)
    str = f'Meta LR : {args.meta_lr} '
    if 'lambd' in args.inner:
        str = str + f'Meta-train Lambd LR: {args.meta_lambd_lr} '
    if args.with_max:
        str = str + f'Meta-train Max LR: {args.meta_max_lr}'
    print(str)
    ndata = len(train_data)
    globalstep = ndata * epoch

    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)

    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)
        if args.domain_mixup:
            split_data = mixup_op(split_data)
        for _ in range(args.meta_step):
            if args.with_max:
                optimizers.zero_grad()
                for data in split_data:
                    if args.domain_bn_shift:
                        meta_model.StochasticBNShift(args.domain_bn_shift_p)
                    with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opt_max.step(sup_loss-unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                    losses[0].backward()
                optimizers.step()
            optimizers.zero_grad()
            for data in split_data:
                if args.domain_bn_shift:
                    meta_model.StochasticBNShift(args.domain_bn_shift_p)
                with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_ ), running_loss, running_corrects, prefix=f'spt_min_')
                        opt_min.step(sup_loss + unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min_')
                    losses[0].backward()
            optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        if globalstep % sub_test == 0 and ndata>3*sub_test:
            acc_, (loss_dict, acc_dict) = EvalFuncs[engine.args.eval](meta_model, engine.target_test, lr, globalstep/sub_test ,
                                                                    engine.args, engine, mode='test_sub', maxiter=sub_test)
            engine.logger.tf_log_file_step
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@TrainFuncs.register('tta_meta1')
def tta_meta1(meta_model, train_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    inner_opt_min, params_min = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, names=['bn'],
                                       lambd_lr=args.meta_lambd_lr, param_names=args.inner, momentum=args.meta_second_order)
    str = f'Meta LR : {args.meta_lr} '
    if 'lambd' in args.inner:
        str = str + f'Meta-train Lambd LR: {args.meta_lambd_lr} '
    if args.with_max:
        str = str + f'Meta-train Max LR: {args.meta_max_lr}'
    print(str)
    ndata = len(train_data)
    globalstep = ndata * epoch

    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)

    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)
        if args.domain_mixup:
            split_data = mixup_op(split_data)
        for _ in range(args.meta_step):
            optimizers.zero_grad()
            for data in split_data:
                if args.domain_bn_shift:
                    meta_model.StochasticBNShift(args.domain_bn_shift_p)
                with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_ ), running_loss, running_corrects, prefix=f'spt_min_')
                        opt_min.step(sup_loss + unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min_')
                    losses[0].backward()
            optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        if globalstep % sub_test == 0 and ndata>3*sub_test:
            acc_, (loss_dict, acc_dict) = EvalFuncs[engine.args.eval](meta_model, engine.target_test, lr, globalstep/sub_test ,
                                                                    engine.args, engine, mode='test_sub', maxiter=sub_test)
            engine.logger.tf_log_file_step
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@TrainFuncs.register('tta_meta_sup2')
def tta_meta_minimax2(meta_model, train_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    mme_opt, lrs = get_mme_optimizers(meta_model, names=['bn'], max_L=args.max_bn_layer,
                                      min_lr=args.meta_lr, lambda_lr=args.meta_lambd_lr, max_lr=args.meta_max_lr,
                                      max_param=['bias'], min_param=args.inner, momentum=args.meta_second_order)
    # inner_opt_max, params_max = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_max_lr, names=['bn'], layers=args.max_bn_layer,  #multiple layers
    #                                    param_names=['bias'], momentum=args.meta_second_order)  # weights and biases
    # inner_opt_min, params_min = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, names=['bn'],  nonpara_name=params_max,
    #                                    lambd_lr=args.meta_lambd_lr, param_names=args.inner, momentum=args.meta_second_order)
    str = f'Meta LR : {args.meta_lr} '
    if 'lambd' in args.inner:
        str = str + f'Meta-train Lambd LR: {args.meta_lambd_lr} '
    if args.with_max:
        str = str + f'Meta-train Max LR: {args.meta_max_lr}'
    print(str)
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    ndata = len(train_data)
    globalstep = ndata * epoch
    max_lrs = lrs.copy()
    min_lrs = lrs.copy()
    max_lrs[1:] = [0]*(len(lrs)-1)
    min_lrs[0] =0
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_mixup:
            split_data = mixup_op(split_data)

        optimizers.zero_grad()
        for data in split_data:
            if args.domain_bn_shift:
                meta_model.StochasticBNShift(args.domain_bn_shift_p)
            with higher.innerloop_ctx(meta_model, mme_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt):
                for _ in range(args.meta_step):
                    if args.with_max:
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opt.step(sup_loss-unsup_loss, override={'lr': max_lrs})
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_min_')
                    opt.step(sup_loss + unsup_loss, override={'lr': min_lrs})
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                losses[0].backward()
        optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        if globalstep %sub_test  == 0 and ndata>sub_test*3:
            acc_, (loss_dict, acc_dict) = EvalFuncs[engine.args.eval](meta_model, engine.target_test, lr, globalstep/sub_test ,
                                                                    engine.args, engine, mode='test_sub', maxiter=sub_test )
            engine.logger.tf_log_file_step('test_sub', globalstep/sub_test, loss_dict, acc_dict)
            meta_model.train()
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@EvalFuncs.register('tta_meta_sup')
def tta_meta_minimax_test(meta_model, eval_data, lr, epoch, args, engine, mode, maxiter=np.inf):
    #import higher
    device = engine.device
    logger = engine.logger
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.reset_shift_bn()
    meta_model.eval()
    mme_opt, lrs = get_mme_optimizers(meta_model, names=['bn'], max_L=args.max_bn_layer,
                                      min_lr=args.meta_lr, lambda_lr=args.meta_lambd_lr, max_lr=args.meta_max_lr,
                                      max_param=['bias'], min_param=args.inner, momentum=args.meta_second_order)
    max_lrs = lrs.copy()
    min_lrs = lrs.copy()
    max_lrs[1:] = [0]*(len(lrs)-1)
    min_lrs[0] =0
    step = 0
    embd_org, embd_label, embd_max, embd_mme=[], [], [], []
    s = round(len(eval_data)/20+1)

    for data in eval_data:
        if  step > maxiter:
            break
        data = to(data, device)
        # Normal Test
        with torch.no_grad():
            ret = meta_model.step(**data, train_mode='test')
            _, o = get_loss_and_acc(ret, running_loss, running_corrects, prefix='original_')
            if step % s==0:
                embd_org.append(ret['vis']['feats'])
                embd_label.append(data['label'])

        with higher.innerloop_ctx(meta_model, mme_opt, track_higher_grads=False) as (fnet, opt):
            fnet.train()
            if args.with_max:
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                                running_corrects, prefix=f'spt_max_')
                    opt.step(sup_loss - unsup_loss, override={'lr': max_lrs})
                if step % s == 0:
                    with torch.no_grad():
                        ret = fnet.step(**data, train_mode='test')
                        embd_max.append(ret['vis']['feats'])
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opt.step(sup_loss + unsup_loss, override={'lr': min_lrs})
            with torch.no_grad():
                fnet.eval()
                ret = fnet(**data, train_mode='test')
                get_loss_and_acc(ret, running_loss, running_corrects)
            if step % s==0:
                embd_mme.append(ret['vis']['feats'])
        step += 1
        if step % 100 == 0:
            loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    # logger.writer.add_embedding(torch.cat(embd_org), metadata=embd_label, tag='epoch/{}/org'.format(mode))
    # if args.with_max:
    #     logger.writer.add_embedding(torch.cat(embd_max), metadata=embd_label, tag='epoch/{}/max'.format(mode))
    # logger.writer.add_embedding(torch.cat(embd_mme), metadata=embd_label, tag='epoch/{}/mme'.format(mode))
    embd_label = torch.cat(embd_label)
    # logger.writer.add_figure('epoch/{}/org'.format(mode),
    #                          draw_tsne(torch.cat(embd_org), embd_label, engine.classes, epoch, mode, 'org_{}'.format(mode)), epoch)
    # if args.with_max:
    #     logger.writer.add_figure('epoch/{}/max'.format(mode),
    #                          draw_tsne(torch.cat(embd_max), embd_label, engine.classes, epoch, mode, 'max_{}'.format(mode)), epoch)
    # logger.writer.add_figure('epoch/{}/mme'.format(mode),
    #                          draw_tsne(torch.cat(embd_mme), embd_label, engine.classes, epoch, mode, 'mme_{}'.format(mode)), epoch)

    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)

def draw_tsne(feats, y, classname, epoch, mode, tag, log_dir=None):
    df = pd.DataFrame()
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(feats.detach().cpu().numpy())
    df["Class"] = y.detach().cpu().numpy()
    df["tsne-axis1"] = z[:, 0]
    df["tsne-axis2"] = z[:, 1]
    markers_class = ['o'] * 12
    try:
        return plot_tsne(df, classname, markers_class, epoch, mode, save_path=log_dir,
                  save_prefix=tag + '_')
    except:
        pass

@EvalFuncs.register('tta_meta1')
def tta_meta_test1(meta_model, eval_data, lr, epoch, args, engine, mode, maxiter=np.inf):
    #import higher
    device = engine.device
    logger = engine.logger
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.reset_shift_bn()
    meta_model.eval()
    inner_opt_min, params_min = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, names=['bn'],
                                       lambd_lr=args.meta_lambd_lr, param_names=args.inner, momentum=args.meta_second_order)
    step = 0
    embd_org, embd_label, embd_max, embd_mme=[], [], [], []
    s = round(len(eval_data)/20+1)

    for data in eval_data:
        if  step > maxiter:
            break
        data = to(data, device)
        # Normal Test
        with torch.no_grad():
            ret = meta_model.step(**data, train_mode='test')
            _, o = get_loss_and_acc(ret, running_loss, running_corrects, prefix='original_')
            if step % s==0:
                embd_org.append(ret['vis']['feats'])
                embd_label.append(data['label'])

        with higher.innerloop_ctx(meta_model, inner_opt_min, track_higher_grads=False) as (fnet, opt_min):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opt_min.step(sup_loss + unsup_loss)
            with torch.no_grad():
                fnet.eval()
                ret = fnet(**data, train_mode='test')
                get_loss_and_acc(ret, running_loss, running_corrects)
            if step % s==0:
                embd_mme.append(ret['vis']['feats'])
        step += 1
        if step % 100 == 0:
            loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)

@EvalFuncs.register('tta_meta_sup1')
def tta_meta_minimax_test1(meta_model, eval_data, lr, epoch, args, engine, mode, maxiter=np.inf):
    #import higher
    device = engine.device
    logger = engine.logger
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.reset_shift_bn()
    meta_model.eval()
    inner_opt_max, params_max = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_max_lr, names=['bn'], layers=args.max_bn_layer,  #multiple layers
                                       param_names=['bias'], momentum=args.meta_second_order)  # weights and biases
    inner_opt_min, params_min = get_new_optimizers(meta_model, opt_type=args.inneropt, lr=args.meta_lr, names=['bn'],  nonpara_name=params_max,
                                       lambd_lr=args.meta_lambd_lr, param_names=args.inner, momentum=args.meta_second_order)
    print(f'Meta LR_min : {args.meta_lr}, Meta LR max : {args.meta_lambd_lr}')
    step = 0
    embd_org, embd_label, embd_mme = [], [], []
    s = round(len(eval_data)/20*args.batch_size/64+1)
    for data in eval_data:
        if  step > maxiter:
            break
        data = to(data, device)

        # Normal Test
        with torch.no_grad():
            ret = meta_model.step(**data, train_mode='test')
            _, o = get_loss_and_acc(ret, running_loss, running_corrects, prefix='original_')
            if step %s==0:
                embd_org.append(ret['vis']['feats'])
                embd_label.append(data['label'])

        with higher.innerloop_ctx(meta_model, inner_opt_max, track_higher_grads=False) as (fnet, opt_max):
            fnet.train()
            for _ in range(args.meta_step):
                if args.with_max:
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                            running_corrects, prefix=f'spt_max_')
                    opt_max.step(sup_loss-unsup_loss)
        with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
            fnet.train()
            for _ in range(args.meta_step):    
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opt_min.step(sup_loss + unsup_loss)
            with torch.no_grad():
                fnet.eval()
                ret = fnet.step(**data, train_mode='test')
                get_loss_and_acc(ret, running_loss, running_corrects)
            if step%s==0:
                embd_mme.append(ret['vis']['feats'])
        step += 1
        if step % 100 == 0:
            loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    #logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/org'.format(mode))
    #if args.with_max:
    #    logger.writer.add_embedding(torch.stack(embd_max), metadata=embd_label, tag='epoch/{}/max'.format(mode))
    #logger.writer.add_embedding(torch.stack(embd_mme), metadata=embd_label, tag='epoch/{}/mme'.format(mode))
    embd_label = torch.cat(embd_label)
    # logger.writer.add_figure('epoch/{}/org'.format(mode),
    #                          draw_tsne(torch.cat(embd_org), embd_label, engine.classes, epoch, mode, 'org_{}'.format(mode)), epoch)
    # logger.writer.add_figure('epoch/{}/mme'.format(mode),
    #                          draw_tsne(torch.cat(embd_mme), embd_label, engine.classes, epoch, mode, 'mme_{}'.format(mode)), epoch)

    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)
