from framework.loss_and_acc import *
from framework.meta_util import split_image_and_label, put_parameters, get_parameters
from framework.registry import EvalFuncs, TrainFuncs
from models.AdaptorHelper import get_new_optimizers, get_mme_optimizers
from utils.tensor_utils import to, AverageMeterDict
from dataloader.augmentations import MixUp
import higher
import copy
"""
ARM
"""
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
    inner_opt = get_mme_optimizers(meta_model, names=['bn'], max_L=args.max_bn_layer, min_lr=args.meta_lr,
                                          max_lr=args.meta_lambd_lr, max_param=['bias'], min_param=args.inner,
                                          momentum=args.meta_second_order)

    print(f'Meta LR_min : {args.meta_lr}, Meta LR max : {args.meta_lambd_lr}')
    globalstep = len(train_data) * epoch
    #randaug_op = TestTimeAug(args)
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_mixup:
            split_data = mixup_op(split_data)
        #optimizers.zero_grad()
        if args.with_max:
            optimizers.zero_grad()
            for data in split_data:
                if args.domain_bn_shift:
                    meta_model.StochasticBNShift(args.domain_bn_shift_p)
                with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                    for param in opt_max.param_groups[0]["params"]:
                        param.requires_grad = False
                    for param in opt_max.param_groups[1]["params"]:
                        param.requires_grad = True
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opt_max.step(sup_loss-unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                    losses[0].backward()
            optimizers.step()
        optimizers.zero_grad()
        for data in split_data:
            if args.domain_bn_shift:
                meta_model.StochasticBNShift(args.domain_bn_shift_p)
            with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                for param in opt_min.param_groups[1]["params"]:
                    param.requires_grad = False
                for param in opt_min.param_groups[0]["params"]:
                    param.requires_grad = True
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_ ), running_loss, running_corrects, prefix=f'spt_min_')
                    opt_min.step(sup_loss + unsup_loss)
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min_')
                losses[0].backward()
            optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@TrainFuncs.register('tta_meta_sup1')
def tta_meta_minimax1(meta_model, train_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    min_opt, max_opt = get_mme_optimizers(meta_model, names=['bn'], max_ls=args.max_bn_layer, min_lr=args.meta_lr,
                                          max_lr=args.meta_lambd_lr, max_param=['bias'], min_param=args.inner,
                                          momentum=args.meta_second_order)
    inner_opts = []
    if min_opt: inner_opts.append(min_opt)
    if max_opt: inner_opts.append(max_opt)

    print(f'Meta LR_min : {args.meta_lr}, Meta LR max : {args.meta_lambd_lr}')
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    globalstep = len(train_data) * epoch
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_mixup:
            split_data = mixup_op(split_data)

        optimizers.zero_grad()
        for data in split_data:
            if args.domain_bn_shift:
                meta_model.StochasticBNShift(args.domain_bn_shift_p)
            with higher.innerloop_ctx(meta_model, inner_opts, copy_initial_weights=False, track_higher_grads=True) as (fnet, opts):
                for _ in range(args.meta_step):
                    if args.with_max:
                        for param in opts.param_groups[1]["params"]:
                            param.requires_grad = True
                        for param in opts.param_groups[0]["params"]:
                            param.requires_grad = False
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opts.step(sup_loss-unsup_loss)
                    for param in opts.param_groups[1]["params"]:
                        param.requires_grad = False
                    for param in opts.param_groups[0]["params"]:
                        param.requires_grad = True
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_min_')
                    opts.step(sup_loss + unsup_loss)
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                losses[0].backward()
        optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@EvalFuncs.register('tta_meta_sup')
def tta_meta_minimax_test(meta_model, eval_data, lr, epoch, args, engine, mode):
    #import higher
    device = engine.device
    logger = engine.logger
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.reset_shift_bn()
    meta_model.eval()
    min_opt, max_opt = get_mme_optimizers(meta_model, names=['bn'], max_L=args.max_bn_layer, min_lr=args.meta_lr,
                                          max_lr=args.meta_lambd_lr, max_param=['bias'], min_param=args.inner,
                                          momentum=args.meta_second_order)
    inner_opts = []
    if min_opt:
        inner_opts.append(min_opt)
    if max_opt:
        inner_opts.append(max_opt)
    step = 0
    embd_org, embd_label, embd_max, embd_mme=[], [], [], []
    for data in eval_data:
        data = to(data, device)

        # Normal Test
        with torch.no_grad():
            ret = meta_model.step(**data, train_mode='test')
            _, o = get_loss_and_acc(ret, running_loss, running_corrects, prefix='original_')
            if step %100==0:
                embd_org.append(ret['vis']['feats'])
                embd_label.append([data['label'][0].cpu().numpy().tolist() for v in data['label']])

        with higher.innerloop_ctx(meta_model, inner_opts, track_higher_grads=False) as (fnet, opts):
            fnet.train()
            if args.with_max:
                for _ in range(args.meta_step):
                    ret = fnet(**data, train_mode='ft', step=_)
                    unsup_loss, sup_loss = get_loss_and_acc(ret, running_loss, running_corrects, prefix=f'spt_max_')
                    opts[1].step(sup_loss - unsup_loss)
                if step%100==0:
                    embd_max.append(ret['vis']['feats'])
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opts[0].step(sup_loss + unsup_loss)
            ret = fnet(**data, train_mode='test')
            if step%100==0:
                embd_mme.append(ret['vis']['feats'])
            get_loss_and_acc(ret, running_loss, running_corrects)
        step += 1
        if step % 100 == 0:
            loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    logger.writer.add_embedding(torch.stack(embd_org), metadata= embd_label, tag='epoch/{}/org'.format(mode))
    if args.with_max:
        logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/max'.format(mode))
    logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/mme'.format(mode))
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)


@EvalFuncs.register('tta_meta_sup1')
def tta_meta_minimax_test1(meta_model, eval_data, lr, epoch, args, engine, mode):
    #import higher
    device = engine.device
    logger = engine.logger
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.reset_shift_bn()
    meta_model.eval()
    min_opt, max_opt = get_mme_optimizers(meta_model, names=['bn'], max_L=args.max_bn_layer, min_lr=args.meta_lr,
                                          max_lr=args.meta_lambd_lr, max_param=['bias'], min_param=args.inner,
                                          momentum=args.meta_second_order)
    inner_opts = []
    if min_opt: inner_opts.append(min_opt)
    if max_opt: inner_opts.append(max_opt)
    step = 0
    embd_org, embd_label, embd_max, embd_mme = [], [], []
    for data in eval_data:
        data = to(data, device)

        # Normal Test
        with torch.no_grad():
            ret = meta_model.step(**data, train_mode='test')
            _, o = get_loss_and_acc(ret, running_loss, running_corrects, prefix='original_')
            if step %100:
                embd_org.append(ret['vis']['feats'])
                embd_label.append([v+str(i) for i, v in enumerate(data[1])])

        with higher.innerloop_ctx(meta_model, inner_opts, track_higher_grads=False) as (fnet, opts):
            fnet.train()
            for _ in range(args.meta_step):
                if args.with_max:
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                            running_corrects, prefix=f'spt_max_')
                    opts[1].step(sup_loss - unsup_loss)
                ret = fnet(**data, train_mode='ft', step=_)
                unsup_loss, sup_loss = get_loss_and_acc(ret, running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opts[0].step(sup_loss + unsup_loss)
            res = fnet(**data, train_mode='test')
            get_loss_and_acc(res, running_loss, running_corrects)
            if step%100:
                embd_max.append(ret['vis']['feats'])
                embd_mme.append(res['vis']['feats'])
        step += 1
        if step % 100 == 0:
            loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/org'.format(mode))
    if args.with_max:
        logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/max'.format(mode))
    logger.writer.add_embedding(torch.stack(embd_org), metadata=embd_label, tag='epoch/{}/mme'.format(mode))

    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)