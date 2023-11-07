from framework.loss_and_acc import *
from framework.meta_util import split_image_and_label, put_parameters, get_parameters, compare_two_dicts
from framework.registry import EvalFuncs, TrainFuncs
from models.AdaptorHelper import get_new_optimizers
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
    inner_opt_conv = get_new_optimizers(meta_model, lr=args.meta_lr, lambd_lr=args.meta_lambd_lr, names=['bn'], momentum=args.meta_second_order)
    #inner_opt_conv = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    globalstep = len(train_data) * epoch
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_mixup:
            split_data = mixup_op(split_data)

        if args.domain_bn_shift:
            meta_model.StochasticBNShift(args.domain_bn_shift_p)

        #optimizers.zero_grad()
        for data in split_data:
            optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_conv, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                if args.bn_momentum:
                    fnet.set_momentum(0)
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
    inner_opt = get_new_optimizers(meta_model, lr=args.meta_lr, lambd_lr=args.meta_lambd_lr, names=['bn'], momentum=args.meta_second_order)
    #inner_opt = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    print(f'Inner optimizer: {type(inner_opt).__name__}')
    for data in eval_data:
        data = to(data, device)

        with torch.no_grad():  # Normal Test
            get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fnet, diffopt):
            if args.bn_momentum:
                fnet.set_momentum(0)
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
    inner_opt_max = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias'], momentum=args.meta_second_order)
    if args.no_inner_lambda:
        inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    else:
        inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, lambd_lr=args.meta_lambd_lr, names=['bn'], momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')
    globalstep = len(train_data) * epoch
    #randaug_op = TestTimeAug(args)
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_bn_shift:
            meta_model.StochasticBNShift(args.domain_bn_shift_p)

        if args.domain_mixup:
            split_data = mixup_op(split_data)

        #optimizers.zero_grad()
        for data in split_data:
            with torch.no_grad():
                res = meta_model(**data, train_mode='ft', step=0)
            if args.with_max and res['doMax']:
                optimizers.zero_grad()
                with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                    if args.bn_momentum:
                        fnet.set_momentum(0)
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opt_max.step(sup_loss-unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                    losses[0].backward()
                optimizers.step()
            optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                if args.momentum:
                    fnet.set_momentum(0)
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_min_')
                    opt_min.step(sup_loss + unsup_loss)
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min_')
                losses[0].backward()
            optimizers.step()
            #loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            #acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            #print(loss_log + '\n' + acc_log + '\n')
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@TrainFuncs.register('tta_meta_sup1')
def tta_meta_minimax1(meta_model, train_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    inner_opt_max = get_new_optimizers(meta_model, lr=args.meta_max_lr, names=['bn'], param_names=['bias'], momentum=args.meta_second_order)
    if args.no_inner_lambda:
        inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    else:
        inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], lambd_lr=args.meta_lambd_lr, momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')
    if args.domain_mixup:
        mixup_op = MixUp(meta_model.num_classes)
    globalstep = len(train_data) * epoch
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        if args.domain_bn_shift:
            meta_model.StochasticBNShift(args.domain_bn_shift_p)

        if args.domain_mixup:
            split_data = mixup_op(split_data)

        #optimizers.zero_grad()
        for data in split_data:
            with torch.no_grad():
                res = meta_model(**data, train_mode='ft', step=0)

            if args.with_max and res['doMax']:
                optimizers.zero_grad()
                with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                    if args.bn_momentum:
                        fnet.set_momentum(0)
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                        opt_max.step(sup_loss-unsup_loss)
                    losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
                    losses[0].backward()
            with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                if args.momentum:
                    fnet.set_momentum(0)
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_min_')
                    opt_min.step(sup_loss + unsup_loss)
                losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min_')
                losses[0].backward()
            optimizers.step()
            #loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            #acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            #print(loss_log + '\n' + acc_log)
        #optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@EvalFuncs.register('tta_meta_sup')
def tta_meta_minimax_test(meta_model, eval_data, lr, epoch, args, engine, mode):
    #import higher
    device = engine.device
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    if args.domain_bn_shift:
        meta_model.reset_shift_bn()
    fast_model = copy.deepcopy(meta_model)
    meta_model.eval()
    fast_model.eval()
    inner_opt_max = get_new_optimizers(meta_model, lr=args.meta_max_lr, names=['bn'], param_names=['bias'], momentum=args.meta_second_order)
    if args.no_inner_lambda:
        inner_opt_min = get_new_optimizers(fast_model, lr=args.meta_lr, names=['bn'], param_names=['bias', 'weight'], momentum=args.meta_second_order)
    else:
        inner_opt_min = get_new_optimizers(fast_model, lr=args.meta_lr, names=['bn'], lambd_lr=args.meta_lambd_lr, momentum=args.meta_second_order)
    step = 0
    for data in eval_data:
        data = to(data, device)

        # Normal Test
        with torch.no_grad():
            _, o = get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')


        if args.with_max:
            with torch.no_grad():
                res = meta_model(**data, train_mode='ft', step=0)
            if res['doMax']:
                with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=False) as (fnet, opt_max):
                    fnet.train()
                    if args.bn_momentum:
                        fnet.set_momentum(0)
                    for _ in range(args.meta_step):
                        unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                                running_corrects, prefix=f'spt_max_')
                        opt_max.step(sup_loss - unsup_loss)

                    with torch.no_grad():
                        params, states = get_parameters(fnet)
                        fast_model = put_parameters(fast_model, params, states)

        with higher.innerloop_ctx(fast_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=False) as (fnet, opt_min):
            fnet.train()
            if args.bn_momentum:
                fnet.set_momentum(0)
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                    running_corrects, prefix=f'spt_min_')
                opt_min.step(sup_loss + unsup_loss)
            get_loss_and_acc(fnet(**data, train_mode='test'), running_loss, running_corrects)

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
