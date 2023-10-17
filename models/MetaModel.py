from framework.loss_and_acc import *
from framework.meta_util import split_image_and_label
from framework.registry import EvalFuncs, TrainFuncs
from models.AdaptorHelper import get_new_optimizers
from utils.tensor_utils import to, AverageMeterDict
from dataloader.augmentations import TestTimeAug
import higher
"""
ARM
"""
@TrainFuncs.register('tta_meta')
def tta_meta_train(meta_model, train_data, lr, epoch, args, engine, mode):
    #import higher
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.train()
    inner_opt_conv = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], momentum=args.meta_second_order)
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
                    unsup_loss, _ = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt{_}_')
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
    inner_opt = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], momentum=args.meta_second_order)
    for data in eval_data:
        data = to(data, device)

        with torch.no_grad():  # Normal Test
            get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss, _ = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt{_}_')
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
    inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['weight'], momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')
    globalstep = len(train_data) * epoch
    #randaug_op = TestTimeAug(args)
    for data_list in train_data:
        #if args.batch_aug:
        #    data_list['x'] = randaug_op.batch_aug(data_list['x'], 1)
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)
        #optimizers.zero_grad()
        for data in split_data:
            optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                    opt_max.step(sup_loss-unsup_loss)
            losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
            losses[0].backward()
            optimizers.step()
            optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
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
    inner_opt_max = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias'], momentum=args.meta_second_order)
    inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['weight'], momentum=args.meta_second_order)
    print(f'Meta LR : {args.meta_lr}')
    globalstep = len(train_data) * epoch
    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)
        optimizers.zero_grad()
        for data in split_data:
            #optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_max):
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_max_')
                    opt_max.step(sup_loss-unsup_loss)
            losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_max_')
            losses[0].backward()
            #optimizers.step()
            #optimizers.zero_grad()
            with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=False, track_higher_grads=True) as (fnet, opt_min):
                for _ in range(args.meta_step):
                    unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'spt_min_')
                    opt_min.step(sup_loss + unsup_loss)
            losses = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, prefix=f'qry_min')
            losses[0].backward()
            #optimizers.step()
            #loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
            #acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
            #print(loss_log + '\n' + acc_log)
        optimizers.step()
        engine.logger.tf_log_file_step(mode, globalstep, running_loss.get_average_dicts(), running_corrects.get_average_dicts())
        globalstep += 1
        #optimizers.step()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

@EvalFuncs.register('tta_meta_sup')
def tta_meta_minimax(meta_model, eval_data, lr, epoch, args, engine, mode):
    #import higher
    device = engine.device
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_model.eval()
    inner_opt_max = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['bias'], momentum=False)
    inner_opt_min = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], param_names=['weight'], momentum=False)
    original_state_dict = meta_model.state_dict()
    step = 0
    for data in eval_data:
        data = to(data, device)
        meta_model.load_state_dict(original_state_dict)
        # Normal Test
        with torch.no_grad():
            get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

        with higher.innerloop_ctx(meta_model, inner_opt_max, copy_initial_weights=True, track_higher_grads=False) as (
        fnet, opt_max):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                        running_corrects, prefix=f'spt_max_')
                opt_max.step(sup_loss - unsup_loss)

        with higher.innerloop_ctx(meta_model, inner_opt_min, copy_initial_weights=True, track_higher_grads=False) as (
        fnet, opt_min):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss, sup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss,
                                                        running_corrects, prefix=f'spt_min_')
                opt_min.step(sup_loss + unsup_loss)

        get_loss_and_acc(meta_model(**data, train_mode='test'), running_loss, running_corrects)
        # step += 1
        # if step % 100 == 0:
        #     loss_log = ' '.join([f'loss[{k}] {v}\t' for k, v in running_loss.get_average_dicts().items()])
        #     acc_log = ' '.join([f'acc[{k}] {v}\t' for k, v in running_corrects.get_average_dicts().items()])
        #     print(loss_log + '\n' + acc_log)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)