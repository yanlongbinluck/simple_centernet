from utils.post_process import ctdet_decode
from utils.losses import _neg_loss, _reg_loss
from utils.image import transform_preds
from utils.utils import _tranpose_and_gather_feature, load_model
from nets.resdcn import get_pose_net
from nets.hourglass import get_hourglass
from datasets.pascal import PascalVOC, PascalVOC_eval
from datasets.coco import COCO, COCO_eval
import torch.distributed as dist
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
import sys
import time
import argparse
from torch.cuda import amp
from torch.nn.utils import clip_grad
import random


# Training settings
parser = argparse.ArgumentParser(description='simple_centernet')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')
parser.add_argument('--root_dir', type=str, default='./train_logs')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--pretrain_name', type=str, default='pretrain')
parser.add_argument('--dataset',type=str,default='coco',choices=['coco','pascal'])
parser.add_argument('--arch', type=str, default='large_hourglass')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)
parser.add_argument('--optim',type=str, default=None,choices=['sgd','adam'])
parser.add_argument('--lr_step', type=list, default=[90,120]) # default [90,120]
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=140)
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument("--use_amp", action="store_true", help="pytorch amp training")
parser.add_argument("--resume", type=str, default="") # resume from checkpoint.pth
cfg = parser.parse_args()


def save_checkpoint(current_iter, model, optimizer, epoch, checkpoint_path):
    print("Saving checkpoint...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_iter': current_iter,
    }, checkpoint_path)

def resume_from_checkpoint(model,optimizer,checkpoint_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path,map_location=f'cuda:{cfg.local_rank}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    current_iter = checkpoint['current_iter']
    return start_epoch, current_iter

def mkdir(path):
    path=path.strip() 
    path=path.rstrip("\\") 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' has been created')

def get_lr(optimizer):
    lr_group=[]
    for param_group in optimizer.param_groups:
        lr_group += [param_group['lr']]
    return lr_group

def adjust_epoch_lr(epoch,lr_step,optimizer):
    if epoch in lr_step:
        cfg.lr = cfg.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr

def warmup_iter_lr(optimizer,warmup_steps,step):
    if step < warmup_steps:
        warmup_factor = step / warmup_steps
        current_lr = cfg.lr * warmup_factor
    else:
        current_lr = cfg.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(epoch,scaler,model,train_loader,optimizer):
    global current_iter
    model.train()

    adjust_epoch_lr(epoch,cfg.lr_step,optimizer)
    for batch_idx, batch in enumerate(train_loader):
        warmup_iter_lr(optimizer,500,current_iter)
        current_base_lr = get_lr(optimizer)[0]
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=cfg.device, non_blocking=True)
        # update
        optimizer.zero_grad()
        if cfg.use_amp == True:
            with amp.autocast(enabled=True):
                outputs = model(batch['image'])
                hmap, regs, w_h_ = zip(*outputs)
                regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]
                hmap_loss = _neg_loss(hmap, batch['hmap'])
                reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
                w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # this is used for grad clip in normal mode
            #clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch['image'])
            hmap, regs, w_h_ = zip(*outputs)
            regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]
            hmap_loss = _neg_loss(hmap, batch['hmap'])
            reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
            loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss 
            loss.backward()
            #clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()

        if (batch_idx % cfg.log_interval == 0) and cfg.local_rank == 0:
            write_str = '[epoch:%d/%d,iter:%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +\
                    ' hmap_loss=%.5f reg_loss=%.5f w_h_loss=%.5f' % (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +\
                    ' current_lr=%.7f' % current_base_lr
            print(write_str)

            with open(log_filename,"a") as f:
                f.writelines(write_str+"\n")

        current_iter = current_iter + 1
    return

def val_map(epoch,model,val_loader,val_dataset):
    global best_mAP
    model.eval()
    torch.cuda.empty_cache()
    max_per_image = 100

    results = {}
    with torch.no_grad():
        for inputs in val_loader:
            img_id, inputs = inputs[0]
            detections = []
            for scale in inputs:
                inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                output = model(inputs[scale]['image'])[-1]
                dets = ctdet_decode(*output, K=cfg.test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
                top_preds = {}
                dets[:,:2] = transform_preds(dets[:,0:2],
                                            inputs[scale]['center'],
                                            inputs[scale]['scale'],
                                            (inputs[scale]['fmap_w'],
                                            inputs[scale]['fmap_h']))
                dets[:,2:4] = transform_preds(dets[:,2:4],
                                            inputs[scale]['center'],
                                            inputs[scale]['scale'],
                                            (inputs[scale]['fmap_w'],
                                                inputs[scale]['fmap_h']))
                clses = dets[:, -1]
                for j in range(val_dataset.num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                                for j in range(1, val_dataset.num_classes + 1)}
            scores = np.hstack([bbox_and_scores[j][:, 4]
                                for j in range(1, val_dataset.num_classes + 1)])
            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, val_dataset.num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            results[img_id] = bbox_and_scores

    eval_results = val_dataset.run_eval(results, save_dir=log_root)
    print(eval_results)
    current_mAP = eval_results[0]
    if current_mAP > best_mAP:
        best_mAP = current_mAP
    write_str = "{} epoch mAP:{},best mAP:{}".format(epoch,current_mAP,best_mAP)
    print(write_str)
    with open(log_filename,"a") as f:
        f.writelines(write_str+"\n")


def main():
    setup_seed(666+cfg.local_rank) # avoid same seed in different rank
    # disable this if OOM at beginning of training
    torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    print('Setting up data...')
    Dataset = COCO if cfg.dataset == 'coco' else PascalVOC
    train_dataset = Dataset(cfg.data_dir,'train',split_ratio=cfg.split_ratio,img_size=cfg.img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_gpus, rank=cfg.local_rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // num_gpus if cfg.dist else cfg.batch_size,
        shuffle=not cfg.dist,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler if cfg.dist else None)

    Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval
    val_dataset = Dataset_eval(cfg.data_dir,'val',test_scales=[1.],test_flip=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn)

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)
    else:
        raise NotImplementedError


    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank, ], output_device=cfg.local_rank)
    else:
        model = nn.DataParallel(model).to(cfg.device)


    start_epoch = 1
    if cfg.optim == "adam":
        cfg.lr = 0.0005 * (cfg.batch_size / 128)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if cfg.optim == "sgd":
        cfg.lr = 0.02 * (cfg.batch_size / 128)
        optimizer = torch.optim.SGD(params = model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0001)
    
    # resume
    global current_iter
    if os.path.isfile(cfg.resume):
        start_epoch, current_iter = resume_from_checkpoint(model,optimizer,cfg.resume)

    if cfg.use_amp == True:
        scaler = amp.GradScaler(enabled=True)
    else:
        scaler = None

    print('Starting training...')
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        train(epoch,scaler,model,train_loader,optimizer)
        end_time = time.time()
        one_epoch_time = end_time - start_time
        remain_time = one_epoch_time * (cfg.num_epochs - epoch)
        time_str = 'training time of 1 epoch:{:.2f} mins,remain time of training:{:.2f} hours'.format(one_epoch_time/60,remain_time/3600)
        print(time_str)
        with open(log_filename,"a") as f:
            f.writelines(time_str+"\n")

        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val_map(epoch,model,val_loader,val_dataset)
        torch.save(model.module.state_dict(), weight_filename)
        save_checkpoint(current_iter, model, optimizer, epoch, checkpoint_filename)

if __name__ == '__main__':
    log_root = cfg.root_dir + "/{}/{}".format(cfg.arch,time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    mkdir(log_root)
    current_iter = 0
    best_mAP = 0
    log_filename = log_root +"/train_log.txt"
    weight_filename = log_root +"/latest.pth"
    checkpoint_filename = log_root +"/checkpoint.pth"
    main()
