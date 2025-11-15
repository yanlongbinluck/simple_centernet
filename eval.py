from utils.post_process import ctdet_decode
from utils.image import transform_preds
from nets.resdcn import get_pose_net
from nets.hourglass import get_hourglass
from datasets.pascal import PascalVOC, PascalVOC_eval
from datasets.coco import COCO_eval
import torch.utils.data
import numpy as np
import argparse
from tqdm import tqdm


def val_map(model,val_loader,val_dataset):
    model.eval()
    torch.cuda.empty_cache()
    max_per_image = 100

    results = {}
    with torch.no_grad():
        for inputs in tqdm(val_loader):
            img_id, inputs = inputs[0]
            detections = []
            for scale in inputs:
                inputs[scale]['image'] = inputs[scale]['image'].cuda()
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

    eval_results = val_dataset.run_eval(results, save_dir="./results")
    print(eval_results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple_centernet')
    parser.add_argument('--weight_dir', type=str, default="./train_logs/resdcn_18/2025-10-02-09-04-07/latest.pth")
    parser.add_argument('--data_dir', type=str, default="/home/yanlb/work_space/dataset/coco2017/")
    parser.add_argument('--pretrain_name', type=str, default='pretrain')
    parser.add_argument('--dataset',type=str,default='coco',choices=['coco','pascal'])
    parser.add_argument('--arch', type=str, default='resdcn_18')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--test_topk', type=int, default=100)
    cfg = parser.parse_args()

    val_dataset = COCO_eval(cfg.data_dir,'val',test_scales=[1.],test_flip=False, fix_size=True) # fix_size=True AP is lower than fix_size=False
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
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=80)
    else:
        raise NotImplementedError
    
    model.load_state_dict(torch.load(cfg.weight_dir), strict = True)
    model.cuda()
    val_map(model,val_loader,val_dataset)
