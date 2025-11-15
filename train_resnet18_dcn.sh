CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20000 train.py --dist \
        --dataset coco \
        --arch resdcn_18 \
        --batch_size 128 \
        --data_dir /home/yanlb/work_space/dataset/coco2017/ \
        --num_workers 4 \
        --optim sgd \
        --use_amp