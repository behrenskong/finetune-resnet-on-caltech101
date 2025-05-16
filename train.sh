#!/bin/bash

# # distributed training
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     train.py \
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3 \
#     --distributed


# # multi gpu training
# python train.py \
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3



# single gpu training
cd /SSD_DISK/users/rongyi/projects/finetune-resnet-on-caltech101/ &&
conda activate resnet_finetune &&
python train.py \
    --config configs/resnet101_finetune.yaml \
    --experiment_name resnet101_finetune \
    --gpu 7

python train.py \
    --config configs/resnet18_scratch.yaml \
    --experiment_name resnet18_scratch \
    --gpu 3