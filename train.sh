#!/bin/bash

# # distributed training
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     src/train.py \
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3 \
#     --distributed


# # multi gpu training
# python src/train.py \
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3



# single gpu training
python src/train.py \
    --config configs/resnet34_finetune.yaml \
    --experiment_name resnet34_finetune \
    --gpu 0