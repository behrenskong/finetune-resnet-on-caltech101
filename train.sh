#!/bin/bash

# # distributed training
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=29500 \
<<<<<<< HEAD
#     train.py \
=======
#     src/train.py \
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3 \
#     --distributed


# # multi gpu training
<<<<<<< HEAD
# python train.py \
=======
# python src/train.py \
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
#     --config configs/resnet34_finetune.yaml \
#     --experiment_name resnet34_finetune \
#     --gpu 0,1,2,3



# single gpu training
<<<<<<< HEAD
python train.py \
=======
python src/train.py \
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
    --config configs/resnet34_finetune.yaml \
    --experiment_name resnet34_finetune \
    --gpu 0