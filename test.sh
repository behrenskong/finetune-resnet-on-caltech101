python test.py \
    --config configs/resnet101_finetune.yaml \
    --model_path outputs/checkpoints/resnet101_finetune/checkpoint_best.pth \
    --experiment_name resnet101_finetune \
    --gpu 0 \
    --output_dir outputs/test_reports

python test.py \
    --config configs/resnet101_scratch.yaml \
    --model_path outputs/checkpoints/resnet101_scratch/checkpoint_best.pth \
    --experiment_name resnet101_scratch \
    --gpu 0 \
    --output_dir outputs/test_reports
