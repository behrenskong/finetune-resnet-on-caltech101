<<<<<<< HEAD
python test.py \
=======
python src/test.py \
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
    --config configs/resnet34_finetune.yaml \
    --model_path outputs/checkpoints/resnet34_finetune/checkpoint_best.pth \
    --experiment_name resnet34_finetune \
    --gpu 0 \
    --output_dir outputs/test_reports
