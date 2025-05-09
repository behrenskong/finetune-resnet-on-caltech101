# finetune-resnet-on-caltech101

本项目基于PyTorch框架，实现了在ImageNet上预训练的卷积神经网络模型对Caltech-101数据集的迁移学习微调，并与从零训练模型进行对比。

## 项目结构

```
caltech101_finetune/
├── data/                        # 数据集目录
│   └── caltech-101/             # Caltech-101数据集
│
├── configs/                     # 配置文件目录
│   ├── resnet34_finetune.yaml   # ResNet34微调配置
│   └── resnet34_scratch.yaml    # ResNet34从零训练配置
│
<<<<<<< HEAD
├── models/                      # 模型目录
│   └── model.py                 # 模型定义
│
├── dataloader                   # 数据读取目录
│   └── data.py                  # 数据加载与预处理
│
├── utils                        # 通用工具目录
│   └── logger.py                # TensorBoard日志工具
=======
├── src/                         # 源代码目录
│   ├── model.py                 # 模型定义
│   ├── data.py                  # 数据加载与预处理
│   ├── logger.py                # TensorBoard日志工具
│   ├── train.py                 # 训练主脚本
│   └── test.py                  # 测试脚本
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
│
├── outputs/                     # 输出目录
│   ├── logs/                    # TensorBoard日志
│   ├── checkpoints/             # 模型权重
│   └── test_reports/            # 测试报告
│
<<<<<<< HEAD
├── train.py                     # 训练主脚本
├── test.py                      # 测试脚本
├── train.sh                     # 训练启动脚本
├── test.sh                      # 测试启动脚本
├── requirements.txt             # 环境配置
=======
├── train.sh                     # 训练启动脚本
├── test.sh                      # 测试启动脚本
>>>>>>> f7f9787a2692ff13254e0a0624755bacd57832d4
└── README.md                    # 项目说明
```

## 环境要求

todo


## 数据集准备

1. 下载 [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集
2. 解压数据集到 `data/` 目录下
3. 确保数据路径符合以下结构:
   ```
   data/caltech-101/101_ObjectCategories/
   ├── accordion/
   ├── airplanes/
   ├── ...
   └── yin_yang/
   ```

## 模型训练

### 配置文件

模型训练参数通过YAML配置文件定义，位于`configs/`目录：
- `resnet34_finetune.yaml`: 使用预训练权重进行微调的配置
- `resnet34_scratch.yaml`: 从零开始训练模型的配置

配置文件包含以下主要部分:
- `data`: 数据集参数
- `model`: 模型架构参数
- `training`: 训练超参数
- `experiment`: 实验输出路径及保存设置

### 训练Shell脚本
具体见`train.sh`
```bash
bash train.sh
```
- 单GPU

```bash
# single gpu training
python train.py \
    --config configs/resnet34_finetune.yaml \
    --experiment_name resnet34_finetune \
    --gpu 0
```
- 分布式训练(DDP)

```bash
# distributed training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config configs/resnet34_finetune.yaml \
    --experiment_name resnet34_finetune \
    --gpu 0,1,2,3 \
    --distributed

```
- DP(不推荐)
```bash
# multi gpu training
python train.py \
    --config configs/resnet34_finetune.yaml \
    --experiment_name resnet34_finetune \
    --gpu 0,1,2,3
```


- `train.py`命令行参数

可用的命令行参数:
- `--config`: 配置文件路径
- `--experiment_name`: 实验名称，用于输出日志和模型
- `--gpu`: 使用的GPU ID，多个ID用逗号分隔
- `--distributed`: 默认为False
- `--local_rank`: 默认为0

## 模型测试

```bash
bash test.sh
```

```bash
python test.py \
    --config configs/resnet34_finetune.yaml \
    --model_path outputs/checkpoints/resnet34_finetune_distributed/checkpoint_best.pth \
    --experiment_name resnet34_finetune_distributed \
    --gpu 0 \
    --output_dir outputs/test_reports
```

### `test.py`命令行参数

- `--config`: 配置文件路径
- `--model_path`: 模型权重文件路径
- `--experiment_name`: 实验名称 
- `--gpu`: 使用的GPU ID
- `--output_dir`: 测试报告保存目录

## 可视化

训练过程中的损失曲线和准确率可以通过TensorBoard查看:

```bash
tensorboard --logdir outputs/logs --port=6006
```

## 预训练模型参数

本项目的预训练模型已上传至以下链接:

- 微调模型: 
- 从零训练模型: 

## 微调/不微调对比


| 模型 | 训练集准确率 | 验证集准确率 | 测试集准确率 |
|-----|------------|------------|------------|
| ResNet34 (预训练微调) | XX.XX% | XX.XX% | XX.XX% |
| ResNet34 (从零训练) | XX.XX% | XX.XX% | XX.XX% |

实验表明，预训练微调模型相比从零训练模型有显著的性能提升，验证了迁移学习在小数据集上的有效性。