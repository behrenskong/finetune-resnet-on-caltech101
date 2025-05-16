import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import time
import warnings
from datetime import datetime
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from models.model import get_model
from dataloader.data import create_dataloaders
from utils.logger import TensorboardLogger

class AverageMeter:
    """accumulative average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output: Tensor, target, topk=(1,)):
    """topk accuracy within mini_batch"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # values, indices
        pred = pred.t() # [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # [maxk, batch_size] (bool)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) # topk accuracy within mini_batch
        return res

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger=None, distributed=False, local_rank=0):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", ncols=60, dynamic_ncols=True,
                        unit='batch', mininterval=1)
    else:
        pbar = train_loader
        
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if distributed:
        metrics_tensor = torch.tensor([losses.sum, losses.count, top1.sum, top1.count], device=device)
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        
        train_loss_sum = metrics_tensor[0].item()
        train_loss_count = metrics_tensor[1].item()
        train_acc_sum = metrics_tensor[2].item()
        train_acc_count = metrics_tensor[3].item()
        
        train_loss = train_loss_sum / train_loss_count
        train_acc = train_acc_sum / train_acc_count
    else:
        train_loss = losses.avg
        train_acc = top1.avg
    
    if local_rank == 0 and logger is not None:
        logger.log_scalar('train/loss', train_loss, epoch)
        logger.log_scalar('train/accuracy', train_acc, epoch)
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device, epoch, logger=None, distributed=False, local_rank=0):
    """accumulate loss and topk accuracy, traverse the whole validation set"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        if local_rank == 0:
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", ncols=60, dynamic_ncols=True,
                        unit='batch', mininterval=1)
        else:
            pbar = val_loader
            
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
    
    if distributed:
        metrics_tensor = torch.tensor([losses.sum, losses.count, top1.sum, top1.count], device=device)
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        
        val_loss_sum = metrics_tensor[0].item()
        val_loss_count = metrics_tensor[1].item()
        val_acc_sum = metrics_tensor[2].item()
        val_acc_count = metrics_tensor[3].item()
        
        val_loss = val_loss_sum / val_loss_count
        val_acc = val_acc_sum / val_acc_count
    else:
        val_loss = losses.avg
        val_acc = top1.avg
    
    if logger is not None and local_rank == 0:
        logger.log_scalar('val/loss', val_loss, epoch)
        logger.log_scalar('val/accuracy', val_acc, epoch)
    
    return val_loss, val_acc

def get_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('factor', 0.1))
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('t_max', 10))
    elif scheduler_name == 'reduce_lr_on_plateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode=kwargs.get('mode', 'max'), 
            factor=kwargs.get('factor', 0.1), 
            patience=kwargs.get('patience', 3), 
            verbose=True
        )
    else:
        raise ValueError(f"不支持的 lr scheduler: {scheduler_name}")

def get_optimizer(optimizer_name, params, **kwargs):
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(
            params,
            lr=kwargs.get('lr', 0.0001),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0001),
            nesterov=kwargs.get('nesterov', False)
        )
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(
            params,
            lr=kwargs.get('lr', 0.001),
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0001)
        )
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            params,
            lr=kwargs.get('lr', 0.001),
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0001)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
 

def train_model(config, args):

    # ----------------------------- load hyperparameters -----------------------------
    data_config = config['data']
    data_dir = data_config['data_dir']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    seed = data_config['seed']
    train_size = data_config['train_size']
    val_size = data_config['val_size']
    test_size = data_config['test_size']
    
    model_config = config['model']
    backbone = model_config['backbone']
    pretrained = model_config['pretrained']
    num_classes = model_config.get('num_classes', 101)
    
    training_config = config['training']
    num_epochs = training_config['num_epochs']
    base_lr = float(training_config['base_lr'])
    fc_lr_multiplier = float(training_config['fc_lr_multiplier'])
    weight_decay = float(training_config['weight_decay'])
    optimizer_name = training_config.get('optimizer', 'sgd') 
    # SGD
    momentum = float(training_config.get('momentum', 0.9))
    nesterov = training_config.get('nesterov', False)
    # Adam/AdamW
    beta1 = float(training_config.get('beta1', 0.9))
    beta2 = float(training_config.get('beta2', 0.999))
    eps = float(training_config.get('eps', 1e-8))
    
    scheduler_name = training_config['scheduler']
    patience = training_config['patience']
    factor = float(training_config['factor'])
    early_stopping = training_config['early_stopping']
    
    experiment_config = config['experiment']
    experiment_name = experiment_config['name']
    if args.experiment_name:
        experiment_name = args.experiment_name
    checkpoint_dir = os.path.join(experiment_config['checkpoint_dir'], experiment_name)
    log_dir = experiment_config['log_dir']
    save_best_only = experiment_config['save_best_only']
    save_every = experiment_config['save_every']
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    distributed = False
    if args.distributed:
        distributed = True
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = args.local_rank
            
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = 1
        local_rank = 0

    per_gpu_batch_size = batch_size
    
    if distributed:
        # DDP
        total_batch_size = per_gpu_batch_size * world_size
        dataloader_batch_size = per_gpu_batch_size
    else:
        if world_size > 1:
            # DP multi gpu
            total_batch_size = per_gpu_batch_size * world_size
            dataloader_batch_size = total_batch_size
        else:
            # single gpu
            total_batch_size = per_gpu_batch_size
            dataloader_batch_size = per_gpu_batch_size
    
    base_lr = base_lr * (total_batch_size / per_gpu_batch_size)

    if local_rank == 0:
        print(f"实验名称: {experiment_name}")
        print(f"backbone: {backbone}")
        print(f"加载ResNet预训练参数: {'Yes' if pretrained else 'No'}")
        print(f"GPU数量: {world_size}")
        print(f"分布式训练: {'Yes' if distributed else 'No'}")
        print(f"per gpu batch_size: {per_gpu_batch_size}")
        print(f"global batch_size: {total_batch_size}")
        print(f"optimizer: {optimizer_name}")
        print(f"base_lr: {base_lr}")
        print(f"FC层学习率倍数: {fc_lr_multiplier}")
        print(f"weight_decay: {weight_decay}")
        if optimizer_name.lower() in ['adam', 'adamw']:
            print(f"{optimizer_name}参数 - beta1: {beta1}, beta2: {beta2}, eps: {eps}")
        elif optimizer_name.lower() == 'sgd':
            print(f"SGD参数 - momentum: {momentum}, nesterov: {nesterov}")
        print(f"Checkpoint保存模式: {'仅保存Best和Last' if save_best_only else f'每 {save_every} 个epoch保存一次'}")
        print(f"使用设备: {device}")
    


    # ----------------------------- load logger dataloader model loss optimizer scheduler -----------------------------
    if local_rank == 0:
        logger = TensorboardLogger(log_dir=log_dir, experiment_name=experiment_name)
    else:
        logger = None
    
    train_loader, val_loader, test_loader, label2name = create_dataloaders(
        data_dir=data_dir, 
        batch_size=dataloader_batch_size,
        num_workers=num_workers,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        distributed=distributed
    )
    
    model = get_model(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    model = model.to(device)
    
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif world_size > 1:
        model = DataParallel(model)
    
    if local_rank == 0 and logger is not None:
        try:
            sample_input = next(iter(train_loader))[0][:1].to(device)
            logger.log_model_graph(model.module if hasattr(model, 'module') else model, sample_input)
        except Exception as e:
            print(f"Logger 记录模型结构失败: {e}")
    
    criterion = nn.CrossEntropyLoss()
    
    if hasattr(model, 'module'): # DP or DDP
        backbone_params = [p for n, p in model.module.named_parameters() if "fc" not in n]
        fc_params = [p for n, p in model.module.named_parameters() if "fc" in n]
    else:
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        fc_params = [p for n, p in model.named_parameters() if "fc" in n]
    

    optimizer_params = [
        {'params': backbone_params, 'lr': base_lr},
        {'params': fc_params, 'lr': base_lr * fc_lr_multiplier}
    ]
    
    optimizer_kwargs = {
        'lr': base_lr,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'momentum': momentum,
        'nesterov': nesterov
    }
    
    optimizer = get_optimizer(optimizer_name, optimizer_params, **optimizer_kwargs)

    
    scheduler_kwargs = {
        'patience': patience,
        'factor': factor,
        't_max': num_epochs,
        'step_size': num_epochs // 3,
        'mode': 'max'
    }
    scheduler = get_scheduler(scheduler_name, optimizer, **scheduler_kwargs)
    

    
    # ----------------------------- Training -----------------------------

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    if local_rank == 0:
        print(f"开始训练 {backbone} 模型，共 {num_epochs} 个epochs")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1): # epoch: [1, num_epochs]

        if distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            logger if local_rank == 0 else None, distributed, local_rank
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, 
            logger if local_rank == 0 else None, distributed, local_rank
        )
        
        if local_rank == 0:
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            epoch_info = {
                'Epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            print(f"Epoch {epoch}/{num_epochs} result:")
            for k, v in epoch_info.items():
                if k == 'epoch':
                    print(f"{k}: {v}", end=' | ')
                else:
                    print(f"{k}: {v:.4f}", end=' | ')
            print()
            
        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if local_rank == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_val_acc': best_val_acc,
                'config': config,
            }
            
            if is_best:
                save_checkpoint(
                    save_dict, 
                    checkpoint_dir=checkpoint_dir,
                    filename='checkpoint_best.pth'
                )
                print(f"已保存最佳模型到 {os.path.join(checkpoint_dir, f'checkpoint_best_{experiment_name}.pth')}")
                
            if save_best_only:
                if epoch == num_epochs:
                    save_checkpoint(
                        save_dict, 
                        checkpoint_dir=checkpoint_dir,
                        filename=f'checkpoint_epoch{epoch}.pth'
                    )
            else:
                if epoch % save_every == 0 or epoch == num_epochs:
                    save_checkpoint(
                        save_dict, 
                        checkpoint_dir=checkpoint_dir,
                        filename=f'checkpoint_epoch{epoch}.pth'
                    )
            
        if early_stopping > 0 and patience_counter >= early_stopping:
            if local_rank == 0:
                print(f"验证准确率 {early_stopping} 个epoch未提高, 提前停止训练")
                save_checkpoint(
                    save_dict, 
                    checkpoint_dir=checkpoint_dir,
                    filename=f'checkpoint_epoch{epoch}.pth'
                )
            if distributed:
                dist.barrier() 
            break
    
    # ----------------------------- Train Over -----------------------------
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"
    
    if local_rank == 0:
        print(f"\n模型训练完成! 总耗时: {time_str}")
        print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        hparams = {
            'backbone': backbone,
            'pretrained': pretrained,
            'batch_size': total_batch_size,
            'optimizer': optimizer_name,
            'base_lr': base_lr,
            'fc_lr_multiplier': fc_lr_multiplier,
        }
        metrics = {
            'hparam/best_train_accuracy': best_train_acc,
            'hparam/best_val_accuracy': best_val_acc,
            'hparam/final_train_accuracy': train_accs[-1],
            'hparam/final_val_accuracy': val_accs[-1],
            'hparam/best_epoch': best_epoch,
        }
 
        print("最终指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
 
        if logger:
            logger.log_hyperparams(hparams, metrics)
            
            plt.figure(figsize=(12, 5))
            plt.suptitle(f'{experiment_name} - Training Curves', fontsize=16)
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Acc')
            plt.plot(val_accs, label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title(f'Accuracy Curves')
            
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, f'{experiment_name}_curves.png'))
            
            logger.log_figure('Training Curves', plt.gcf(), 0)
            plt.close()
            
            logger.close()
    
    if distributed:
        dist.destroy_process_group()


def main():
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description='finetune ResNet on Caltech-101')
    parser.add_argument('--config', type=str, required=True, help='配置文件(yaml)路径')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU ID') # '0,1,2,3'
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称') # assign unique experiment name, None then use the .yaml experiment name
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练(DDP)')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0, help='分布式训练的本地进程标识')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config, args)


if __name__ == "__main__":
    main() 