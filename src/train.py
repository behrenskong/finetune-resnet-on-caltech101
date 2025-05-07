import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import time
from datetime import datetime
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from model import get_model
from data import create_dataloaders
from logger import TensorboardLogger

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

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger=None):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
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
        
        pbar.set_postfix({'loss': losses.avg, 'acc@1': top1.avg}) # update progress bar
        
        if logger is not None:
            step = (epoch - 1) * len(train_loader) + batch_idx
            logger.log_scalar('train/loss', loss.item(), step)
            logger.log_scalar('train/accuracy', acc1.item(), step)
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device, epoch, logger=None):
    """accumulate loss and topk accuracy, traverse the whole validation set"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))

            pbar.set_postfix({'loss': losses.avg, 'acc@1': top1.avg}) # accumulative average
            

            if logger is not None:
                step = (epoch - 1) * len(val_loader) + batch_idx
                logger.log_scalar('val/loss', loss.item(), step)
                logger.log_scalar('val/accuracy', acc1.item(), step)
    
    if logger is not None:
        logger.log_scalar('val/epoch_loss', losses.avg, epoch)
        logger.log_scalar('val/epoch_accuracy', top1.avg, epoch)
    
    return losses.avg, top1.avg

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
    
    print(f"实验名称: {experiment_name}")
    print(f"backbone: {backbone}")
    print(f"加载ResNet预训练参数: {'Yes' if pretrained else 'No'}")
    print(f"batch_size: {batch_size}")
    print(f"优化器: AdamW")
    print(f"base_lr: {base_lr}")
    print(f"FC层学习率倍率: {fc_lr_multiplier}")
    print(f"weight_decay: {weight_decay}")
    if optimizer_name.lower() in ['adam', 'adamw']:
        print(f"{optimizer_name}参数 - beta1: {beta1}, beta2: {beta2}, eps: {eps}")
    elif optimizer_name.lower() == 'sgd':
        print(f"SGD参数 - momentum: {momentum}, nesterov: {nesterov}")
    print(f"Checkpoint保存模式: {'仅保存Best和Last' if save_best_only else f'每 {save_every} 个epoch保存一次'}")

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    


    # ----------------------------- load logger dataloader model loss optimizer scheduler -----------------------------

    logger = TensorboardLogger(log_dir=log_dir, experiment_name=experiment_name)

    train_loader, val_loader, test_loader, label2name = create_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    
    model = get_model(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    model = model.to(device)
    
    try:
        sample_input = next(iter(train_loader))[0][:1].to(device)
        logger.log_model_graph(model, sample_input) # visualize model architecture by tensorboard
    except Exception as e:
        print(f"Logger 记录模型结构失败: {e}")
    
    criterion = nn.CrossEntropyLoss()
    
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
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"开始训练 {backbone} 模型，共 {num_epochs} 个epochs")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1): # epoch: [1, num_epochs]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, logger
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_info = {
            'Epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        for k, v in epoch_info.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}", end=' | ')
            else:
                print(f"{k}: {v}", end=' | ')
        print()
        
        logger.log_scalar('epoch/train_loss', train_loss, epoch)
        logger.log_scalar('epoch/train_accuracy', train_acc, epoch)
        logger.log_scalar('epoch/val_loss', val_loss, epoch)
        logger.log_scalar('epoch/val_accuracy', val_acc, epoch)
        
        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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
            print(f"已保存最佳模型到 {os.path.join(checkpoint_dir, 'checkpoint_best.pth')}")
            
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
            print(f"验证准确率 {early_stopping} 个epoch未提高, 提前停止训练")
            save_checkpoint(
                save_dict, 
                checkpoint_dir=checkpoint_dir,
                filename=f'checkpoint_epoch{epoch}.pth'
            )
            break
    
    # ----------------------------- Train Over -----------------------------
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"
    
    print(f"\n模型训练完成! 总耗时: {time_str}")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    hparams = {
        'backbone': backbone,
        'pretrained': pretrained,
        'batch_size': batch_size,
        'base_lr': base_lr,
        'fc_lr_multiplier': fc_lr_multiplier,
    }
    metrics = {
        'hparam/best_val_accuracy': best_val_acc,
        'hparam/final_train_accuracy': train_accs[-1],
        'hparam/final_val_accuracy': val_accs[-1],
        'hparam/best_epoch': best_epoch,
    }

    print("最终指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    logger.log_hyperparams(hparams, metrics)
    
    plt.figure(figsize=(12, 5))
    
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
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'{experiment_name}_curves.png'))
    
    logger.log_figure('Training Curves', plt.gcf(), 0)
    plt.close()

    logger.close()

def main():
    parser = argparse.ArgumentParser(description='finetune ResNet on Caltech-101')
    parser.add_argument('--config', type=str, required=True, help='配置文件(yaml)路径')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU ID') # '0,1,2,3'
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称') # assign unique experiment name, None then use the .yaml experiment name
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config, args)


if __name__ == "__main__":
    main() 