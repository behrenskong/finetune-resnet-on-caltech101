import os
import argparse
import yaml
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

import torch

from models.model import get_model
from dataloader.data import create_dataloaders

def test_model(model, test_loader, device):
    model.eval()
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            corrects += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Accuracy': 100 * corrects / total})
    
    accuracy = 100 * corrects / total
    return accuracy, np.array(all_preds), np.array(all_labels)

def run_test(config, args):

    data_config = config['data']
    data_dir = data_config['data_dir']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    seed = data_config['seed']
    train_size = data_config.get('train_size', 0.5)
    val_size = data_config.get('val_size', 0.25)
    test_size = data_config.get('test_size', 0.25)
    
    model_config = config['model']
    backbone = model_config['backbone']
    num_classes = model_config.get('num_classes', 101)
    
    experiment_name = args.experiment_name if args.experiment_name else config['experiment']['name']
    
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    _, _, test_loader, label2name = create_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    
    model = get_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    
    print(f"experiment_name: {experiment_name}")
    print(f"model_path: {args.model_path}")
    print(f"backbone: {backbone}")
    
    start_time = time.time()
    accuracy, all_preds, all_labels = test_model(model, test_loader, device)
    test_time = time.time() - start_time
    
    class_names = [label2name[i] for i in range(len(label2name))]
    
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("分类报告:")
    print(report_text)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"测试准确率: {accuracy:.2f}%\n")
        f.write(f"测试耗时: {test_time:.2f}秒\n\n")
        f.write("分类报告:\n")
        f.write(report_text)
    
    print(f"测试完成，结果保存在 {output_dir} 目录下")
    return accuracy, test_time

def main():
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU ID')
    parser.add_argument('--output_dir', type=str, default='outputs/test_results', help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    accuracy, test_time = run_test(config, args)
    print(f"测试准确率: {accuracy:.2f}%")
    print(f"测试耗时: {test_time:.2f}秒")

if __name__ == "__main__":
    main()