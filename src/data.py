import os 
import glob
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Caltech101Dataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, self.labels[index]

def load_data(data_dir='data/caltech-101/101_ObjectCategories'):
    classes = sorted([f for f in os.listdir(data_dir) if f != 'BACKGROUND_Google'])
    label2name = {i: c for i, c in enumerate(classes)}
    name2label = {c: i for i, c in enumerate(classes)}
    
    image_paths = []
    labels = []
    
    for c in classes:
        class_dir = os.path.join(data_dir, c)
        for img_file in glob.glob(os.path.join(class_dir, '*.jpg')):
            image_paths.append(img_file)
            labels.append(name2label[c])
    
    labels = np.array(labels)
    
    print(f"共{len(image_paths)} 张图片, {len(classes)} 个类别")
    return image_paths, labels, label2name

def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transforms, val_transforms

def create_dataloaders(data_dir='data/caltech-101/101_ObjectCategories', batch_size=32, 
                        num_workers=8, seed=42, train_size=0.5, val_size=0.25, test_size=0.25):
    image_paths, labels, label2name = load_data(data_dir)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=val_size+test_size, stratify=labels, random_state=seed
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size/(val_size+test_size), stratify=temp_labels, random_state=seed
    )
    
    print(f"训练集: {len(train_paths)}")
    print(f"验证集: {len(val_paths)}")
    print(f"测试集: {len(test_paths)}")
    
    train_transforms, val_transforms = get_transforms()

    train_dataset = Caltech101Dataset(train_paths, train_labels, train_transforms)
    val_dataset = Caltech101Dataset(val_paths, val_labels, val_transforms)
    test_dataset = Caltech101Dataset(test_paths, test_labels, val_transforms)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, label2name