import torch
import torchvision
import torch.nn as nn

class FineTuneResNet(nn.Module):
    def __init__(self, num_classes=101, backbone='resnet34', pretrained=True):
        super(FineTuneResNet, self).__init__()

        if backbone == 'resnet18':
            base_model = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            base_model = torchvision.models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            base_model = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            base_model = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        in_features = base_model.fc.in_features
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        return out

def get_model(num_classes=101, backbone='resnet34', pretrained=True):
    model = FineTuneResNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    return model