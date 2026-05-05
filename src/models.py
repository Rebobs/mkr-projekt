import torch.nn as nn
from torchvision import models


def get_model(model_name: str, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if model_name == 'resnet18':
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 1)

    elif model_name == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 1)

    elif model_name == 'efficientnet_b0':
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)

    elif model_name == 'mobilenet_v3_small':
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)

    else:
        raise ValueError(f'Unknown model: {model_name}')

    if freeze_backbone:
        head_names = ('fc', 'classifier')
        for name, param in m.named_parameters():
            if not any(h in name for h in head_names):
                param.requires_grad = False

    return m


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / 1024 / 1024
