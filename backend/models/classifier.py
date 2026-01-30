import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple

class MedicalImageClassifier(nn.Module):
    """
    Transfer learning based medical image classifier
    Supports: EfficientNet-B0, ResNet50, DenseNet121
    """
    def __init__(self, model_name: str = 'efficientnet', num_classes: int = 2, pretrained: bool = True):
        super(MedicalImageClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'efficientnet':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        elif model_name == 'densenet121':
            self.base_model = models.densenet121(pretrained=pretrained)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps for Grad-CAM visualization"""
        features = []
        if self.model_name == 'efficientnet':
            for name, module in self.base_model.features.named_children():
                x = module(x)
                if name == '8':  # Last conv layer
                    features.append(x)
        elif self.model_name == 'resnet50':
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            features.append(x)
        return features[-1]


def create_model(dataset_type: str, model_name: str = 'efficientnet') -> nn.Module:
    """
    Factory function to create appropriate model for each dataset
    
    Args:
        dataset_type: 'aptos', 'ham10000', or 'mura'
        model_name: 'efficientnet', 'resnet50', or 'densenet121'
    
    Returns:
        PyTorch model
    """
    num_classes_map = {
        'aptos': 5,      # 5 DR severity levels
        'ham10000': 7,   # 7 skin lesion types
        'mura': 2        # Binary classification
    }
    
    if dataset_type not in num_classes_map:
        raise ValueError(f"Dataset type {dataset_type} not supported")
    
    num_classes = num_classes_map[dataset_type]
    model = MedicalImageClassifier(model_name=model_name, num_classes=num_classes, pretrained=True)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    for dataset in ['aptos', 'ham10000', 'mura']:
        model = create_model(dataset, 'efficientnet')
        print(f"\n{dataset.upper()} Model:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
