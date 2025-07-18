import torch
import torch.nn as nn
import torchvision.models as models


# Define the three components of DenseNet-121 for split learning
class DenseNet121Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1):
        super(DenseNet121Head, self).__init__()
        # Load pretrained DenseNet-121 model
        densenet = models.densenet121(pretrained=False)
        
        # Extract features up to the cut layer
        self.features = nn.Sequential()
        
        # Initial conv layers (always included)
        self.features.add_module('conv0', nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(64))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # If using pretrained weights and in_channels is 3, copy the pretrained weights
        if in_channels == 3:
            self.features.conv0.weight.data = densenet.features.conv0.weight.data
            self.features.norm0.weight.data = densenet.features.norm0.weight.data
            self.features.norm0.bias.data = densenet.features.norm0.bias.data
            self.features.norm0.running_mean.data = densenet.features.norm0.running_mean.data
            self.features.norm0.running_var.data = densenet.features.norm0.running_var.data
            
        # Add dense blocks and transitions based on cut_layer
        if cut_layer >= 1:
            # Dense block 1 + transition 1
            self.features.add_module('denseblock1', densenet.features.denseblock1)
            self.features.add_module('transition1', densenet.features.transition1)
            
        if cut_layer >= 2:
            # Dense block 2 + transition 2
            self.features.add_module('denseblock2', densenet.features.denseblock2)
            self.features.add_module('transition2', densenet.features.transition2)
            
        if cut_layer >= 3:
            # Dense block 3 + transition 3
            self.features.add_module('denseblock3', densenet.features.denseblock3)
            self.features.add_module('transition3', densenet.features.transition3)
            
        if cut_layer >= 4:
            # Dense block 4 + norm5
            self.features.add_module('denseblock4', densenet.features.denseblock4)
            self.features.add_module('norm5', densenet.features.norm5)
            
        self.cut_layer = cut_layer
    
    def forward(self, x):
        return self.features(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class DenseNet121Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(DenseNet121Backbone, self).__init__()
        # Load pretrained DenseNet-121 model
        densenet = models.densenet121(pretrained=False)
        
        self.features = nn.Sequential()
        
        # Add layers based on cut_layer
        if cut_layer < 1:
            # Dense block 1 + transition 1
            self.features.add_module('denseblock1', densenet.features.denseblock1)
            self.features.add_module('transition1', densenet.features.transition1)
            
        if cut_layer < 2:
            # Dense block 2 + transition 2
            self.features.add_module('denseblock2', densenet.features.denseblock2)
            self.features.add_module('transition2', densenet.features.transition2)
            
        if cut_layer < 3:
            # Dense block 3 + transition 3
            self.features.add_module('denseblock3', densenet.features.denseblock3)
            self.features.add_module('transition3', densenet.features.transition3)
            
        if cut_layer < 4:
            # Dense block 4 + norm5
            self.features.add_module('denseblock4', densenet.features.denseblock4)
            self.features.add_module('norm5', densenet.features.norm5)
            
        # Final layers
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cut_layer = cut_layer
    
    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class DenseNet121Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNet121Tail, self).__init__()
        # Create the final classifier layer
        # DenseNet121 has 1024 features after avgpool
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        return self.classifier(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path)) 