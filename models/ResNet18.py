import torch
import torch.nn as nn
import torchvision.models as models


# Define the three components of ResNet-18 for split learning
class ResNet18Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1):
        super(ResNet18Head, self).__init__()
        # Load pretrained ResNet-18 model
        resnet = models.resnet18(pretrained=False)
        
        # Modify the first conv layer to accept different number of input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If using pretrained weights and in_channels is 3, copy the pretrained weights
        if in_channels == 3:
            self.conv1.weight.data = resnet.conv1.weight.data
            
        # Extract the other initial layers for the head
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Add ResNet layers based on cut_layer parameter
        if cut_layer >= 1:
            self.layer1 = resnet.layer1
        if cut_layer >= 2:
            self.layer2 = resnet.layer2
        if cut_layer >= 3:
            self.layer3 = resnet.layer3
        if cut_layer >= 4:
            self.layer4 = resnet.layer4
            
        self.cut_layer = cut_layer
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.cut_layer >= 1:
            x = self.layer1(x)
        if self.cut_layer >= 2:
            x = self.layer2(x)
        if self.cut_layer >= 3:
            x = self.layer3(x)
        if self.cut_layer >= 4:
            x = self.layer4(x)
            
        return x
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class ResNet18Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(ResNet18Backbone, self).__init__()
        # Load pretrained ResNet-18 model
        resnet = models.resnet18(pretrained=False)
        
        # Add ResNet layers based on cut_layer parameter
        if cut_layer < 1:
            self.layer1 = resnet.layer1
        
        if cut_layer < 2:
            self.layer2 = resnet.layer2
        
        if cut_layer < 3:
            self.layer3 = resnet.layer3
        
        if cut_layer < 4:
            self.layer4 = resnet.layer4
            
        self.avgpool = resnet.avgpool
        self.cut_layer = cut_layer
    
    def forward(self, x):
        if self.cut_layer < 1 and hasattr(self, 'layer1'):
            x = self.layer1(x)
        if self.cut_layer < 2 and hasattr(self, 'layer2'):
            x = self.layer2(x)
        if self.cut_layer < 3 and hasattr(self, 'layer3'):
            x = self.layer3(x)
        if self.cut_layer < 4 and hasattr(self, 'layer4'):
            x = self.layer4(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class ResNet18Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18Tail, self).__init__()
        # Create the final FC layer for the tail
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


