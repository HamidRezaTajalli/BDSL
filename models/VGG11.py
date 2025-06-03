import torch
import torch.nn as nn
import torchvision.models as models

# Define the three components of VGG-11 for split learning
class VGG11Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1):
        super(VGG11Head, self).__init__()
        # Load pretrained VGG-11 model
        vgg = models.vgg11(pretrained=False)
        
        # Modify the first conv layer to accept different number of input channels
        self.features = nn.Sequential()
        
        # Add initial layers (conv1, relu1, maxpool1)
        self.features.add_module('conv1', nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        
        # If using pretrained weights and in_channels is 3, copy the pretrained weights
        if in_channels == 3:
            self.features.conv1.weight.data = vgg.features[0].weight.data
            
        # Add remaining layers based on cut_layer
        if cut_layer >= 1:
            # Add conv2, relu2, maxpool2
            self.features.add_module('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2', nn.ReLU(inplace=True))
            self.features.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 2:
            # Add conv3, relu3, conv4, relu4, maxpool3
            self.features.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3', nn.ReLU(inplace=True))
            self.features.add_module('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 3:
            # Add conv5, relu5, conv6, relu6, maxpool4
            self.features.add_module('conv5', nn.Conv2d(256, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5', nn.ReLU(inplace=True))
            self.features.add_module('conv6', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu6', nn.ReLU(inplace=True))
            self.features.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 4:
            # Add conv7, relu7, conv8, relu8, maxpool5
            self.features.add_module('conv7', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu7', nn.ReLU(inplace=True))
            self.features.add_module('conv8', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu8', nn.ReLU(inplace=True))
            self.features.add_module('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.cut_layer = cut_layer
    
    def forward(self, x):
        return self.features(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class VGG11Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(VGG11Backbone, self).__init__()
        # Load pretrained VGG-11 model
        vgg = models.vgg11(pretrained=False)
        
        self.features = nn.Sequential()
        
        # Add layers based on cut_layer
        if cut_layer < 1:
            # Add conv2, relu2, maxpool2
            self.features.add_module('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2', nn.ReLU(inplace=True))
            self.features.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 2:
            # Add conv3, relu3, conv4, relu4, maxpool3
            self.features.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3', nn.ReLU(inplace=True))
            self.features.add_module('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 3:
            # Add conv5, relu5, conv6, relu6, maxpool4
            self.features.add_module('conv5', nn.Conv2d(256, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5', nn.ReLU(inplace=True))
            self.features.add_module('conv6', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu6', nn.ReLU(inplace=True))
            self.features.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 4:
            # Add conv7, relu7, conv8, relu8, maxpool5
            self.features.add_module('conv7', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu7', nn.ReLU(inplace=True))
            self.features.add_module('conv8', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu8', nn.ReLU(inplace=True))
            self.features.add_module('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Add the classifier layers (except the final linear layer)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        
        self.cut_layer = cut_layer
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class VGG11Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG11Tail, self).__init__()
        # Only the final linear layer in the tail
        self.fc = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))
