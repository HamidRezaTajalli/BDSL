import torch
import torch.nn as nn
import torchvision.models as models

# Define the three components of VGG-19 for split learning
class VGG19Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1):
        super(VGG19Head, self).__init__()
        # Load pretrained VGG-19 model
        vgg = models.vgg19(pretrained=False)
        
        # Modify the first conv layer to accept different number of input channels
        self.features = nn.Sequential()
        
        # Block 1: 2 conv layers (64 filters) + maxpool
        self.features.add_module('conv1_1', nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        self.features.add_module('relu1_1', nn.ReLU(inplace=True))
        self.features.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.features.add_module('relu1_2', nn.ReLU(inplace=True))
        self.features.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        
        # If using pretrained weights and in_channels is 3, copy the pretrained weights
        if in_channels == 3:
            self.features.conv1_1.weight.data = vgg.features[0].weight.data
            self.features.conv1_2.weight.data = vgg.features[2].weight.data
            
        # Add remaining layers based on cut_layer
        if cut_layer >= 1:
            # Block 2: 2 conv layers (128 filters) + maxpool
            self.features.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2_1', nn.ReLU(inplace=True))
            self.features.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2_2', nn.ReLU(inplace=True))
            self.features.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 2:
            # Block 3: 4 conv layers (256 filters) + maxpool
            self.features.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_1', nn.ReLU(inplace=True))
            self.features.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_2', nn.ReLU(inplace=True))
            self.features.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_3', nn.ReLU(inplace=True))
            self.features.add_module('conv3_4', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 3:
            # Block 4: 4 conv layers (512 filters) + maxpool
            self.features.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_1', nn.ReLU(inplace=True))
            self.features.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_2', nn.ReLU(inplace=True))
            self.features.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_3', nn.ReLU(inplace=True))
            self.features.add_module('conv4_4', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer >= 4:
            # Block 5: 4 conv layers (512 filters) + maxpool
            self.features.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_1', nn.ReLU(inplace=True))
            self.features.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_2', nn.ReLU(inplace=True))
            self.features.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_3', nn.ReLU(inplace=True))
            self.features.add_module('conv5_4', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.cut_layer = cut_layer
    
    def forward(self, x):
        return self.features(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class VGG19Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(VGG19Backbone, self).__init__()
        # Load pretrained VGG-19 model
        vgg = models.vgg19(pretrained=False)
        
        self.features = nn.Sequential()
        
        # Add layers based on cut_layer
        if cut_layer < 1:
            # Block 2: 2 conv layers (128 filters) + maxpool
            self.features.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2_1', nn.ReLU(inplace=True))
            self.features.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1))
            self.features.add_module('relu2_2', nn.ReLU(inplace=True))
            self.features.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 2:
            # Block 3: 4 conv layers (256 filters) + maxpool
            self.features.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_1', nn.ReLU(inplace=True))
            self.features.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_2', nn.ReLU(inplace=True))
            self.features.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_3', nn.ReLU(inplace=True))
            self.features.add_module('conv3_4', nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.features.add_module('relu3_4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 3:
            # Block 4: 4 conv layers (512 filters) + maxpool
            self.features.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_1', nn.ReLU(inplace=True))
            self.features.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_2', nn.ReLU(inplace=True))
            self.features.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_3', nn.ReLU(inplace=True))
            self.features.add_module('conv4_4', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu4_4', nn.ReLU(inplace=True))
            self.features.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
            
        if cut_layer < 4:
            # Block 5: 4 conv layers (512 filters) + maxpool
            self.features.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_1', nn.ReLU(inplace=True))
            self.features.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_2', nn.ReLU(inplace=True))
            self.features.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_3', nn.ReLU(inplace=True))
            self.features.add_module('conv5_4', nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.features.add_module('relu5_4', nn.ReLU(inplace=True))
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

class VGG19Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19Tail, self).__init__()
        # Only the final linear layer in the tail
        self.fc = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path)) 