import torch
import torch.nn as nn
import torchvision.models as models


# Define the three components of ResNet-50 for split learning
class ResNet50Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1):
        super(ResNet50Head, self).__init__()
        # Load pretrained ResNet-50 model
        resnet = models.resnet50(pretrained=False)
        
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

class ResNet50Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(ResNet50Backbone, self).__init__()
        # Load pretrained ResNet-50 model
        resnet = models.resnet50(pretrained=False)
        
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

class ResNet50Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Tail, self).__init__()
        # Create the final FC layer for the tail
        # ResNet50 has 2048 features after avgpool, not 512 like ResNet18
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class ResNet50Decoder(nn.Module):
    """Decoder for ResNet50 to reconstruct images from backbone output"""
    def __init__(self, cut_layer, input_size=(224, 224)):
        super(ResNet50Decoder, self).__init__()
        self.cut_layer = cut_layer
        self.input_size = input_size
        
        # Build decoder architecture based on cut layer
        if cut_layer == 0:
            # Cut after first conv layer: expect [B, 64, 112, 112]
            self.decoder = nn.Sequential(
                # Upsample to 224x224
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 224, 224]
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(32, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()  # Output in [-1, 1] range
            )
        elif cut_layer == 1:
            # Cut after layer1: expect [B, 256, 56, 56]
            self.decoder = nn.Sequential(
                # Upsample to 112x112
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 224x224
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 2:
            # Cut after layer2: expect [B, 512, 28, 28]
            self.decoder = nn.Sequential(
                # Upsample to 56x56
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 224x224
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 3:
            # Cut after layer3: expect [B, 1024, 14, 14]
            self.decoder = nn.Sequential(
                # Upsample to 28x28
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 224x224
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 4:
            # Cut after layer4: expect [B, 2048, 7, 7]
            self.decoder = nn.Sequential(
                # Upsample to 14x14
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # [B, 1024, 14, 14]
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                
                # Upsample to 28x28
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 224x224
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        else:
            raise Exception(f"Cut layer {cut_layer} not supported for ResNet50")
    
    def forward(self, backbone_output):
        """Forward pass through decoder - backbone_output is [B, C, H, W]"""
        reconstructed = self.decoder(backbone_output)
        # Convert from [-1, 1] to [0, 1]
        reconstructed = (reconstructed + 1) / 2
        return reconstructed
 