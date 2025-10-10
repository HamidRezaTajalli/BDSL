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
    """Decoder for ResNet50 to reconstruct images from backbone output
    
    Note: Backbone outputs flattened features [B, 2048] after avgpool + flatten.
    The decoder first reshapes this to [B, 2048, 7, 7] then upsamples to [B, 3, 224, 224].
    """
    def __init__(self, cut_layer, input_size=(224, 224)):
        super(ResNet50Decoder, self).__init__()
        self.cut_layer = cut_layer
        self.input_size = input_size
        
        # ResNet50 backbone always outputs [B, 2048] after avgpool + flatten
        # Strategy: Linear projection to expand features, then reshape and upsample
        self.initial_size = 7  # Target spatial size after reshape
        self.initial_channels = 2048
        
        # Project flattened features to spatial representation
        # [B, 2048] -> [B, 2048*7*7]
        self.fc_projection = nn.Sequential(
            nn.Linear(2048, 2048 * self.initial_size * self.initial_size),
            nn.ReLU(inplace=True)
        )
        
        # Build decoder architecture - same for all cut layers since backbone output is always [B, 2048]
        # Input: [B, 2048, 7, 7] after projection & reshape -> Upsample to [B, 3, 224, 224]
        self.decoder = nn.Sequential(
            # Start from [B, 2048, 7, 7]
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
    
    def forward(self, backbone_output):
        """Forward pass through decoder
        
        Args:
            backbone_output: Flattened features from backbone [B, 2048]
            
        Returns:
            reconstructed: Reconstructed image [B, 3, 224, 224] in range [0, 1]
        """
        batch_size = backbone_output.size(0)
        
        # Project from [B, 2048] to [B, 2048*7*7]
        projected = self.fc_projection(backbone_output)
        
        # Reshape to [B, 2048, 7, 7]
        spatial_features = projected.view(batch_size, self.initial_channels, 
                                         self.initial_size, self.initial_size)
        
        # Pass through decoder
        reconstructed = self.decoder(spatial_features)
        
        # Convert from [-1, 1] to [0, 1]
        reconstructed = (reconstructed + 1) / 2
        return reconstructed
