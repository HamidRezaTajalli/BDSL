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


class VGG11Decoder(nn.Module):
    """Decoder for VGG11 to reconstruct images from backbone output
    
    Note: Backbone outputs flattened features [B, 4096] after avgpool + classifier.
    The decoder first projects this to spatial representation then upsamples to [B, 3, 224, 224].
    """
    def __init__(self, cut_layer, input_size=(224, 224)):
        super(VGG11Decoder, self).__init__()
        self.cut_layer = cut_layer
        self.input_size = input_size
        
        # VGG11 backbone always outputs [B, 4096] after avgpool + classifier
        # Strategy: Linear projection to expand features, then reshape and upsample
        self.initial_size = 7  # Target spatial size after reshape
        self.initial_channels = 512  # Use 512 channels for spatial representation
        
        # Project flattened features to spatial representation
        # [B, 4096] -> [B, 512*7*7]
        self.fc_projection = nn.Sequential(
            nn.Linear(4096, self.initial_channels * self.initial_size * self.initial_size),
            nn.ReLU(inplace=True)
        )
        
        # Build decoder architecture - same for all cut layers since backbone output is always [B, 4096]
        # Input: [B, 512, 7, 7] after projection & reshape -> Upsample to [B, 3, 224, 224]
        self.decoder = nn.Sequential(
            # Start from [B, 512, 7, 7]
            # Upsample to 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 14, 14]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample to 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample to 112x112
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample to 224x224
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 224, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer to RGB
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
            nn.Tanh()
        )
    
    def forward(self, backbone_output):
        """Forward pass through decoder
        
        Args:
            backbone_output: Flattened features from backbone [B, 4096]
            
        Returns:
            reconstructed: Reconstructed image [B, 3, 224, 224] in range [0, 1]
        """
        batch_size = backbone_output.size(0)
        
        # Project from [B, 4096] to [B, 512*7*7]
        projected = self.fc_projection(backbone_output)
        
        # Reshape to [B, 512, 7, 7]
        spatial_features = projected.view(batch_size, self.initial_channels, 
                                         self.initial_size, self.initial_size)
        
        # Pass through decoder
        reconstructed = self.decoder(spatial_features)
        
        # Convert from [-1, 1] to [0, 1]
        reconstructed = (reconstructed + 1) / 2
        return reconstructed
