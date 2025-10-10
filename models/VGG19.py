import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1, pretrained=False):
        """
        cut_layer: index in vgg.features where we cut (e.g. 0..36)
        in_channels: number of input channels (default=3)
        """
        
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights)

        cut_dict = {0: 5, 1: 10, 2: 19, 3: 28}

        # Take layers up to cut_layer
        self.features = nn.Sequential(*list(vgg.features.children())[:cut_dict[cut_layer]])

        # Optionally modify first conv layer
        if in_channels != 3:
            old_conv = self.features[0]
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding)
            self.features[0] = new_conv

    def forward(self, x):
        return self.features(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))

class VGG19Backbone(nn.Module):
    def __init__(self, cut_layer=1, pretrained=False):
        """
        cut_layer: index in vgg.features where head stopped
        """
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights)

        cut_dict = {0: 5, 1: 10, 2: 19, 3: 28}

        # Take layers after cut_layer
        self.features = nn.Sequential(*list(vgg.features.children())[cut_dict[cut_layer]:])

        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])  # keep up to 4096-dim

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
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights)

        # Only final FC layer
        self.fc = vgg.classifier[-1]

        # Replace for custom num_classes
        if num_classes != 1000:
            self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.fc(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class VGG19Decoder(nn.Module):
    """Decoder for VGG19 to reconstruct images from backbone output
    
    Note: Backbone outputs flattened features [B, 4096] after avgpool + classifier.
    The decoder first projects this to spatial representation then upsamples to [B, 3, 224, 224].
    """
    def __init__(self, cut_layer, input_size=(224, 224)):
        super(VGG19Decoder, self).__init__()
        self.cut_layer = cut_layer
        self.input_size = input_size
        
        # VGG19 backbone always outputs [B, 4096] after avgpool + classifier
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


