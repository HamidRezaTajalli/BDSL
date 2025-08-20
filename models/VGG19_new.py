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
        return self.classifier(x)


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
