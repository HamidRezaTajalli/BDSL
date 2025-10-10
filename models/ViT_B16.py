import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


# Define the three components of ViT-B/16 for split learning
class ViTB16Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1, img_size=224, patch_size=16):
        super(ViTB16Head, self).__init__()
        
        # Load pretrained ViT-B/16 model from timm
        vit_full = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # ViT-B/16 has 12 transformer blocks
        # cut_layer determines how many blocks go to the head
        # cut_layer=0: 1 block in head, cut_layer=1: 2 blocks in head, etc.
        self.cut_layer = cut_layer
        self.num_head_blocks = cut_layer + 1
        
        # Extract components from the original ViT (no need to keep full model)
        self.patch_embed = vit_full.patch_embed
        self.cls_token = vit_full.cls_token
        self.pos_embed = vit_full.pos_embed
        self.pos_drop = vit_full.pos_drop
        
        # Extract only the blocks that belong to the head
        self.blocks = nn.ModuleList()
        for i in range(self.num_head_blocks):
            if i < len(vit_full.blocks):
                self.blocks.append(vit_full.blocks[i])
        
        # Clear reference to full model to save memory
        del vit_full
        
        # Modify patch embedding if input channels are different
        if in_channels != 3:
            self.patch_embed.proj = nn.Conv2d(
                in_channels, self.patch_embed.proj.out_channels,
                kernel_size=self.patch_embed.proj.kernel_size,
                stride=self.patch_embed.proj.stride,
                padding=self.patch_embed.proj.padding
            )
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token and position embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        # Pass through head blocks
        for block in self.blocks:
            x = block(x)
        
        return x
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class ViTB16Backbone(nn.Module):
    def __init__(self, cut_layer=1):
        super(ViTB16Backbone, self).__init__()
        
        # Load pretrained ViT-B/16 model from timm
        vit_full = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        self.cut_layer = cut_layer
        self.num_head_blocks = cut_layer + 1
        
        # Extract remaining blocks for backbone
        self.blocks = nn.ModuleList()
        total_blocks = len(vit_full.blocks)
        
        for i in range(self.num_head_blocks, total_blocks):
            self.blocks.append(vit_full.blocks[i])
        
        # Final normalization layer
        self.norm = vit_full.norm
        
        # Clear reference to full model to save memory
        del vit_full
    
    def forward(self, x):
        # Pass through remaining blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Extract class token (first token)
        cls_token = x[:, 0]
        
        return cls_token
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class ViTB16Tail(nn.Module):
    def __init__(self, num_classes=1000):
        super(ViTB16Tail, self).__init__()
        
        # ViT-B/16 has 768 dimensional embeddings
        self.head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        return self.head(x)
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path):
        self.load_state_dict(torch.load(path))


class ViTB16Decoder(nn.Module):
    """Decoder for ViT-B/16 to reconstruct images from backbone output"""
    def __init__(self, cut_layer, input_size=(224, 224)):
        super(ViTB16Decoder, self).__init__()
        self.cut_layer = cut_layer
        self.input_size = input_size
        
        # For ViT-B/16, the input is [B, 197, 768] (seq_len=197, embed_dim=768)
        # We need to convert this to 224x224 RGB images
        
        # First, we reshape the sequence to a spatial representation
        # 197 tokens = 1 CLS token + 196 patch tokens (14x14 patches)
        # We'll use the 196 patch tokens and reshape to [B, 768, 14, 14]
        
        self.patch_to_spatial = nn.Sequential(
            # Input: [B, 196*768] -> [B, 768, 14, 14]
            nn.Linear(196 * 768, 768 * 14 * 14),
            nn.ReLU()
        )
        
        # Then use transposed convolutions to upsample to 224x224
        if cut_layer == 0:
            # Early cut: more upsampling needed
            self.decoder = nn.Sequential(
                # Start from [B, 768, 14, 14]
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 1:
            # Second cut
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 2:
            # Third cut
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 3:
            # Fourth cut
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif cut_layer == 4:
            # Fifth cut
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        else:
            raise Exception(f"Cut layer {cut_layer} not supported for ViT-B/16")
    
    def forward(self, backbone_output):
        """Forward pass through decoder
        For ViT-B/16: backbone_output is [B, 197, 768]
        """
        # Extract patch tokens (exclude CLS token at index 0)
        patch_tokens = backbone_output[:, 1:, :]  # [B, 196, 768]
        
        # Flatten patch tokens
        patch_flat = patch_tokens.view(patch_tokens.size(0), -1)  # [B, 196*768]
        
        # Convert to spatial representation
        spatial_features = self.patch_to_spatial(patch_flat)  # [B, 768*14*14]
        spatial_features = spatial_features.view(spatial_features.size(0), 768, 14, 14)  # [B, 768, 14, 14]
        
        # Pass through decoder
        reconstructed = self.decoder(spatial_features)
        
        # Convert from [-1, 1] to [0, 1]
        reconstructed = (reconstructed + 1) / 2
        
        return reconstructed 