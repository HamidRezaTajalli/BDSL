import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


# Define the three components of ViT-B/16 for split learning
class ViTB16Head(nn.Module):
    def __init__(self, in_channels=3, cut_layer=1, img_size=224, patch_size=16):
        super(ViTB16Head, self).__init__()
        
        # Load pretrained ViT-B/16 model from timm
        vit_full = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
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
        vit_full = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
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