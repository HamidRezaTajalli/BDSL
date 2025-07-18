from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import torchvision.models as models
import torchvision.utils as vutils
import timm
from models.ResNet18 import ResNet18Head, ResNet18Tail, ResNet18Backbone
from models.ResNet50 import ResNet50Head, ResNet50Tail, ResNet50Backbone
from models.VGG11 import VGG11Head, VGG11Tail, VGG11Backbone
from models.VGG19 import VGG19Head, VGG19Tail, VGG19Backbone
from models.DenseNet121 import DenseNet121Head, DenseNet121Tail, DenseNet121Backbone
from models.ViT_B16 import ViTB16Head, ViTB16Tail, ViTB16Backbone




class GatedFusion(nn.Module):
    def __init__(self, model_name, cut_layer, checkpoint_dir="./checkpoints"):
        super().__init__()
        
        self.model_name = model_name

        if model_name == 'resnet18':
            z1_dim = 512
            if cut_layer == 0:
                z2_channels = 64
            elif cut_layer == 1:
                z2_channels = 64
            elif cut_layer == 2:
                z2_channels = 128
            elif cut_layer == 3:
                z2_channels = 256
            elif cut_layer == 4:
                z2_channels = 512
        elif model_name == 'resnet50':
            z1_dim = 2048
            if cut_layer == 0:
                z2_channels = 64
            elif cut_layer == 1:
                z2_channels = 256
            elif cut_layer == 2:
                z2_channels = 512
            elif cut_layer == 3:
                z2_channels = 1024
            elif cut_layer == 4:
                z2_channels = 2048
        elif model_name == 'vgg11':
            z1_dim = 4096
            if cut_layer == 0:
                z2_channels = 64
            elif cut_layer == 1:
                z2_channels = 128
            elif cut_layer == 2:
                z2_channels = 256
            elif cut_layer == 3:
                z2_channels = 512
            elif cut_layer == 4:
                z2_channels = 512
        elif model_name == 'vgg19':
            z1_dim = 4096
            if cut_layer == 0:
                z2_channels = 64
            elif cut_layer == 1:
                z2_channels = 128
            elif cut_layer == 2:
                z2_channels = 256
            elif cut_layer == 3:
                z2_channels = 512
            elif cut_layer == 4:
                z2_channels = 512
        elif model_name == 'densenet121':
            z1_dim = 1024
            if cut_layer == 0:
                z2_channels = 64
            elif cut_layer == 1:
                z2_channels = 128
            elif cut_layer == 2:
                z2_channels = 256
            elif cut_layer == 3:
                z2_channels = 512
            elif cut_layer == 4:
                z2_channels = 1024
        elif model_name == 'vit_b16':
            z1_dim = 768
            # ViT maintains 768 embedding dimension throughout all layers
            z2_channels = 768
        else:
            raise Exception(f"Model {model_name} not supported")

        if model_name == 'vit_b16':
            # For ViT, z2 is [B, seq_len, embed_dim], we'll use all tokens except CLS
            # seq_len = 197 (196 patches + 1 CLS token)
            self.z2_proj = nn.Sequential(
                nn.Linear(z2_channels * 196, z1_dim * 2),  # 196 patch tokens
                nn.ReLU(),
                nn.Linear(z1_dim * 2, z1_dim)
            )
        else:
            # For CNN models, z2 is [B, C, H, W]
            self.spatial_size = 2  # or try 2, 4, 7, depending on how much detail you want
            self.pool_z2 = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))  # [B, C, 4, 4]
            self.z2_proj = nn.Sequential(
                nn.Linear(z2_channels * self.spatial_size * self.spatial_size, z1_dim * 2),
                nn.ReLU(),
                nn.Linear(z1_dim * 2, z1_dim)
            )
        self.gate = nn.Sequential(
            nn.Linear(z1_dim + z1_dim, z1_dim),  # concat of z1 and projected z2
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "gated_fusion.pth")



    def forward(self, z1, z2):
        if self.model_name == 'vit_b16':
            # For ViT: z2 is [B, seq_len, embed_dim] where seq_len = 197
            # Extract patch tokens (exclude CLS token at index 0)
            patch_tokens = z2[:, 1:, :]  # [B, 196, 768]
            z2_flat = patch_tokens.view(patch_tokens.size(0), -1)  # [B, 196*768]
            z2_proj = self.z2_proj(z2_flat)  # [B, z1_dim]
        else:
            # For CNN models: z2 is [B, C, H, W]
            z2_pooled = self.pool_z2(z2)  # [B, C, 4, 4]
            z2_flat = z2_pooled.view(z2_pooled.size(0), -1)  # [B, C*4*4]
            z2_proj = self.z2_proj(z2_flat)  # [B, z1_dim]

        z_cat = torch.cat([z1, z2_proj], dim=1)  # [B, z1_dim + z1_dim]
        g = self.gate(z_cat)  # [B, z1_dim]
        
        fused = g * z1 + (1 - g) * z2_proj  # weighted fusion
        return fused  # [B, z1_dim]



    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        print(f"GatedFusion saved to {self.checkpoint_path}")
    
    def load_model(self):
        if os.path.exists(self.checkpoint_path):
            self.load_state_dict(torch.load(self.checkpoint_path))
            print(f"GatedFusion loaded from {self.checkpoint_path}")
            return True
        return False



# Client implementation
class Client:
    def __init__(self, model_name, is_malicious, client_id, dataset, batch_size=32, num_classes=10, device='cpu', 
                 checkpoint_dir="./checkpoints", cut_layer=1):
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if model_name == 'resnet18':
            self.head = ResNet18Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = ResNet18Tail(num_classes=num_classes).to(device)
        elif model_name == 'resnet50':
            self.head = ResNet50Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = ResNet50Tail(num_classes=num_classes).to(device)
        elif model_name == 'vgg11':
            self.head = VGG11Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = VGG11Tail(num_classes=num_classes).to(device)
        elif model_name == 'vgg19':
            self.head = VGG19Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = VGG19Tail(num_classes=num_classes).to(device)
        elif model_name == 'densenet121':
            self.head = DenseNet121Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = DenseNet121Tail(num_classes=num_classes).to(device)
        elif model_name == 'vit_b16':
            self.head = ViTB16Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = ViTB16Tail(num_classes=num_classes).to(device)
        else:
            raise Exception(f"Model {model_name} not supported")
        
        self.criterion = nn.CrossEntropyLoss()
        self.head_optimizer = optim.Adam(self.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.tail_optimizer = optim.Adam(self.tail.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint paths
        self.head_checkpoint = os.path.join(checkpoint_dir, f"client_{client_id}_head.pth")
        self.tail_checkpoint = os.path.join(checkpoint_dir, f"client_{client_id}_tail.pth")
    
    def forward_pass(self, inputs):
        """Forward pass through the client's head model"""
        self.head.eval()  # Set to evaluation mode for forward pass
        with torch.no_grad():
            smashed_data = self.head(inputs)
        return smashed_data
    
    def compute_loss(self, tail_output, labels):
        """Compute loss using the client's tail model"""
        self.tail.train()  # Set to training mode
        loss = self.criterion(tail_output, labels)
        return loss
    
    def backward_pass(self, head_output, backbone_input_grad):
        """Backward pass to update the client's head and tail models"""
        self.head_optimizer.zero_grad()
        if backbone_input_grad is not None:
            head_output.backward(backbone_input_grad)
        else:
            raise Exception("Backbone input gradient is None")
        self.head_optimizer.step()
        
    def train_step(self, server, gated_fusion, epochs=1):
        """Complete training step with the server"""
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.head.train()  # Set head to training mode
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Client: Head forward pass
                head_output = self.head(inputs)

                backbone_input = head_output.detach().clone().requires_grad_(True)
                gate_input_z2 = head_output.detach().clone().requires_grad_(True)
                
                # Server: Process through backbone
                backbone_output = server.process(backbone_input)
                

                # tail_input = backbone_output.detach().clone().requires_grad_(True)
                gate_input_z1 = backbone_output.detach().clone().requires_grad_(True)


                gate_output = gated_fusion(gate_input_z1, gate_input_z2)

                tail_input = gate_output.detach().clone().requires_grad_(True)




                self.tail.train()

                tail_output = self.tail(tail_input)

            
                
                # Client: Compute loss with tail
                loss = self.compute_loss(tail_output, labels)

                self.tail_optimizer.zero_grad()
                loss.backward()
                self.tail_optimizer.step()


                # Gated Fusion: Backward pass
                gated_fusion.optimizer.zero_grad()
                gate_output.backward(tail_input.grad)
                gated_fusion.optimizer.step()

                
                
                # Server: Backward pass with gradient
                # server.backward(backbone_output, tail_input.grad)
                server.backward(backbone_output, gate_input_z1.grad)

                head_grad = gate_input_z2.grad + backbone_input.grad
                
                # Client: Complete backward pass
                self.backward_pass(head_output, head_grad)
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = tail_output.max(1)
                epoch_total += labels.size(0)
                epoch_correct += predicted.eq(labels).sum().item()
            
            # Calculate epoch statistics
            accuracy = 100 * epoch_correct / epoch_total
            avg_loss = epoch_loss / len(self.dataloader)
            
            print(f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            running_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
        
        # Overall statistics
        final_accuracy = 100 * correct / total
        final_avg_loss = running_loss / epochs
        
        return final_avg_loss, final_accuracy
    
    def save_models(self):
        """Save client models for next client to use"""
        self.head.save_state_dict(self.head_checkpoint)
        self.tail.save_state_dict(self.tail_checkpoint)
        print(f"Client {self.client_id} saved models to {self.checkpoint_dir}")
    
    def load_models(self, prev_client_id=None):
        """Load models from previous client or initialize if first client"""
        if prev_client_id is not None:
            prev_head_checkpoint = os.path.join(self.checkpoint_dir, f"client_{prev_client_id}_head.pth")
            prev_tail_checkpoint = os.path.join(self.checkpoint_dir, f"client_{prev_client_id}_tail.pth")
            
            if os.path.exists(prev_head_checkpoint) and os.path.exists(prev_tail_checkpoint):
                self.head.load_state_dict_from_path(prev_head_checkpoint)
                self.tail.load_state_dict_from_path(prev_tail_checkpoint)
                print(f"Client {self.client_id} loaded models from Client {prev_client_id}")
                return True
        return False

# Server implementation
class Server:
    def __init__(self, model_name, num_classes=10, device='cpu', checkpoint_dir="./checkpoints", cut_layer=1):
        self.device = device
        if model_name == 'resnet18':
            self.backbone = ResNet18Backbone(cut_layer=cut_layer).to(device)
        elif model_name == 'resnet50':
            self.backbone = ResNet50Backbone(cut_layer=cut_layer).to(device)
        elif model_name == 'vgg11':
            self.backbone = VGG11Backbone(cut_layer=cut_layer).to(device)
        elif model_name == 'vgg19':
            self.backbone = VGG19Backbone(cut_layer=cut_layer).to(device)
        elif model_name == 'densenet121':
            self.backbone = DenseNet121Backbone(cut_layer=cut_layer).to(device)
        elif model_name == 'vit_b16':
            self.backbone = ViTB16Backbone(cut_layer=cut_layer).to(device)
        else:
            raise Exception(f"Model {model_name} not supported")
            
        self.backbone_optimizer = optim.Adam(self.backbone.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.last_input = None
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint path
        self.backbone_checkpoint = os.path.join(checkpoint_dir, "server_backbone.pth")
    
    def process(self, smashed_data):
        """Process the smashed data through the backbone"""
        self.backbone.train()  # Set to training mode
        output = self.backbone(smashed_data)
        return output
    
    def backward(self, backbone_output, tail_input_grad):
        """Backward pass through the backbone"""
        self.backbone_optimizer.zero_grad()
        if tail_input_grad is not None:
            backbone_output.backward(tail_input_grad)
        else:
            raise Exception("Tail input gradient is None")
        self.backbone_optimizer.step()
    
    def save_model(self):
        """Save server model"""
        self.backbone.save_state_dict(self.backbone_checkpoint)
        print(f"Server saved backbone model to {self.backbone_checkpoint}")
    
    def load_model(self):
        """Load server model"""
        if os.path.exists(self.backbone_checkpoint):
            self.backbone.load_state_dict_from_path(self.backbone_checkpoint)
            print(f"Server loaded backbone model from {self.backbone_checkpoint}")
            return True
        return False




class Decoder(nn.Module):
    def __init__(self, model_name, cut_layer, input_size=(224, 224), device='cpu', checkpoint_dir="./checkpoints"):
        super().__init__()
        self.model_name = model_name
        self.cut_layer = cut_layer
        self.input_size = input_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, f"decoder_{model_name}_cut{cut_layer}.pth")
        
        # Define decoder architecture based on model and cut layer
        if model_name == 'resnet18':
            self._build_resnet18_decoder()
        elif model_name == 'resnet50':
            self._build_resnet50_decoder()
        elif model_name == 'vgg11':
            self._build_vgg11_decoder()
        elif model_name == 'vgg19':
            self._build_vgg19_decoder()
        elif model_name == 'densenet121':
            self._build_densenet121_decoder()
        elif model_name == 'vit_b16':
            self._build_vit_b16_decoder()
        else:
            raise Exception(f"Model {model_name} not supported")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        
        # Loss function for reconstruction
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = nn.L1Loss()
        
    def _build_resnet18_decoder(self):
        """Build decoder for ResNet18 based on cut layer"""
        if self.cut_layer == 0:
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
        elif self.cut_layer == 1:
            # Cut after layer1: expect [B, 64, 56, 56]
            self.decoder = nn.Sequential(
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
        elif self.cut_layer == 2:
            # Cut after layer2: expect [B, 128, 28, 28]
            self.decoder = nn.Sequential(
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
        elif self.cut_layer == 3:
            # Cut after layer3: expect [B, 256, 14, 14]
            self.decoder = nn.Sequential(
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
        elif self.cut_layer == 4:
            # Cut after layer4: expect [B, 512, 7, 7]
            self.decoder = nn.Sequential(
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
        else:
            raise Exception(f"Cut layer {self.cut_layer} not supported for ResNet18")
            
    def _build_resnet50_decoder(self):
        """Build decoder for ResNet50 based on cut layer"""
        if self.cut_layer == 0:
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
        elif self.cut_layer == 1:
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
        elif self.cut_layer == 2:
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
        elif self.cut_layer == 3:
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
        elif self.cut_layer == 4:
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
            raise Exception(f"Cut layer {self.cut_layer} not supported for ResNet50")
            
    def _build_vgg11_decoder(self):
        """Build decoder for VGG11 based on cut layer"""
        if self.cut_layer == 0:
            # Early cut in VGG11: expect [B, 64, 112, 112]
            self.decoder = nn.Sequential(
                # Upsample to 224x224
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 224, 224]
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(32, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif self.cut_layer == 1:
            # Mid cut in VGG11: expect [B, 128, 56, 56]
            self.decoder = nn.Sequential(
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 2:
            # Later cut in VGG11: expect [B, 256, 28, 28]
            self.decoder = nn.Sequential(
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 3:
            # Deep cut in VGG11: expect [B, 512, 14, 14]
            self.decoder = nn.Sequential(
                # Upsample to 28x28
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 28, 28]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        else:
            raise Exception(f"Cut layer {self.cut_layer} not supported for VGG11")
            
    def _build_vgg19_decoder(self):
        """Build decoder for VGG19 based on cut layer"""
        if self.cut_layer == 0:
            # After block 1: expect [B, 64, 112, 112]
            self.decoder = nn.Sequential(
                # Upsample to 224x224
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 224, 224]
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Final layer to RGB
                nn.Conv2d(32, 3, kernel_size=3, padding=1),  # [B, 3, 224, 224]
                nn.Tanh()
            )
        elif self.cut_layer == 1:
            # After block 2: expect [B, 128, 56, 56]
            self.decoder = nn.Sequential(
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 2:
            # After block 3: expect [B, 256, 28, 28]
            self.decoder = nn.Sequential(
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 3:
            # After block 4: expect [B, 512, 14, 14]
            self.decoder = nn.Sequential(
                # Upsample to 28x28
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 28, 28]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 4:
            # After block 5: expect [B, 512, 7, 7]
            self.decoder = nn.Sequential(
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
        else:
            raise Exception(f"Cut layer {self.cut_layer} not supported for VGG19")
            
    def _build_densenet121_decoder(self):
        """Build decoder for DenseNet121 based on cut layer"""
        if self.cut_layer == 0:
            # After initial conv layers: expect [B, 64, 56, 56]
            self.decoder = nn.Sequential(
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
        elif self.cut_layer == 1:
            # After dense block 1 + transition 1: expect [B, 128, 28, 28]
            self.decoder = nn.Sequential(
                # Upsample to 56x56
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 2:
            # After dense block 2 + transition 2: expect [B, 256, 14, 14]
            self.decoder = nn.Sequential(
                # Upsample to 28x28
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 28, 28]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 3:
            # After dense block 3 + transition 3: expect [B, 512, 7, 7]
            self.decoder = nn.Sequential(
                # Upsample to 14x14
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 14, 14]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Upsample to 28x28
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 28, 28]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        elif self.cut_layer == 4:
            # After dense block 4 + norm5: expect [B, 1024, 7, 7]
            self.decoder = nn.Sequential(
                # Upsample to 14x14
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 14, 14]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Upsample to 28x28
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 28, 28]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Upsample to 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 56, 56]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Upsample to 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 112, 112]
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
        else:
            raise Exception(f"Cut layer {self.cut_layer} not supported for DenseNet121")
            
    def _build_vit_b16_decoder(self):
        """Build decoder for ViT-B/16 based on cut layer"""
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
        if self.cut_layer == 0:
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
        elif self.cut_layer == 1:
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
        elif self.cut_layer == 2:
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
        elif self.cut_layer == 3:
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
        elif self.cut_layer == 4:
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
            raise Exception(f"Cut layer {self.cut_layer} not supported for ViT-B/16")
    
    def forward(self, backbone_output):
        """Forward pass through decoder"""
        if self.model_name == 'vit_b16':
            # For ViT-B/16: backbone_output is [B, 197, 768]
            # Extract patch tokens (exclude CLS token at index 0)
            patch_tokens = backbone_output[:, 1:, :]  # [B, 196, 768]
            
            # Flatten patch tokens
            patch_flat = patch_tokens.view(patch_tokens.size(0), -1)  # [B, 196*768]
            
            # Convert to spatial representation
            spatial_features = self.patch_to_spatial(patch_flat)  # [B, 768*14*14]
            spatial_features = spatial_features.view(spatial_features.size(0), 768, 14, 14)  # [B, 768, 14, 14]
            
            # Pass through decoder
            reconstructed = self.decoder(spatial_features)
        else:
            # For CNN models: backbone_output is [B, C, H, W]
            reconstructed = self.decoder(backbone_output)
        
        # Ensure output is in correct range [0, 1] for image reconstruction
        reconstructed = (reconstructed + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return reconstructed
    
    def compute_reconstruction_loss(self, reconstructed, original):
        """Compute reconstruction loss using MSE and perceptual loss"""
        # Normalize original to [0, 1] if needed
        if original.max() > 1.0:
            original = original / 255.0
        
        # MSE loss for pixel-wise reconstruction
        mse_loss = self.reconstruction_loss(reconstructed, original)
        
        # L1 loss for perceptual quality
        l1_loss = self.perceptual_loss(reconstructed, original)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * l1_loss
        
        return total_loss, mse_loss, l1_loss
    
    def train_step(self, backbone_output, original_images):
        """Single training step for decoder"""
        self.train()
        
        # Forward pass
        reconstructed = self.forward(backbone_output)
        
        # Compute loss
        total_loss, mse_loss, l1_loss = self.compute_reconstruction_loss(reconstructed, original_images)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mse_loss.item(), l1_loss.item(), reconstructed
    
    def evaluate(self, backbone_output, original_images):
        """Evaluate decoder without updating weights"""
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(backbone_output)
            total_loss, mse_loss, l1_loss = self.compute_reconstruction_loss(reconstructed, original_images)
            
        return total_loss.item(), mse_loss.item(), l1_loss.item(), reconstructed
    
    def save_model(self):
        """Save decoder model"""
        torch.save(self.state_dict(), self.checkpoint_path)
        print(f"Decoder saved to {self.checkpoint_path}")
    
    def load_model(self):
        """Load decoder model"""
        if os.path.exists(self.checkpoint_path):
            self.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Decoder loaded from {self.checkpoint_path}")
            return True
        return False
    
    def get_reconstruction_quality_metrics(self, reconstructed, original):
        """Calculate quality metrics for reconstruction"""
        # Ensure both images are in the same range
        if original.max() > 1.0:
            original = original / 255.0
        
        # Calculate PSNR
        mse = torch.mean((reconstructed - original) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Calculate SSIM (simplified version)
        def ssim_simple(img1, img2):
            mu1 = torch.mean(img1)
            mu2 = torch.mean(img2)
            sigma1 = torch.var(img1)
            sigma2 = torch.var(img2)
            sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return ssim
        
        ssim = ssim_simple(reconstructed, original)
        
        return psnr.item(), ssim.item()


def print_model_structures(model_name, cut_layer=1):
    """Print and compare the structure of original model and our split model"""
    if model_name == 'resnet18':
        original_model = models.resnet18(pretrained=False)
        head = ResNet18Head(in_channels=3, cut_layer=cut_layer)
        backbone = ResNet18Backbone(cut_layer=cut_layer)
        tail = ResNet18Tail(num_classes=1000)
    elif model_name == 'resnet50':
        original_model = models.resnet50(pretrained=False)
        head = ResNet50Head(in_channels=3, cut_layer=cut_layer)
        backbone = ResNet50Backbone(cut_layer=cut_layer)
        tail = ResNet50Tail(num_classes=1000)
    elif model_name == 'vgg11':
        original_model = models.vgg11(pretrained=False)
        head = VGG11Head(in_channels=3, cut_layer=cut_layer)
        backbone = VGG11Backbone(cut_layer=cut_layer)
        tail = VGG11Tail(num_classes=1000)
    elif model_name == 'vgg19':
        original_model = models.vgg19(pretrained=False)
        head = VGG19Head(in_channels=3, cut_layer=cut_layer)
        backbone = VGG19Backbone(cut_layer=cut_layer)
        tail = VGG19Tail(num_classes=1000)
    elif model_name == 'densenet121':
        original_model = models.densenet121(pretrained=False)
        head = DenseNet121Head(in_channels=3, cut_layer=cut_layer)
        backbone = DenseNet121Backbone(cut_layer=cut_layer)
        tail = DenseNet121Tail(num_classes=1000)
    elif model_name == 'vit_b16':
        original_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
        head = ViTB16Head(in_channels=3, cut_layer=cut_layer)
        backbone = ViTB16Backbone(cut_layer=cut_layer)
        tail = ViTB16Tail(num_classes=1000)
    else:
        raise Exception(f"Model {model_name} not supported")
    
    print("\n" + "="*50)
    print(f"ORIGINAL {model_name.upper()} STRUCTURE")
    print("="*50)
    print(original_model)
    
    print("\n" + "="*50)
    print(f"SPLIT {model_name.upper()} STRUCTURE (cut_layer={cut_layer})")
    print("="*50)
    print("\nHEAD:")
    print(head)
    print("\nBACKBONE:")
    print(backbone)
    print("\nTAIL:")
    print(tail)
    
    # Count parameters
    original_params = sum(p.numel() for p in original_model.parameters())
    split_params = sum(p.numel() for p in head.parameters()) + \
                  sum(p.numel() for p in backbone.parameters()) + \
                  sum(p.numel() for p in tail.parameters())
    
    print("\n" + "="*50)
    print("PARAMETER COMPARISON")
    print("="*50)
    print(f"Original {model_name} parameters: {original_params:,}")
    print(f"Split {model_name} parameters: {split_params:,}")
    print(f"Parameter difference: {abs(original_params - split_params):,}")


def train_federated_learning_with_decoder(clients, server, gated_fusion, decoder, 
                                         num_rounds=10, epochs_per_round=1, 
                                         decoder_training_epochs=5, save_reconstructions=False,
                                         reconstruction_save_dir="./reconstructions"):
    """
    Training function that integrates decoder for image reconstruction experiments
    
    Args:
        clients: List of Client objects
        server: Server object
        gated_fusion: GatedFusion object
        decoder: Decoder object
        num_rounds: Number of federated learning rounds
        epochs_per_round: Training epochs per round for each client
        decoder_training_epochs: Number of epochs to train decoder per round
        save_reconstructions: Whether to save reconstructed images
        reconstruction_save_dir: Directory to save reconstructed images
    """
    
    if save_reconstructions:
        os.makedirs(reconstruction_save_dir, exist_ok=True)
    
    # Load existing models if available
    server.load_model()
    gated_fusion.load_model()
    decoder.load_model()
    
    print(f"Starting Federated Learning with Image Reconstruction for {num_rounds} rounds")
    print("="*80)
    
    for round_num in range(num_rounds):
        print(f"\n--- ROUND {round_num + 1}/{num_rounds} ---")
        
        round_reconstruction_losses = []
        round_classification_losses = []
        round_accuracies = []
        
        # Train each client and collect reconstruction data
        for i, client in enumerate(clients):
            print(f"\nTraining Client {client.client_id}...")
            
            # Load previous client's models if not first client
            if i > 0:
                prev_client_id = clients[i-1].client_id
                client.load_models(prev_client_id)
            
            # Standard federated learning training
            avg_loss, accuracy = client.train_step(server, gated_fusion, epochs=epochs_per_round)
            round_classification_losses.append(avg_loss)
            round_accuracies.append(accuracy)
            
            # Save client models
            client.save_models()
            
            print(f"Client {client.client_id} - Classification Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Train decoder with reconstruction task
        print(f"\nTraining Decoder for {decoder_training_epochs} epochs...")
        decoder_losses = train_decoder_reconstruction(clients, server, decoder, 
                                                     training_epochs=decoder_training_epochs,
                                                     save_reconstructions=save_reconstructions,
                                                     save_dir=reconstruction_save_dir,
                                                     round_num=round_num)
        
        round_reconstruction_losses.extend(decoder_losses)
        
        # Save all models after each round
        server.save_model()
        gated_fusion.save_model()
        decoder.save_model()
        
        # Print round summary
        avg_classification_loss = sum(round_classification_losses) / len(round_classification_losses)
        avg_accuracy = sum(round_accuracies) / len(round_accuracies)
        avg_reconstruction_loss = sum(round_reconstruction_losses) / len(round_reconstruction_losses)
        
        print(f"\nRound {round_num + 1} Summary:")
        print(f"  Average Classification Loss: {avg_classification_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.2f}%")
        print(f"  Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
        print("-" * 60)
    
    print("\nFederated Learning with Image Reconstruction completed!")
    return round_classification_losses, round_reconstruction_losses, round_accuracies


def train_decoder_reconstruction(clients, server, decoder, training_epochs=5, 
                                save_reconstructions=False, save_dir="./reconstructions", 
                                round_num=0):
    """
    Train decoder to reconstruct original images from backbone outputs
    
    Args:
        clients: List of Client objects
        server: Server object
        decoder: Decoder object
        training_epochs: Number of epochs to train decoder
        save_reconstructions: Whether to save reconstructed images
        save_dir: Directory to save reconstructed images
        round_num: Current round number for naming saved files
    """
    
    decoder_losses = []
    
    for epoch in range(training_epochs):
        epoch_losses = []
        
        # Collect data from all clients for decoder training
        for client in clients:
            client_losses = []
            
            for batch_idx, (inputs, labels) in enumerate(client.dataloader):
                inputs = inputs.to(decoder.device)
                
                # Forward pass through client head
                client.head.eval()
                with torch.no_grad():
                    head_output = client.head(inputs)
                
                # Forward pass through server backbone
                server.backbone.eval()
                with torch.no_grad():
                    backbone_output = server.backbone(head_output)
                
                # Train decoder to reconstruct original images
                total_loss, mse_loss, l1_loss, reconstructed = decoder.train_step(backbone_output, inputs)
                
                client_losses.append(total_loss)
                epoch_losses.append(total_loss)
                
                # Save reconstructed images periodically
                if save_reconstructions and batch_idx == 0 and epoch == training_epochs - 1:
                    save_reconstruction_samples(inputs, reconstructed, save_dir, 
                                              round_num, client.client_id, epoch)
                
                # Print progress for first batch of each client
                if batch_idx == 0:
                    psnr, ssim = decoder.get_reconstruction_quality_metrics(reconstructed, inputs)
                    print(f"  Client {client.client_id}, Epoch {epoch+1}/{training_epochs}, "
                          f"Batch {batch_idx+1}: Loss={total_loss:.4f}, "
                          f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
            
            # Print client summary
            avg_client_loss = sum(client_losses) / len(client_losses) if client_losses else 0
            print(f"  Client {client.client_id} average reconstruction loss: {avg_client_loss:.4f}")
        
        # Print epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"Decoder Epoch {epoch+1}/{training_epochs} average loss: {avg_epoch_loss:.4f}")
        decoder_losses.append(avg_epoch_loss)
    
    return decoder_losses


def save_reconstruction_samples(original_images, reconstructed_images, save_dir, 
                               round_num, client_id, epoch, num_samples=8):
    """
    Save sample reconstructed images for visual inspection
    
    Args:
        original_images: Original input images
        reconstructed_images: Reconstructed images from decoder
        save_dir: Directory to save images
        round_num: Current round number
        client_id: Client ID
        epoch: Current epoch
        num_samples: Number of samples to save
    """
    
    # Create subdirectory for this round
    round_dir = os.path.join(save_dir, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)
    
    # Select samples to save
    num_samples = min(num_samples, original_images.size(0))
    
    # Save original images
    original_grid = vutils.make_grid(original_images[:num_samples], nrow=4, normalize=True)
    original_path = os.path.join(round_dir, f"original_client_{client_id}_epoch_{epoch}.png")
    vutils.save_image(original_grid, original_path)
    
    # Save reconstructed images
    reconstructed_grid = vutils.make_grid(reconstructed_images[:num_samples], nrow=4, normalize=True)
    reconstructed_path = os.path.join(round_dir, f"reconstructed_client_{client_id}_epoch_{epoch}.png")
    vutils.save_image(reconstructed_grid, reconstructed_path)
    
    print(f"    Saved reconstruction samples to {round_dir}")


def create_decoder_for_experiment(model_name, cut_layer, device='cpu', checkpoint_dir="./checkpoints"):
    """
    Factory function to create decoder for experiments
    
    Args:
        model_name: Name of the model ('resnet18' or 'vgg11')
        cut_layer: Layer where the model is cut
        device: Device to run on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Decoder object ready for training
    """
    
    decoder = Decoder(
        model_name=model_name,
        cut_layer=cut_layer,
        input_size=(224, 224),
        device=device,
        checkpoint_dir=checkpoint_dir
    ).to(device)
    
    print(f"Created {model_name} decoder for cut_layer={cut_layer}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    return decoder


def evaluate_reconstruction_quality(decoder, test_dataloader, server, clients, device='cpu'):
    """
    Evaluate reconstruction quality on test dataset
    
    Args:
        decoder: Trained decoder
        test_dataloader: DataLoader for test data
        server: Server object
        clients: List of clients (use first client's head)
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    decoder.eval()
    server.backbone.eval()
    clients[0].head.eval()  # Use first client's head for evaluation
    
    total_mse_loss = 0
    total_l1_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            
            # Forward pass through head and backbone
            head_output = clients[0].head(inputs)
            backbone_output = server.backbone(head_output)
            
            # Reconstruct images
            reconstructed = decoder(backbone_output)
            
            # Calculate losses
            total_loss, mse_loss, l1_loss = decoder.compute_reconstruction_loss(reconstructed, inputs)
            
            # Calculate quality metrics
            psnr, ssim = decoder.get_reconstruction_quality_metrics(reconstructed, inputs)
            
            total_mse_loss += mse_loss
            total_l1_loss += l1_loss
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
    
    # Calculate averages
    avg_mse = total_mse_loss / num_batches
    avg_l1 = total_l1_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    results = {
        'mse_loss': avg_mse,
        'l1_loss': avg_l1,
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
    
    print("Reconstruction Quality Evaluation:")
    print(f"  MSE Loss: {avg_mse:.6f}")
    print(f"  L1 Loss: {avg_l1:.6f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    return results
