from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import torchvision.models as models
import torchvision.utils as vutils
import timm
from models.ResNet18 import ResNet18Head, ResNet18Tail, ResNet18Backbone, ResNet18Decoder
from models.ResNet50 import ResNet50Head, ResNet50Tail, ResNet50Backbone, ResNet50Decoder
from models.VGG11 import VGG11Head, VGG11Tail, VGG11Backbone, VGG11Decoder
from models.VGG19 import VGG19Head, VGG19Tail, VGG19Backbone, VGG19Decoder
from models.DenseNet121 import DenseNet121Head, DenseNet121Tail, DenseNet121Backbone, DenseNet121Decoder
from models.ViT_B16 import ViTB16Head, ViTB16Tail, ViTB16Backbone, ViTB16Decoder




class HoneyPot(nn.Module):
    """Module for projecting z2 (head output) to match z1 dimension"""
    def __init__(self, model_name, cut_layer, z1_dim, z2_channels):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'vit_b16':
            # For ViT, z2 is [B, seq_len, embed_dim], we'll use all tokens except CLS
            # seq_len = 197 (196 patches + 1 CLS token)
            self.z2_proj = nn.Sequential(
                nn.Linear(z2_channels * 196, z1_dim * 2),  # 196 patch tokens
                nn.ReLU(),
                nn.Linear(z1_dim * 2, z1_dim)
            )
            self.pool_z2 = None
        else:
            # For CNN models, z2 is [B, C, H, W]
            self.spatial_size = 2  # or try 2, 4, 7, depending on how much detail you want
            self.pool_z2 = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))  # [B, C, spatial_size, spatial_size]
            self.z2_proj = nn.Sequential(
                nn.Linear(z2_channels * self.spatial_size * self.spatial_size, z1_dim * 2),
                nn.ReLU(),
                nn.Linear(z1_dim * 2, z1_dim)
            )
    
    def forward(self, z2):
        if self.model_name == 'vit_b16':
            # For ViT: z2 is [B, seq_len, embed_dim] where seq_len = 197
            # Extract patch tokens (exclude CLS token at index 0)
            patch_tokens = z2[:, 1:, :]  # [B, 196, 768]
            z2_flat = patch_tokens.view(patch_tokens.size(0), -1)  # [B, 196*768]
            z2_proj = self.z2_proj(z2_flat)  # [B, z1_dim]
        else:
            # For CNN models: z2 is [B, C, H, W]
            z2_pooled = self.pool_z2(z2)  # [B, C, spatial_size, spatial_size]
            # print(f"z2_pooled.shape: {z2_pooled.shape}")
            z2_flat = z2_pooled.view(z2_pooled.size(0), -1)  # [B, C*spatial_size*spatial_size]
            # print(f"z2_flat.shape: {z2_flat.shape}")
            z2_proj = self.z2_proj(z2_flat)  # [B, z1_dim]
            # print(f"z2_proj.shape: {z2_proj.shape}")
        return z2_proj


class Gate(nn.Module):
    """Module for gating mechanism that fuses z1 and projected z2"""
    def __init__(self, z1_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(z1_dim + z1_dim, z1_dim * 2),  # concat of z1 and projected z2
            nn.SiLU(),
            nn.Linear(z1_dim * 2, z1_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z1, z2_proj):
        z_cat = torch.cat([z1, z2_proj], dim=1)  # [B, z1_dim + z1_dim]
        # print(f"z_cat.shape: {z_cat.shape}")
        g = self.gate(z_cat)  # [B, z1_dim]
        # print(f"g.shape: {g.shape}")  
        fused = g * z1 + (1 - g) * z2_proj  # weighted fusion
        # print(f"fused.shape: {fused.shape}")
        return fused


class GatedFusion(nn.Module):
    def __init__(self, model_name, cut_layer, checkpoint_dir="./checkpoints", 
                 honey_pot_lr=None, gate_lr=None):
        super().__init__()
        
        self.model_name = model_name
        self.cut_layer = cut_layer

        # Determine z1_dim and z2_channels based on model and cut layer
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

        # Create HoneyPot and Gate modules
        self.honey_pot = HoneyPot(model_name, cut_layer, z1_dim, z2_channels)
        self.gate = Gate(z1_dim)

        # Set default learning rates based on model if not provided
        if honey_pot_lr is None or gate_lr is None:
            if model_name in ['resnet18', 'resnet50', 'densenet121']:
                default_lr = 0.001
                default_optimizer_class = optim.Adam
                default_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 1e-4}
            elif model_name == 'vit_b16':
                default_lr = 3e-4
                default_optimizer_class = optim.AdamW
                default_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.05}
            elif model_name in ['vgg11', 'vgg19']:
                default_lr = 0.001
                default_optimizer_class = optim.Adam
                default_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 1e-4}
            else:
                default_lr = 0.001
                default_optimizer_class = optim.Adam
                default_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 1e-4}
            
            if honey_pot_lr is None:
                honey_pot_lr = default_lr
            if gate_lr is None:
                gate_lr = default_lr
        else:
            # Use Adam as default if custom learning rates are provided
            default_optimizer_class = optim.Adam
            default_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 1e-4}

        # Create separate optimizers for HoneyPot and Gate
        if model_name == 'vit_b16':
            self.honey_pot_optimizer = optim.AdamW(self.honey_pot.parameters(), lr=honey_pot_lr, 
                                                   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
            self.gate_optimizer = optim.AdamW(self.gate.parameters(), lr=gate_lr, 
                                             betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        else:
            self.honey_pot_optimizer = default_optimizer_class(self.honey_pot.parameters(), 
                                                               lr=honey_pot_lr, **default_kwargs)
            self.gate_optimizer = default_optimizer_class(self.gate.parameters(), 
                                                         lr=gate_lr, **default_kwargs)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.honey_pot_checkpoint = os.path.join(checkpoint_dir, "honey_pot.pth")
        self.gate_checkpoint = os.path.join(checkpoint_dir, "gate.pth")



    def forward(self, z1, z2):
        # HoneyPot: Project z2 to match z1 dimension
        z2_proj = self.honey_pot(z2)  # [B, z1_dim]
        
        # Gate: Fuse z1 and z2_proj
        fused = self.gate(z1, z2_proj)  # [B, z1_dim]
        
        return fused  # [B, z1_dim]



    def save_model(self):
        """Save both HoneyPot and Gate modules"""
        torch.save(self.honey_pot.state_dict(), self.honey_pot_checkpoint)
        torch.save(self.gate.state_dict(), self.gate_checkpoint)
        print(f"HoneyPot saved to {self.honey_pot_checkpoint}")
        print(f"Gate saved to {self.gate_checkpoint}")
    
    def load_model(self):
        """Load both HoneyPot and Gate modules"""
        loaded = False
        if os.path.exists(self.honey_pot_checkpoint):
            self.honey_pot.load_state_dict(torch.load(self.honey_pot_checkpoint))
            print(f"HoneyPot loaded from {self.honey_pot_checkpoint}")
            loaded = True
        else:
            loaded = False
        if os.path.exists(self.gate_checkpoint):
            self.gate.load_state_dict(torch.load(self.gate_checkpoint))
            print(f"Gate loaded from {self.gate_checkpoint}")
            loaded = True
        else:
            loaded = False
        return loaded
    
    def zero_grad(self):
        """Zero gradients for both optimizers"""
        self.honey_pot_optimizer.zero_grad()
        self.gate_optimizer.zero_grad()
    
    def step(self):
        """Step both optimizers"""
        self.honey_pot_optimizer.step()
        self.gate_optimizer.step()



# Client implementation
class Client:
    def __init__(self, args, model_name, is_malicious, client_id, dataset, batch_size=32, num_classes=10, device='cpu', 
                 checkpoint_dir="./checkpoints", cut_layer=1):
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.device = device
        self.args = args
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers)
        
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
        
        # Model-specific optimizers
        if model_name in ['resnet18', 'resnet50', 'densenet121']:
            # Adam for ResNet and DenseNet
            self.head_optimizer = optim.Adam(self.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
            self.tail_optimizer = optim.Adam(self.tail.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        elif model_name == 'vit_b16':
            # AdamW for Vision Transformer
            self.head_optimizer = optim.AdamW(self.head.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
            self.tail_optimizer = optim.AdamW(self.tail.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        elif model_name in ['vgg11', 'vgg19']:
            # SGD with momentum for VGG models
            self.head_optimizer = optim.SGD(self.head.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            self.tail_optimizer = optim.SGD(self.tail.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            # Default to Adam for unknown models
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

                # print(f"inputs.shape: {inputs.shape}")
                # print(f"labels.shape: {labels.shape}")
                # print(f"inputs.dtype: {inputs.dtype}")
                # print(f"labels.dtype: {labels.dtype}")
                
                # Client: Head forward pass
                head_output = self.head(inputs)
                # print(f"head_output.shape: {head_output.shape}")

                backbone_input = head_output.detach().clone().requires_grad_(True)
                gate_input_z2 = head_output.detach().clone().requires_grad_(True)
                # print(f"backbone_input.shape: {backbone_input.shape}")
                # print(f"gate_input_z2.shape: {gate_input_z2.shape}")
                
                # Server: Process through backbone
                backbone_output = server.process(backbone_input)                
                # print(f"backbone_output.shape: {backbone_output.shape}")

                # tail_input = backbone_output.detach().clone().requires_grad_(True)
                gate_input_z1 = backbone_output.detach().clone().requires_grad_(True)
                # print(f"gate_input_z1.shape: {gate_input_z1.shape}")

                gate_output = gated_fusion(gate_input_z1, gate_input_z2)
                # print(f"gate_output.shape: {gate_output.shape}")
                # exit()

                tail_input = gate_output.detach().clone().requires_grad_(True)




                self.tail.train()

                tail_output = self.tail(tail_input)

            
                
                # Client: Compute loss with tail
                loss = self.compute_loss(tail_output, labels)

                self.tail_optimizer.zero_grad()
                loss.backward()
                self.tail_optimizer.step()


                # Gated Fusion: Backward pass (both HoneyPot and Gate)
                gated_fusion.zero_grad()
                gate_output.backward(tail_input.grad)
                gated_fusion.step()

                
                
                # Server: Backward pass with gradient
                # server.backward(backbone_output, tail_input.grad)
                server.backward(backbone_output, gate_input_z1.grad)
                
                head_grad = gate_input_z2.grad
                if backbone_input.grad is not None:
                    head_grad = head_grad + backbone_input.grad
                
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
    

    def train_step_attack_only(self, server, epochs=1):
        """Complete just attack training step with the server"""
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
                
                # Server: Process through backbone
                backbone_output = server.process(backbone_input)
                

                tail_input = backbone_output.detach().clone().requires_grad_(True)


                self.tail.train()

                tail_output = self.tail(tail_input)

            
                
                # Client: Compute loss with tail
                loss = self.compute_loss(tail_output, labels)

                self.tail_optimizer.zero_grad()
                loss.backward()
                self.tail_optimizer.step()
                
                
                # Server: Backward pass with gradient
                server.backward(backbone_output, tail_input.grad)
                
                # Client: Complete backward pass
                self.backward_pass(head_output, backbone_input.grad)
                
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




class Surrogate_Head:
    def __init__(self, args, model_name, dataset, batch_size=32, num_classes=10, device='cpu', 
                 checkpoint_dir="./checkpoints", cut_layer=1):
        self.device = device
        self.args = args
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers)
        
        if model_name == 'resnet18':
            self.head = ResNet18Head(in_channels=3, cut_layer=cut_layer).to(device)
        elif model_name == 'resnet50':
            self.head = ResNet50Head(in_channels=3, cut_layer=cut_layer).to(device)
        elif model_name == 'vgg11':
            self.head = VGG11Head(in_channels=3, cut_layer=cut_layer).to(device)
        elif model_name == 'vgg19':
            self.head = VGG19Head(in_channels=3, cut_layer=cut_layer).to(device)
        elif model_name == 'densenet121':
            self.head = DenseNet121Head(in_channels=3, cut_layer=cut_layer).to(device)
        elif model_name == 'vit_b16':
            self.head = ViTB16Head(in_channels=3, cut_layer=cut_layer).to(device)
        else:
            raise Exception(f"Model {model_name} not supported")
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Model-specific optimizers
        if model_name in ['resnet18', 'resnet50', 'densenet121']:
            # Adam for ResNet and DenseNet
            self.head_optimizer = optim.Adam(self.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        elif model_name == 'vit_b16':
            # AdamW for Vision Transformer
            self.head_optimizer = optim.AdamW(self.head.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        elif model_name in ['vgg11', 'vgg19']:
            # SGD with momentum for VGG models
            self.head_optimizer = optim.SGD(self.head.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            # Default to Adam for unknown models
            self.head_optimizer = optim.Adam(self.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint paths
        self.head_checkpoint = os.path.join(checkpoint_dir, f"surrogate_head.pth")
    
    def forward_pass(self, inputs):
        """Forward pass through the surrogate head model"""
        self.head.eval()  # Set to evaluation mode for forward pass
        with torch.no_grad():
            smashed_data = self.head(inputs)
        return smashed_data
    
    def backward_pass(self, head_output, backbone_input_grad):
        """Backward pass to update the client's head and tail models"""
        self.head_optimizer.zero_grad()
        if backbone_input_grad is not None:
            head_output.backward(backbone_input_grad)
        else:
            raise Exception("Backbone input gradient is None")
        self.head_optimizer.step()
        

    def save_model(self):
        """Save surrogate head model"""
        self.head.save_state_dict(self.head_checkpoint)
        print(f"Surrogate head saved to {self.checkpoint_dir}")
    
    def load_model(self, head_checkpoint=None):
        """Load models from previous client or initialize if first client"""
        if head_checkpoint is not None:
            load_path = head_checkpoint
        else:
            load_path = self.head_checkpoint
            
        if os.path.exists(load_path):
            self.head.load_state_dict_from_path(load_path)
            print(f"Surrogate head loaded from {load_path}")
            return True
        else:
            return False

    def train_step(self, server, decoder, epochs=1):
        """Complete training step with the server and decoder for reconstruction"""
        running_loss = 0.0
        
        self.head.train()  # Set head to training mode
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # print(f"inputs.shape: {inputs.shape}")
                # print(f"labels.shape: {labels.shape}")
                # print(f"inputs.dtype: {inputs.dtype}")
                # print(f"labels.dtype: {labels.dtype}")
                
                # Client: Head forward pass
                head_output = self.head(inputs)
                # print(f"head_output.shape: {head_output.shape}")

                backbone_input = head_output.detach().clone().requires_grad_(True)
                # print(f"backbone_input.shape: {backbone_input.shape}")

                
                # Server: Process through backbone
                backbone_output = server.process(backbone_input)                
                # print(f"backbone_output.shape: {backbone_output.shape}")

                decoder_input = backbone_output.detach().clone().requires_grad_(True)
                # print(f"decoder_input.shape: {decoder_input.shape}")

                decoder_output = decoder(decoder_input)
                # print(f"decoder_output.shape: {decoder_output.shape}")
            
                
                # compute decoder loss
                loss, mse_loss, tv_loss = decoder.compute_reconstruction_loss(decoder_output, inputs)

                decoder.optimizer.zero_grad()
                loss.backward()
                decoder.optimizer.step()

                
                
                # Server: Backward pass with gradient
                server.backward(backbone_output, decoder_input.grad)
                
                
                # Surrogate Head: Complete backward pass
                self.backward_pass(head_output, backbone_input.grad)
                
                # Statistics
                epoch_loss += loss.item()
   
            
            # Calculate epoch statistics
            avg_loss = epoch_loss / len(self.dataloader)
            
            print(f"Surrogate Head, Epoch {epoch+1}/{epochs}, "
                  f"Reconstruction Loss: {avg_loss:.4f}")
            
            running_loss += avg_loss
        
        # Overall statistics
        final_avg_loss = running_loss / epochs
        
        return final_avg_loss
    





# Server implementation
class Server:
    def __init__(self, model_name, num_classes=10, device='cpu', checkpoint_dir="./checkpoints", cut_layer=1):
        self.model_name = model_name
        self.device = device
        self.freeze_backbone = False
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
            
        # Model-specific optimizers for backbone
        if model_name in ['resnet18', 'resnet50', 'densenet121']:
            # Adam for ResNet and DenseNet
            self.backbone_optimizer = optim.Adam(self.backbone.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        elif model_name == 'vit_b16':
            # AdamW for Vision Transformer
            self.backbone_optimizer = optim.AdamW(self.backbone.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        elif model_name in ['vgg11', 'vgg19']:
            # SGD with momentum for VGG models
            self.backbone_optimizer = optim.SGD(self.backbone.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            # Default to Adam for unknown models
            self.backbone_optimizer = optim.Adam(self.backbone.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.last_input = None
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint path
        self.backbone_checkpoint = os.path.join(checkpoint_dir, "server_backbone.pth")
    
    def set_backbone_freeze(self, enable: bool):
        """Enable or disable backbone freezing"""
        self.freeze_backbone = enable
        status = "FROZEN" if enable else "ACTIVE"
        print(f"Server backbone is now: {status}")

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
        if not self.freeze_backbone:
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
    """Wrapper class for model-specific decoders with training and evaluation utilities"""
    def __init__(self, args, model_name, cut_layer, input_size=(224, 224), device='cpu', checkpoint_dir="./checkpoints", 
                 normalization_mean=None, normalization_std=None):
        super().__init__()
        self.args = args
        self.model_name = model_name
        self.cut_layer = cut_layer
        self.input_size = input_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, f"decoder_{model_name}_cut{cut_layer}.pth")
        
        # Store normalization parameters for denormalization
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        
        # Instantiate the appropriate model-specific decoder
        if model_name == 'resnet18':
            self.decoder = ResNet18Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        elif model_name == 'resnet50':
            self.decoder = ResNet50Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        elif model_name == 'vgg11':
            self.decoder = VGG11Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        elif model_name == 'vgg19':
            self.decoder = VGG19Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        elif model_name == 'densenet121':
            self.decoder = DenseNet121Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        elif model_name == 'vit_b16':
            self.decoder = ViTB16Decoder(cut_layer=cut_layer, input_size=input_size).to(device)
        else:
            raise Exception(f"Model {model_name} not supported")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
                
        # Loss function for reconstruction
        self.reconstruction_loss = nn.MSELoss()

        self.l1_loss = nn.L1Loss()
        
        # Total variation loss weight
        self.tv_weight = 0.001  # Small weight to avoid over-smoothing
        self.l1_weight = 0.001  # Small weight to avoid over-smoothing
    
    def forward(self, backbone_output):
        """Forward pass through decoder"""
        # Delegate to the model-specific decoder
        return self.decoder(backbone_output)
    
    def total_variation_loss(self, img):
        """Compute total variation loss for smoothness regularization
        
        Args:
            img: Image tensor [B, C, H, W]
        
        Returns:
            Total variation loss (scalar)
        """
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)
    
    def compute_reconstruction_loss(self, reconstructed, original):
        """Compute reconstruction loss using MSE and Total Variation loss
        
        Args:
            reconstructed: Reconstructed images in [0, 1] range from decoder
            original: Original images in normalized range (e.g., ImageNet normalization)
        
        Returns:
            total_loss, mse_loss, tv_loss
        """
        # Denormalize original images from normalized range to [0, 1]
        if self.normalization_mean is not None and self.normalization_std is not None:
            mean = torch.tensor(self.normalization_mean).view(1, 3, 1, 1).to(original.device)
            std = torch.tensor(self.normalization_std).view(1, 3, 1, 1).to(original.device)
            original_denorm = original * std + mean  # Denormalize to [0, 1]
        else:
            # Fallback: if max > 1, assume it's in [0, 255] range
            if original.max() > 1.0:
                original_denorm = original / 255.0
            else:
                raise Exception("Normalization parameters are not set and the original images are not in [0, 255] range")
        
        # MSE loss for pixel-wise reconstruction fidelity
        mse_loss = self.reconstruction_loss(reconstructed, original_denorm)

        l1_loss = self.l1_loss(reconstructed, original_denorm)
        
        # Total variation loss for smoothness regularization
        tv_loss = self.total_variation_loss(reconstructed)
        
        # Combined loss: MSE for fidelity + TV for smoothness
        total_loss = mse_loss + self.tv_weight * tv_loss + self.l1_weight * l1_loss
        
        return total_loss, mse_loss, tv_loss
    
    def train_step(self, backbone_output, original_images):
        """Single training step for decoder"""
        self.train()
        
        # Forward pass
        reconstructed = self.forward(backbone_output)
        
        # Compute loss
        total_loss, mse_loss, tv_loss = self.compute_reconstruction_loss(reconstructed, original_images)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mse_loss.item(), tv_loss.item(), reconstructed
    
    def evaluate(self, backbone_output, original_images):
        """Evaluate decoder without updating weights"""
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(backbone_output)
            total_loss, mse_loss, tv_loss = self.compute_reconstruction_loss(reconstructed, original_images)
            
        return total_loss.item(), mse_loss.item(), tv_loss.item(), reconstructed
    
    
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
        """Calculate quality metrics for reconstruction
        
        Args:
            reconstructed: Reconstructed images in [0, 1] range from decoder
            original: Original images in normalized range (e.g., ImageNet normalization)
        
        Returns:
            psnr, ssim
        """
        # Denormalize original images from normalized range to [0, 1]
        if self.normalization_mean is not None and self.normalization_std is not None:
            mean = torch.tensor(self.normalization_mean).view(1, 3, 1, 1).to(original.device)
            std = torch.tensor(self.normalization_std).view(1, 3, 1, 1).to(original.device)
            original_denorm = original * std + mean  # Denormalize to [0, 1]
        else:
            # Fallback: if max > 1, assume it's in [0, 255] range
            if original.max() > 1.0:
                original_denorm = original / 255.0
            else:
                original_denorm = original
        
        # Calculate PSNR
        mse = torch.mean((reconstructed - original_denorm) ** 2)
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
        
        ssim = ssim_simple(reconstructed, original_denorm)
        
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