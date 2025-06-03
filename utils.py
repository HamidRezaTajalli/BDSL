from pathlib import Path
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time
import copy
import os
import argparse
from models.ResNet18 import ResNet18Head, ResNet18Tail, ResNet18Backbone
from models.VGG11 import VGG11Head, VGG11Tail, VGG11Backbone
import torchvision.models as models
import math
import numpy as np

from attacks.badnet import create_poisoned_set


def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning Training')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg11'],
                      help='Model architecture to use (default: resnet18)')
    parser.add_argument('--num_clients', type=int, default=10,
                      help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=10,
                      help='Number of training rounds (default: 10)')
    parser.add_argument('--epochs_per_client', type=int, default=1,
                      help='Number of epochs per client per round (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--cut_layer', type=int, default=1,
                      help='Cut layer for model splitting (default: 1)')
    parser.add_argument('--checkpoint_dir', type=str, default='./split_learning_checkpoints',
                      help='Directory to save checkpoints (default: ./split_learning_checkpoints)')
    parser.add_argument('--poisoning_rate', type=float, default=0.9,
                      help='Poisoning rate for malicious clients (default: 0.4)')
    return parser.parse_args()



class GatedFusion(nn.Module):
    def __init__(self, model_name, cut_layer, checkpoint_dir="./checkpoints"):
        super().__init__()

        if model_name == 'resnet18':
            z1_dim = 512
            if cut_layer == 1:
                z2_channels = 64
            elif cut_layer == 2:
                z2_channels = 128
            elif cut_layer == 3:
                z2_channels = 256
            elif cut_layer == 4:
                z2_channels = 512
        elif model_name == 'vgg11':
            z1_dim, z2_channels = None, None
        else:
            raise Exception(f"Model {model_name} not supported")

        self.spatial_size = 4  # or try 2, 4, 7, depending on how much detail you want
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
        z2_pooled = self.pool_z2(z2)  # [B, C, 4, 4]
        z2_flat = z2_pooled.view(z2_pooled.size(0), -1)  # [B, C*4*4]
        z2_proj = self.z2_proj(z2_flat)  # [B, z1_dim]

        z_cat = torch.cat([z1, z2_proj], dim=1)  # [B, 1024]
        g = self.gate(z_cat)  # [B, 512]
        
        fused = g * z1 + (1 - g) * z2_proj  # weighted fusion
        return fused  # [B, 512]



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
        elif model_name == 'vgg11':
            self.head = VGG11Head(in_channels=3, cut_layer=cut_layer).to(device)
            self.tail = VGG11Tail(num_classes=num_classes).to(device)
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
        elif model_name == 'vgg11':
            self.backbone = VGG11Backbone(cut_layer=cut_layer).to(device)
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


# Multi-client round-robin split learning system
class RoundRobinSplitLearningSystem:
    def __init__(self, server, clients, gated_fusion, checkpoint_dir="./checkpoints"):
        self.server = server
        self.clients = clients
        self.gated_fusion = gated_fusion
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_round(self, epochs_per_client=1):
        """Train all clients in a round-robin fashion for one round"""
        print(f"Starting round-robin training with {len(self.clients)} clients...")
        
        # First client initializes or loads from previous round
        self.server.load_model()  # Load server model if available
        self.gated_fusion.load_model()  # Load gated fusion model if available

        for i, client in enumerate(self.clients):
            print(f"\n--- Training Client {client.client_id} ---")
            start_time = time.time()
            
            # Load model parameters from previous client or from last round
            if i == 0:
                # First client in the round - try to load from the last client of previous round
                prev_client_id = self.clients[-1].client_id
                client.load_models(prev_client_id)
            else:
                # Load from the previous client in this round
                prev_client_id = self.clients[i-1].client_id
                client.load_models(prev_client_id)
            
            # Train client
            loss, accuracy = client.train_step(self.server, self.gated_fusion, epochs=epochs_per_client)
            
            # Save client models for next client
            client.save_models()
            
            # Save server model after each client (optional, could also save only at the end of the round)
            self.server.save_model()
            
            # Save gated fusion model after each client
            self.gated_fusion.save_model()
            
            end_time = time.time()
            print(f"Client {client.client_id} completed training in {end_time - start_time:.2f}s")
            print(f"Final Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("\n--- Round completed ---")
    
    def train_multiple_rounds(self, num_rounds=5, epochs_per_client=1):
        """Run multiple rounds of training"""
        for round_num in range(num_rounds):
            print(f"\n===== Starting Round {round_num+1}/{num_rounds} =====")
            self.train_round(epochs_per_client)
            print(f"===== Completed Round {round_num+1}/{num_rounds} =====\n")




# Additional utility functions

def evaluate_model(clients, server, test_dataset, device):
    """Evaluate the model on test data using the latest client model"""
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Use the last client's models for evaluation
    last_client = clients[-1]
    head = last_client.head
    tail = last_client.tail
    backbone = server.backbone
    
    head.eval()
    backbone.eval()
    tail.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass through the head
            smashed_data = head(inputs)
            
            # Forward pass through the backbone
            backbone_output = backbone(smashed_data)
            
            # Forward pass through the tail
            outputs = tail(backbone_output)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def print_model_structures(model_name, cut_layer=1):
    """Print and compare the structure of original model and our split model"""
    if model_name == 'resnet18':
        original_model = models.resnet18(pretrained=False)
        head = ResNet18Head(in_channels=3, cut_layer=cut_layer)
        backbone = ResNet18Backbone(cut_layer=cut_layer)
        tail = ResNet18Tail(num_classes=1000)
    elif model_name == 'vgg11':
        original_model = models.vgg11(pretrained=False)
        head = VGG11Head(in_channels=3, cut_layer=cut_layer)
        backbone = VGG11Backbone(cut_layer=cut_layer)
        tail = VGG11Tail(num_classes=1000)
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

# Example usage with CIFAR-10
if __name__ == "__main__":
    args = parse_args()


    print("="*50)
    print("ARGUMENTS:")
    print("="*50)
    print(args)
    print("="*50)
    
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(224),  # Both ResNet and VGG require 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.model}_{args.cut_layer}_{args.num_clients}_{args.num_rounds}_{args.epochs_per_client}_{args.poisoning_rate}"

    checkpoint_dir = checkpoint_dir / suffix
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    
    # Create server
    server = Server(
        model_name=args.model,
        num_classes=10,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        cut_layer=args.cut_layer
    )

    # Create gated fusion
    gated_fusion = GatedFusion(model_name=args.model, cut_layer=args.cut_layer, checkpoint_dir=args.checkpoint_dir).to(device)
    
    # Create multiple clients with different data partitions
    clients = []
    malicious_clients = []
    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size)
    split_size = dataset_size // args.num_clients

    num_innocent_clients = args.num_clients // 2 + 1
    innocent_indices = torch.randperm(args.num_clients)[:num_innocent_clients]
    
    for j in range(args.num_clients):
        is_malicious = j not in innocent_indices
        malicious_clients.append(is_malicious)
    
    for i in range(args.num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < args.num_clients - 1 else dataset_size
        subset_indices = indices[start_idx:end_idx]
        subset = torch.utils.data.Subset(train_dataset, subset_indices)
        

        if i in malicious_clients:
            poisoned_subset, poisoned_indices = create_poisoned_set(trigger_size=0.08, poisoning_rate=args.poisoning_rate, target_label=0, subset=subset)
            subset = poisoned_subset
        
        clients.append(Client(
            model_name=args.model,
            client_id=i,
            is_malicious=malicious_clients[i],
            dataset=subset,
            batch_size=args.batch_size,
            num_classes=10,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            cut_layer=args.cut_layer
        ))
    
    # Create round-robin split learning system
    system = RoundRobinSplitLearningSystem(server, clients, gated_fusion, checkpoint_dir=args.checkpoint_dir)
    
    # Train for multiple rounds
    system.train_multiple_rounds(num_rounds=args.num_rounds, epochs_per_client=args.epochs_per_client)

    # Evaluate on test set
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_accuracy = evaluate_model(clients, server, test_dataset, device)
    print(f"\nFinal test accuracy after training: {test_accuracy:.2f}%")

    # Create a Subset from test_dataset with all its indices
    test_dataset_size = len(test_dataset)
    test_indices = list(range(test_dataset_size))
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create a poisoned dataset with full poisoning rate
    full_poisoning_rate = 1.0
    target_label = 0  # Assuming target label for ASR is 0
    poisoned_test_subset, _ = create_poisoned_set(
        trigger_size=0.08, 
        poisoning_rate=full_poisoning_rate, 
        target_label=target_label, 
        subset=test_subset
    )
    
    # Evaluate the model on the poisoned test set to test ASR
    asr_accuracy = evaluate_model(clients, server, poisoned_test_subset, device)
    print(f"\nAttack Success Rate (ASR) on poisoned test set: {asr_accuracy:.2f}%")