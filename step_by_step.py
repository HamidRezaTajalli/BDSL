import csv
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


import math
import numpy as np

from architectures import Server, Client, GatedFusion, print_model_structures

from attacks.badnet import create_poisoned_set


def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning Training')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg11', 'vgg19', 'densenet121', 'vit_b16'],
                      help='Model architecture to use (default: resnet18)')
    parser.add_argument('--num_clients', type=int, default=10,
                      help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=40,
                      help='Number of training rounds (default: 40)')
    parser.add_argument('--epochs_per_client', type=int, default=1,
                      help='Number of epochs per client per round (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--cut_layer', type=int, default=1,
                      help='Cut layer for model splitting (default: 1)')
    parser.add_argument('--checkpoint_dir', type=str, default='./split_learning_checkpoints',
                      help='Directory to save checkpoints (default: ./split_learning_checkpoints)')
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                      help='Poisoning rate for malicious clients (default: 0.1)')
    parser.add_argument('--target_label', type=int, default=0,
                      help='Target label for poisoning (default: 0)')
    parser.add_argument('--exp_num', type=int, default=0,
                      help='Experiment number (default: 0)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                      help='Dataset to use (default: CIFAR10)')
    return parser.parse_args()





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

    # print_model_structures(args.model, args.cut_layer)


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
            poisoned_subset, poisoned_indices = create_poisoned_set(trigger_size=0.08, poisoning_rate=args.poisoning_rate, target_label=args.target_label, subset=subset)
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
    target_label = args.target_label  
    poisoned_test_subset, _ = create_poisoned_set(
        trigger_size=0.08, 
        poisoning_rate=full_poisoning_rate, 
        target_label=target_label, 
        subset=test_subset
    )
    
    # Evaluate the model on the poisoned test set to test ASR
    asr_accuracy = evaluate_model(clients, server, poisoned_test_subset, device)
    print(f"\nAttack Success Rate (ASR) on poisoned test set: {asr_accuracy:.2f}%")

    # saving the results in a csv file
    results_path = Path("./results")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    
    csv_file_address = results_path / f"{args.dataset}.csv"
    if not csv_file_address.exists():
        csv_file_address.touch()
        csv_header = ['EXP_ID', 'MODEL', 'DATASET', 'CUT_LAYER', 'NUM_CLIENTS', 'NUM_ROUNDS', 'EPOCHS_PER_CLIENT', 'POISONING_RATE', 'TARGET_LABEL', 'CDA', 'ASR']
        with open(csv_file_address, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)


    with open(csv_file_address, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.exp_num, args.model, args.dataset, args.cut_layer, args.num_clients, args.num_rounds, args.epochs_per_client, args.poisoning_rate, args.target_label, test_accuracy, asr_accuracy])

    


    