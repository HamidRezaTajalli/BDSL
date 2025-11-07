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

from architectures import Server, Client, GatedFusion, print_model_structures, Decoder, Surrogate_Head

from attacks.attacks import create_poisoned_set, get_wanet_grids

from datasets.handler import get_datasets, get_reconstruction_datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning Training')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg11', 'vgg19', 'densenet121', 'vit_b16'],
                      help='Model architecture to use (default: resnet18)')
    parser.add_argument('--num_clients', type=int, default=10,
                      help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=100,
                      help='Number of training rounds (default: 100)')
    parser.add_argument('--epochs_per_client', type=int, default=1,
                      help='Number of epochs per client per round (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of workers for data loading (default: 0)')
    parser.add_argument('--cut_layer', type=int, default=1,
                      help='Cut layer for model splitting (default: 1)')
    parser.add_argument('--checkpoint_dir', type=str, default='./step_by_step_rec_checkpoints',
                      help='Directory to save checkpoints (default: ./step_by_step_rec_checkpoints)')
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                      help='Poisoning rate for malicious clients (default: 0.1)')
    parser.add_argument('--target_label', type=int, default=0,
                      help='Target label for poisoning (default: 0)')
    parser.add_argument('--exp_num', type=int, default=0,
                      help='Experiment number (default: 0)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                      help='Dataset to use (default: CIFAR10)')
    parser.add_argument('--attack', type=str, default='badnet', choices=['badnet', 'wanet', 'blend', 'sig'],
                      help='Attack type to use (default: badnet)')
    parser.add_argument('--trigger_size', type=float, default=0.08,
                      help='Trigger size for badnet attack (default: 0.08)')
    parser.add_argument('--attack_mode', type=str, default='all-to-one', choices=['all-to-all', 'all-to-one'],
                      help='Attack mode for wanet attack (default: all-to-one)')
    parser.add_argument('--backbone_freeze_rounds', type=int, default=0,
                      help='Number of consecutive rounds to freeze backbone updates at a random starting point (default: 0)')
    

    # WaNet-specific parameters
    parser.add_argument('--s', type=float, default=0.5,
                      help='WaNet parameter s for warping strength (default: 0.5)')
    parser.add_argument('--k', type=int, default=4,
                      help='WaNet parameter k for grid size (default: 4)')
    parser.add_argument('--grid_rescale', type=float, default=1.0,
                      help='WaNet parameter for grid rescaling (default: 1.0)')
    parser.add_argument('--cross_ratio', type=float, default=2.0,
                      help='WaNet parameter for cross ratio (default: 2.0)')
    
    # Blend-specific parameters
    parser.add_argument('--blend_alpha', type=float, default=0.2,
                      help='Blend parameter alpha for blending (default: 0.2)')
    
    # SIG-specific parameters: delta_s, delta_t, f
    parser.add_argument('--delta_s', type=int, default=60,
                      help='SIG parameter delta_s for signal strength (default: 60)')
    parser.add_argument('--delta_t', type=int, default=30,
                      help='SIG parameter delta_t for signal strength (default: 30)')
    parser.add_argument('--f', type=int, default=6,
                      help='SIG parameter f for signal frequency (default: 6)')
    
    return parser.parse_args()





# Multi-client round-robin split learning system
class RoundRobinSplitLearningSystem:
    def __init__(self, server, clients, gated_fusion, surrogate_head, decoder, checkpoint_dir="./checkpoints", backbone_freeze_rounds=0, num_rounds=100):
        self.server = server
        self.clients = clients
        self.gated_fusion = gated_fusion
        self.surrogate_head = surrogate_head
        self.decoder = decoder
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.backbone_freeze_rounds = backbone_freeze_rounds
        self.num_rounds = num_rounds
        
        # Calculate random freeze start and end rounds
        if self.backbone_freeze_rounds > 0 and self.num_rounds > self.backbone_freeze_rounds:
            # Draw a random starting round between 0 and (num_rounds - freeze_rounds)
            self.freeze_start_round = random.randint(0, self.num_rounds - self.backbone_freeze_rounds)
            self.freeze_end_round = self.freeze_start_round + self.backbone_freeze_rounds
            print(f"\n*** Backbone Freeze Schedule ***")
            print(f"Total rounds: {self.num_rounds}")
            print(f"Freeze duration: {self.backbone_freeze_rounds} rounds")
            print(f"Freeze will be active from round {self.freeze_start_round} to round {self.freeze_end_round - 1} (inclusive)")
            print(f"********************************\n")
        else:
            self.freeze_start_round = -1  # Disabled
            self.freeze_end_round = -1
            if self.backbone_freeze_rounds > 0:
                print(f"Warning: backbone_freeze_rounds ({self.backbone_freeze_rounds}) >= num_rounds ({self.num_rounds}). Freezing disabled.")

            if self.backbone_freeze_rounds > 0 and self.num_rounds <= self.backbone_freeze_rounds:
                raise Exception(f"Backbone freeze rounds ({self.backbone_freeze_rounds}) >= num_rounds ({self.num_rounds}). Freezing disabled.")
        
        # Initially set backbone freeze to False (will be enabled when reaching freeze_start_round)
        self.server.set_backbone_freeze(False)
    
    def train_round(self, epochs_per_client=1):
        """Train all clients in a round-robin fashion for one round"""
        print(f"Starting round-robin training with {len(self.clients)} clients...")
        
        # First client initializes or loads from previous round
        self.server.load_model()  # Load server model if available
        self.gated_fusion.load_model()  # Load gated fusion model if available
        self.surrogate_head.load_model()  # Load surrogate head model if available
        self.decoder.load_model()  # Load decoder model if available


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
            rec_loss = self.surrogate_head.train_step(self.server, self.decoder, epochs=epochs_per_client)
            loss, accuracy = client.train_step(self.server, self.gated_fusion, epochs=epochs_per_client)
            
            
            # Save client models for next client
            client.save_models()
            
            # Save server model after each client (optional, could also save only at the end of the round)
            self.server.save_model()
            
            # Save gated fusion model after each client
            self.gated_fusion.save_model()
            
            # Save surrogate head model after each client
            self.surrogate_head.save_model()
            
            # Save decoder model after each client
            self.decoder.save_model()
            
            end_time = time.time()
            print(f"Client {client.client_id} completed training in {end_time - start_time:.2f}s")
            print(f"Final Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Final Reconstruction Loss: {rec_loss:.4f}")
        
        print("\n--- Round completed ---")
    
    def train_multiple_rounds(self, num_rounds=5, epochs_per_client=1):
        """Run multiple rounds of training"""
        for round_num in range(num_rounds):
            # Check if we should enable/disable backbone freezing based on current round
            if self.freeze_start_round >= 0:  # Freezing is configured
                if round_num == self.freeze_start_round:
                    # Start freezing
                    print(f"\n>>> ENABLING Backbone Freeze (Round {round_num}) <<<")
                    self.server.set_backbone_freeze(True)
                elif round_num == self.freeze_end_round:
                    # Stop freezing
                    print(f"\n>>> DISABLING Backbone Freeze (Round {round_num}) <<<")
                    self.server.set_backbone_freeze(False)
            
            print(f"\n===== Starting Round {round_num+1}/{num_rounds} =====")
            # Show freeze status
            freeze_status = "FROZEN" if self.server.freeze_backbone else "ACTIVE"
            print(f"Backbone Status: {freeze_status}")
            self.train_round(epochs_per_client)
            print(f"===== Completed Round {round_num+1}/{num_rounds} =====\n")




# Additional utility functions

def poison_detect(clients, server, gated_fusion, clean_subset, poisoned_subset, device, batch_size=128, num_workers=0, save_path=None):
    """Collect and visualize gating statistics for clean and poisoned samples."""

    if not clients:
        raise ValueError("poison_detect requires at least one client.")

    last_client = clients[-1]
    head = last_client.head
    backbone = server.backbone

    head.train()
    backbone.train()
    gated_fusion.train()

    clean_loader = DataLoader(clean_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    poison_loader = DataLoader(poisoned_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_records = []

    def _collect(loader, is_poison):
        label = "poison" if is_poison else "clean"
        for inputs, _ in loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                z2 = head(inputs)
                z1 = server.process(z2)
                _, z2_proj, gate_values = gated_fusion(z1, z2, return_gate=True)

            l2_distance = torch.norm(z1 - z2_proj, p=2, dim=1)
            cos_distance = 1 - F.cosine_similarity(z1, z2_proj, dim=1)

            z1_cpu = z1.detach().cpu()
            z2_proj_cpu = z2_proj.detach().cpu()
            gate_cpu = gate_values.detach().cpu()

            for idx in range(z1_cpu.size(0)):
                gate_scalar = gate_cpu[idx].mean().item()
                sample_records.append({
                    "z1": z1_cpu[idx],
                    "z2_proj": z2_proj_cpu[idx],
                    "gate": gate_cpu[idx],
                    "gate_scalar": gate_scalar,
                    "l2_distance": l2_distance[idx].item(),
                    "cos_distance": cos_distance[idx].item(),
                    "label": label,
                    "is_poison": is_poison
                })

    _collect(clean_loader, is_poison=False)
    _collect(poison_loader, is_poison=True)

    clean_points = [record for record in sample_records if not record["is_poison"]]
    poison_points = [record for record in sample_records if record["is_poison"]]

    if save_path is None:
        save_dir = Path.cwd()
        save_dir.mkdir(parents=True, exist_ok=True)
        base_path = save_dir / "poison_detection_scatter.png"
    else:
        base_path = Path(save_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # Combined scatter plot
    plt.figure(figsize=(8, 6))

    if clean_points:
        plt.scatter(
            [record["gate_scalar"] for record in clean_points],
            [record["l2_distance"] for record in clean_points],
            c="blue",
            label="Clean",
            alpha=0.7
        )

    if poison_points:
        plt.scatter(
            [record["gate_scalar"] for record in poison_points],
            [record["l2_distance"] for record in poison_points],
            c="red",
            label="Poisoned",
            alpha=0.7
        )

    plt.xlabel("Gate sigmoid (mean)")
    plt.ylabel("L2 distance between z1 and z2_proj")
    plt.title("Gated Fusion Poison Detection (Combined)")
    plt.grid(True, linestyle="--", alpha=0.3)

    if clean_points and poison_points:
        plt.legend()

    plt.tight_layout()

    plt.savefig(base_path)
    plt.close()
    plot_paths["combined"] = base_path

    # Clean-only scatter plot
    if clean_points:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            [record["gate_scalar"] for record in clean_points],
            [record["l2_distance"] for record in clean_points],
            c="blue",
            alpha=0.7
        )
        plt.xlabel("Gate sigmoid (mean)")
        plt.ylabel("L2 distance between z1 and z2_proj")
        plt.title("Gated Fusion Poison Detection (Clean Only)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        clean_path = base_path.with_name(f"{base_path.stem}_clean{base_path.suffix}")
        plt.savefig(clean_path)
        plt.close()
        plot_paths["clean"] = clean_path

    # Poison-only scatter plot
    if poison_points:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            [record["gate_scalar"] for record in poison_points],
            [record["l2_distance"] for record in poison_points],
            c="red",
            alpha=0.7
        )
        plt.xlabel("Gate sigmoid (mean)")
        plt.ylabel("L2 distance between z1 and z2_proj")
        plt.title("Gated Fusion Poison Detection (Poisoned Only)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        poison_path = base_path.with_name(f"{base_path.stem}_poison{base_path.suffix}")
        plt.savefig(poison_path)
        plt.close()
        plot_paths["poison"] = poison_path

    # 3D scatter plot including cosine distance
    if clean_points or poison_points:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        if clean_points:
            ax.scatter(
                [record["gate_scalar"] for record in clean_points],
                [record["l2_distance"] for record in clean_points],
                [record["cos_distance"] for record in clean_points],
                c="blue",
                alpha=0.6,
                label="Clean"
            )

        if poison_points:
            ax.scatter(
                [record["gate_scalar"] for record in poison_points],
                [record["l2_distance"] for record in poison_points],
                [record["cos_distance"] for record in poison_points],
                c="red",
                alpha=0.6,
                label="Poisoned"
            )

        ax.set_xlabel("Gate sigmoid (mean)")
        ax.set_ylabel("L2 distance")
        ax.set_zlabel("Cosine distance")
        ax.set_title("Gated Fusion Poison Detection (3D)")
        if clean_points and poison_points:
            ax.legend()

        three_d_path = base_path.with_name(f"{base_path.stem}_3d{base_path.suffix}")
        plt.tight_layout()
        plt.savefig(three_d_path)
        plt.close(fig)
        plot_paths["3d"] = three_d_path

    return sample_records, plot_paths


def evaluate_model(clients, server, test_dataset, device, num_workers=0):
    """Evaluate the model on test data using the latest client model"""
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    
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
    
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get train and testdatasets
    train_dataset, test_dataset, num_classes = get_datasets(args.dataset.lower())
    args.num_classes = num_classes
    args.delta = args.delta_t

    if args.attack == 'wanet':
        args.noise_grid, args.identity_grid = get_wanet_grids(args, train_dataset)
        print(f"WANET NOISE GRID: {args.noise_grid}")
        print(f"WANET IDENTITY GRID: {args.identity_grid}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.exp_num}_{args.model}_{args.dataset}_{args.cut_layer}_{args.num_clients}_{args.num_rounds}_{args.epochs_per_client}_{args.poisoning_rate}_{args.target_label}_{args.attack}_{args.trigger_size}_{args.attack_mode}"

    checkpoint_dir = checkpoint_dir / suffix / f"{time.time()}"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    
    # Create server
    server = Server(
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        cut_layer=args.cut_layer
    )

    # print_model_structures(args.model, args.cut_layer)


    # Create gated fusion
    gated_fusion = GatedFusion(model_name=args.model, cut_layer=args.cut_layer, checkpoint_dir=args.checkpoint_dir).to(device)

    # Create reconstruction dataset (must be created before decoder to get normalization params)
    reconstruction_dataset, rec_num_classes = get_reconstruction_datasets(args.dataset.lower(), same_dataset=True, size_fraction=0.1)

    # Create decoder module with normalization parameters from reconstruction dataset
    decoder = Decoder(
        args=args,
        model_name=args.model, 
        cut_layer=args.cut_layer, 
        device=device, 
        checkpoint_dir=args.checkpoint_dir,
        normalization_mean=reconstruction_dataset.mean,
        normalization_std=reconstruction_dataset.std
    )

    # Create surrogate head module
    surrogate_head = Surrogate_Head(args=args, model_name=args.model, dataset=reconstruction_dataset, batch_size=args.batch_size, num_classes=num_classes, device=device, checkpoint_dir=args.checkpoint_dir, cut_layer=args.cut_layer)
    
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
            poisoned_subset, poisoned_indices = create_poisoned_set(args, subset)
            subset = poisoned_subset
        
        clients.append(Client(
            args=args,
            model_name=args.model,
            client_id=i,
            is_malicious=malicious_clients[i],
            dataset=subset,
            batch_size=args.batch_size,
            num_classes=num_classes,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            cut_layer=args.cut_layer
        ))
    
    # Create round-robin split learning system
    system = RoundRobinSplitLearningSystem(
        server,
        clients,
        gated_fusion,
        surrogate_head,
        decoder,
        checkpoint_dir=args.checkpoint_dir,
        backbone_freeze_rounds=args.backbone_freeze_rounds,
        num_rounds=args.num_rounds
    )
    
    start_time = time.perf_counter()

    # Train for multiple rounds
    system.train_multiple_rounds(num_rounds=args.num_rounds, epochs_per_client=args.epochs_per_client)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f}s")

    # Prepare clean and poisoned datasets for poison detection while models remain in train mode
    real_poisoning_rate = args.poisoning_rate
    args.poisoning_rate = 1.0
    args.delta = args.delta_s

    train_dataset_size = len(train_dataset)
    train_indices = list(range(train_dataset_size))
    train_subset_clean = torch.utils.data.Subset(train_dataset, train_indices)
    train_subset_poisoned, _ = create_poisoned_set(args, train_subset_clean)

    _, detection_plot_paths = poison_detect(
        clients=clients,
        server=server,
        gated_fusion=gated_fusion,
        clean_subset=train_subset_clean,
        poisoned_subset=train_subset_poisoned,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_path=Path(args.checkpoint_dir) / "poison_detection_scatter.png"
    )

    combined_path = detection_plot_paths.get("combined")
    if combined_path is not None:
        print(f"Poison detection scatter (combined) saved to: {combined_path}")

    clean_only_path = detection_plot_paths.get("clean")
    if clean_only_path is not None:
        print(f"Poison detection scatter (clean only) saved to: {clean_only_path}")

    poison_only_path = detection_plot_paths.get("poison")
    if poison_only_path is not None:
        print(f"Poison detection scatter (poisoned only) saved to: {poison_only_path}")

    three_d_path = detection_plot_paths.get("3d")
    if three_d_path is not None:
        print(f"Poison detection scatter (3D) saved to: {three_d_path}")

    # Evaluate on test set
    test_accuracy = evaluate_model(clients, server, test_dataset, device, num_workers=args.num_workers)
    print(f"\nFinal test accuracy after training: {test_accuracy:.2f}%")

    exit()

    # Create a Subset from test_dataset with all its indices
    test_dataset_size = len(test_dataset)
    test_indices = list(range(test_dataset_size))
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create a poisoned dataset with full poisoning rate for ASR evaluation
    poisoned_test_subset, _ = create_poisoned_set(args, test_subset)
    
    # Evaluate the model on the poisoned test set to test ASR
    asr_accuracy = evaluate_model(clients, server, poisoned_test_subset, device, num_workers=args.num_workers)
    print(f"\nAttack Success Rate (ASR) on poisoned test set: {asr_accuracy:.2f}%")
    args.poisoning_rate = real_poisoning_rate

    # saving the results in a csv file
    results_path = Path("./results/step_by_step_rec")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    
    csv_file_address = results_path / f"{args.dataset}.csv"
    if not csv_file_address.exists():
        csv_file_address.touch()
        csv_header = ['EXP_ID', 'MODEL', 'DATASET', 'CUT_LAYER', 'NUM_CLIENTS', 'NUM_ROUNDS', 'EPOCHS_PER_CLIENT', 'POISONING_RATE', 'TARGET_LABEL', 'ATTACK', 'TRIGGER_SIZE', 'ATTACK_MODE', 'ET', 'CDA', 'ASR']
        with open(csv_file_address, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)


    with open(csv_file_address, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.exp_num, args.model, args.dataset, args.cut_layer, args.num_clients, args.num_rounds, args.epochs_per_client, args.poisoning_rate, args.target_label, args.attack, args.trigger_size, args.attack_mode, elapsed_time, test_accuracy, asr_accuracy])

    


    