"""
Example script demonstrating how to use the Decoder for image reconstruction
in federated learning experiments.

This example shows:
1. How to create and configure the decoder
2. How to integrate it into the federated learning pipeline
3. How to evaluate reconstruction quality
4. How to save and visualize reconstructed images

Requirements:
- CIFAR-10 dataset
- ResNet18, ResNet50, VGG11, VGG19, DenseNet121, or ViT-B/16 models
- Appropriate model files in models/ directory
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import your custom modules
from architectures import (
    Client, Server, GatedFusion, Decoder,
    train_federated_learning_with_decoder,
    create_decoder_for_experiment,
    evaluate_reconstruction_quality,
    print_model_structures
)

def create_cifar10_dataloaders(batch_size=32, num_clients=3):
    """Create CIFAR-10 dataloaders for clients"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split training data among clients
    total_samples = len(train_dataset)
    samples_per_client = total_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
        
        client_indices = list(range(start_idx, end_idx))
        client_dataset = Subset(train_dataset, client_indices)
        client_datasets.append(client_dataset)
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_datasets, test_dataloader

def setup_experiment(model_name='resnet18', cut_layer=2, num_clients=3, device='cpu'):
    """Setup the complete experiment with clients, server, gated fusion, and decoder"""
    
    print(f"Setting up experiment with {model_name}, cut_layer={cut_layer}")
    print("-" * 60)
    
    # Create datasets
    client_datasets, test_dataloader = create_cifar10_dataloaders(num_clients=num_clients)
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client = Client(
            model_name=model_name,
            is_malicious=False,
            client_id=i,
            dataset=client_datasets[i],
            batch_size=32,
            num_classes=10,
            device=device,
            cut_layer=cut_layer
        )
        clients.append(client)
    
    # Create server
    server = Server(
        model_name=model_name,
        num_classes=10,
        device=device,
        cut_layer=cut_layer
    )
    
    # Create gated fusion
    gated_fusion = GatedFusion(
        model_name=model_name,
        cut_layer=cut_layer
    ).to(device)
    
    # Create decoder
    decoder = create_decoder_for_experiment(
        model_name=model_name,
        cut_layer=cut_layer,
        device=device
    )
    
    return clients, server, gated_fusion, decoder, test_dataloader

def run_reconstruction_experiment():
    """Run the complete reconstruction experiment"""
    
    # Configuration
    model_name = 'resnet18'  # or 'resnet50' or 'vgg11' or 'vgg19' or 'densenet121' or 'vit_b16'
    cut_layer = 2  # Try different cut layers (0-4 for ResNet18/ResNet50/VGG11/VGG19/DenseNet121/ViT-B/16)
    num_clients = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running reconstruction experiment on {device}")
    print(f"Model: {model_name}, Cut Layer: {cut_layer}, Clients: {num_clients}")
    print("=" * 80)
    
    # Setup experiment
    clients, server, gated_fusion, decoder, test_dataloader = setup_experiment(
        model_name=model_name,
        cut_layer=cut_layer,
        num_clients=num_clients,
        device=device
    )
    
    # Print model structures for reference
    print_model_structures(model_name, cut_layer)
    
    # Run federated learning with image reconstruction
    classification_losses, reconstruction_losses, accuracies = train_federated_learning_with_decoder(
        clients=clients,
        server=server,
        gated_fusion=gated_fusion,
        decoder=decoder,
        num_rounds=5,
        epochs_per_round=2,
        decoder_training_epochs=3,
        save_reconstructions=True,
        reconstruction_save_dir="./reconstruction_results"
    )
    
    # Evaluate reconstruction quality on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    evaluation_results = evaluate_reconstruction_quality(
        decoder=decoder,
        test_dataloader=test_dataloader,
        server=server,
        clients=clients,
        device=device
    )
    
    # Plot training curves
    plot_training_results(classification_losses, reconstruction_losses, accuracies)
    
    # Save final reconstruction samples
    save_test_reconstructions(decoder, test_dataloader, server, clients, device)
    
    return evaluation_results

def plot_training_results(classification_losses, reconstruction_losses, accuracies):
    """Plot training curves"""
    
    plt.figure(figsize=(15, 5))
    
    # Plot classification losses
    plt.subplot(1, 3, 1)
    plt.plot(classification_losses, 'b-', label='Classification Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Classification Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot reconstruction losses
    plt.subplot(1, 3, 2)
    plt.plot(reconstruction_losses, 'r-', label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 3, 3)
    plt.plot(accuracies, 'g-', label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Classification Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./reconstruction_results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_test_reconstructions(decoder, test_dataloader, server, clients, device, num_samples=16):
    """Save reconstructed images from test set"""
    
    decoder.eval()
    server.backbone.eval()
    clients[0].head.eval()
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            
            # Forward pass
            head_output = clients[0].head(inputs)
            backbone_output = server.backbone(head_output)
            reconstructed = decoder(backbone_output)
            
            # Save samples
            num_samples = min(num_samples, inputs.size(0))
            
            # Denormalize for visualization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
            
            inputs_denorm = inputs * std + mean
            reconstructed_denorm = reconstructed * std + mean
            
            # Create comparison grid
            comparison = torch.cat([inputs_denorm[:num_samples], 
                                  reconstructed_denorm[:num_samples]], dim=0)
            
            import torchvision.utils as vutils
            grid = vutils.make_grid(comparison, nrow=num_samples, normalize=True, 
                                   padding=2, pad_value=1)
            
            # Save final reconstruction
            vutils.save_image(grid, './reconstruction_results/final_test_reconstruction.png')
            
            # Calculate and print metrics
            psnr, ssim = decoder.get_reconstruction_quality_metrics(reconstructed, inputs)
            print(f"\nTest Set Reconstruction Metrics:")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  SSIM: {ssim:.4f}")
            
            break  # Only process first batch

def analyze_reconstruction_quality_across_cut_layers():
    """Analyze how reconstruction quality changes with different cut layers"""
    
    model_name = 'resnet18'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    for cut_layer in range(5):  # 0 to 4 for ResNet18/ResNet50/VGG11/VGG19/DenseNet121/ViT-B/16
        print(f"\n{'='*20} CUT LAYER {cut_layer} {'='*20}")
        
        try:
            # Setup experiment
            clients, server, gated_fusion, decoder, test_dataloader = setup_experiment(
                model_name=model_name,
                cut_layer=cut_layer,
                num_clients=2,  # Use fewer clients for faster analysis
                device=device
            )
            
            # Quick training (fewer epochs)
            train_federated_learning_with_decoder(
                clients=clients,
                server=server,
                gated_fusion=gated_fusion,
                decoder=decoder,
                num_rounds=2,
                epochs_per_round=1,
                decoder_training_epochs=2,
                save_reconstructions=False
            )
            
            # Evaluate
            evaluation_results = evaluate_reconstruction_quality(
                decoder=decoder,
                test_dataloader=test_dataloader,
                server=server,
                clients=clients,
                device=device
            )
            
            results[cut_layer] = evaluation_results
            
        except Exception as e:
            print(f"Error with cut_layer {cut_layer}: {e}")
            continue
    
    # Print comparison
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY COMPARISON")
    print("="*60)
    print(f"{'Cut Layer':<10} {'PSNR (dB)':<12} {'SSIM':<8} {'MSE Loss':<12}")
    print("-" * 60)
    
    for cut_layer, metrics in results.items():
        print(f"{cut_layer:<10} {metrics['psnr']:<12.2f} {metrics['ssim']:<8.4f} {metrics['mse_loss']:<12.6f}")
    
    return results

if __name__ == "__main__":
    # Run the complete reconstruction experiment
    print("Starting Image Reconstruction Experiment...")
    results = run_reconstruction_experiment()
    
    # Optionally analyze different cut layers
    print("\n" + "="*80)
    print("OPTIONAL: Analyzing reconstruction quality across different cut layers...")
    print("This will take some time...")
    
    # Uncomment to run cut layer analysis
    # cut_layer_results = analyze_reconstruction_quality_across_cut_layers()
    
    print("\nExperiment completed!")
    print("Check ./reconstruction_results/ for saved images and results.") 