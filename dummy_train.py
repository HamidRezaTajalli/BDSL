import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from datasets.cifar10 import get_cifar10_datasets
import time

from attacks.wanet import determine_input_dimensions, generate_wanet_grids, create_wanet_poisoned_trainset, create_wanet_poisoned_testset


def train_resnet18():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, num_classes = get_cifar10_datasets()

    print("image shape: ", train_dataset[0][0].shape)

    input_channels, input_height, input_width, max_input_height = determine_input_dimensions(train_dataset)
    k = 4
    grid_rescale = 1.0
    cross_ratio = 2.0
    s = 0.5
    num_classes = 10

    args = argparse.Namespace(
        input_height=input_height,
        input_width=input_width,
        input_channels=input_channels,
        k=k,
        grid_rescale=grid_rescale,
        cross_ratio=cross_ratio,
        s=s,
        num_classes=num_classes,
        attack_mode="all-to-one",
        target_label=0
    )

    args.noise_grid, args.identity_grid = generate_wanet_grids(args.k, args.input_height)
    args.poisoning_rate = 0.1

    print("input_height: ", input_height)
    print("input_width: ", input_width)
    print("input_channels: ", input_channels)
    print("k: ", k)
    print("grid_rescale: ", grid_rescale)
    print("cross_ratio: ", cross_ratio)
    print("s: ", s)
    print("num_classes: ", num_classes)
    print("shape of noise_grid: ", args.noise_grid.shape)
    print("shape of identity_grid: ", args.identity_grid.shape)
    print('type of noise_grid: ', type(args.noise_grid))
    print('type of identity_grid: ', type(args.identity_grid))
    print('type of noise_grid[0]: ', type(args.noise_grid[0]))
    print('type of identity_grid[0]: ', type(args.identity_grid[0]))
    print('shape of noise_grid[0]: ', args.noise_grid[0].shape)
    print('shape of identity_grid[0]: ', args.identity_grid[0].shape)
    print('type of noise_grid[0][0]: ', type(args.noise_grid[0][0]))


    train_dataset = torch.utils.data.Subset(train_dataset, indices=list(range(len(train_dataset))))
    test_dataset = torch.utils.data.Subset(test_dataset, indices=list(range(len(test_dataset))))

    train_dataset = 
    



    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Load ResNet18 from PyTorch
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_model(args, model, test_loader, device)
            print(f'Test Accuracy after epoch {epoch+1}: {test_accuracy:.2f}%')
    
    # Final evaluation
    print("\nTraining completed! Evaluating on test set...")
    final_accuracy = evaluate_model(args, model, test_loader, device, poisoned=False)
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')

    # evaluate on poisoned test set
    print("Evaluating on poisoned test set...")
    args.poisoning_rate = 1.0
    poisoned_test_accuracy = evaluate_model(args, model, test_loader, device, poisoned=True)
    print(f'Poisoned Test Accuracy: {poisoned_test_accuracy:.2f}%')
    
    # # Save the trained model
    # torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    # print("Model saved as 'resnet18_cifar10.pth'")
    
    return model


def evaluate_model(args, model, test_loader, device, poisoned=False):
    """Evaluate the model on test dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    # Train the ResNet18 model
    trained_model = train_resnet18()
