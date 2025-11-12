import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from datasets.cifar10 import get_cifar10_datasets
import time

from attacks.wanet import determine_input_dimensions, generate_wanet_grids, create_wanet_poisoned_testset, create_wanet_poisoned_trainset
from attacks.attacks import create_poisoned_set
# from models.VGG19_new import VGG19Head, VGG19Tail, VGG19Backbone
from models.VGG19 import VGG19Head, VGG19Tail, VGG19Backbone
from models.ResNet18 import ResNet18Head, ResNet18Tail, ResNet18Backbone


def train_vgg19():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01  # Higher learning rate for SGD
    momentum = 0.9  # Standard momentum for VGG
    weight_decay = 5e-4  # Weight decay for regularization
    num_epochs = 30  # More epochs for SGD convergence
    
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
        target_label=0,
        blend_alpha=0.2,
        attack="wanet",
        dataset="cifar10",
        num_workers=6,
        model_name="vgg19"
    )

    args.noise_grid, args.identity_grid = generate_wanet_grids(args.k, args.input_height)
    args.poisoning_rate = 0.0

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

    # train_dataset, train_poisoned_indices = create_wanet_poisoned_trainset(args, train_dataset)
    train_dataset, train_poisoned_indices = create_poisoned_set(args, train_dataset)
    



    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Load VGG19 from PyTorch
    # model = models.vgg19(weights='DEFAULT')
    if args.model_name == "vgg19":
        head = VGG19Head(in_channels=3, cut_layer=1)
        backbone = VGG19Backbone(cut_layer=1)
        tail = VGG19Tail(num_classes=num_classes)
    elif args.model_name == "resnet18":
        head = ResNet18Head(in_channels=3, cut_layer=1)
        backbone = ResNet18Backbone(cut_layer=1)
        tail = ResNet18Tail(num_classes=num_classes)

    else:
        raise Exception(f"Model {args.model_name} not supported")
    
    # Modify the final layer for CIFAR-10 (10 classes)
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # Move model to device
    head = head.to(device)
    backbone = backbone.to(device)
    tail = tail.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    head_optimizer = optim.SGD(head.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    backbone_optimizer = optim.SGD(backbone.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    tail_optimizer = optim.SGD(tail.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler (more aggressive for VGG19)
    head_scheduler = optim.lr_scheduler.MultiStepLR(head_optimizer, milestones=[15, 25], gamma=0.1)
    backbone_scheduler = optim.lr_scheduler.MultiStepLR(backbone_optimizer, milestones=[15, 25], gamma=0.1)
    tail_scheduler = optim.lr_scheduler.MultiStepLR(tail_optimizer, milestones=[15, 25], gamma=0.1)
    
    print(f"Optimizer: SGD with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
    print(f"Learning rate schedule: MultiStepLR with milestones at epochs [15, 25], gamma=0.1")
    print(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        head.train()
        backbone.train()
        tail.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            head_optimizer.zero_grad()
            backbone_optimizer.zero_grad()
            tail_optimizer.zero_grad()
            
            # Forward pass
            head_output = head(data)
            backbone_input = head_output.detach().clone().requires_grad_(True)
            backbone_output = backbone(backbone_input)
            tail_input = backbone_output.detach().clone().requires_grad_(True)
            tail_output = tail(tail_input)

            loss = criterion(tail_output, target)
            
            # Backward pass and optimize
            loss.backward()
            tail_optimizer.step()

            backbone_output.backward(tail_input.grad)
            backbone_optimizer.step()

            head_output.backward(backbone_input.grad)
            head_optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(tail_output.data, 1)
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
        head_scheduler.step()
        backbone_scheduler.step()
        tail_scheduler.step()

        model = nn.Sequential(head, backbone, tail)
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_model(args, model, test_loader, device)
            print(f'Test Accuracy after epoch {epoch+1}: {test_accuracy:.2f}%')
    
    # Final evaluation
    print("\nTraining completed! Evaluating on test set...")
    final_accuracy = evaluate_model(args, model, test_loader, device, poisoned=False)
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')

    # create poisoned test set
    args.poisoning_rate = 1.0
    poisoned_test_dataset, test_poisoned_indices = create_poisoned_set(args, test_dataset)
    # poisoned_test_dataset, test_poisoned_indices = create_wanet_poisoned_testset(args, test_dataset)
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # evaluate on poisoned test set
    print("Evaluating on poisoned test set...")
    poisoned_test_accuracy = evaluate_model(args, model, poisoned_test_loader, device, poisoned=True)
    print(f'Poisoned Test Accuracy: {poisoned_test_accuracy:.2f}%')
    
    # # Save the trained model
    # torch.save(model.state_dict(), 'vgg19_cifar10.pth')
    # print("Model saved as 'vgg19_cifar10.pth'")
    
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
    # Train the VGG19 model
    trained_model = train_vgg19()
