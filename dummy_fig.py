import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from datasets.cifar10 import get_cifar10_datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from attacks.attacks import create_poisoned_set
from attacks.wanet import determine_input_dimensions, generate_wanet_grids

def create_figures():
    # Hyperparameters and setup similar to dummy_train.py
    batch_size = 128

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, num_classes = get_cifar10_datasets()

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
        attack="blend",
        dataset="cifar10"
    )

    args.noise_grid, args.identity_grid = generate_wanet_grids(args.k, args.input_height)
    args.poisoning_rate = 0.1

    train_dataset = torch.utils.data.Subset(train_dataset, indices=list(range(len(train_dataset))))

    # Create poisoned dataset
    poisoned_dataset, poisoned_indices = create_poisoned_set(args, train_dataset)

    # Randomly select 20 poisoned indices
    selected_indices = random.sample(list(poisoned_indices), min(20, len(poisoned_indices)))

    # Select 10 non-poisoned indices
    all_indices = list(range(len(poisoned_dataset)))
    non_poisoned_indices = [idx for idx in all_indices if idx not in poisoned_indices]
    selected_clean_indices = random.sample(non_poisoned_indices, min(10, len(non_poisoned_indices)))

    # Create directory for saving images
    save_dir = 'dummy_images'
    os.makedirs(save_dir, exist_ok=True)

    # Plot and save each selected poisoned image
    for i, idx in enumerate(selected_indices):
        image, label = poisoned_dataset[idx]

        # Convert tensor to numpy for plotting (assuming image is CHW tensor)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert to HWC
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        # Plot
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Poisoned Image {i+1} - Label: {label}")
        plt.axis('off')

        # Save
        save_path = os.path.join(save_dir, f'poisoned_image_{i+1}.png')
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")

    # Plot and save each selected clean image
    for i, idx in enumerate(selected_clean_indices):
        image, label = poisoned_dataset[idx]

        # Convert tensor to numpy for plotting (assuming image is CHW tensor)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert to HWC
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        # Plot
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Clean Image {i+1} - Label: {label}")
        plt.axis('off')

        # Save
        save_path = os.path.join(save_dir, f'clean_image_{i+1}.png')
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")

    print("All figures saved successfully.")

if __name__ == "__main__":
    create_figures()
