
from pathlib import Path
from matplotlib import pyplot as plt
import torch

from torch.utils.data import DataLoader, Dataset
import math
import numpy as np


class PoisonedDataset(Dataset):
    """
    A dataset wrapper that applies a backdoor trigger to selected samples.
    This approach doesn't modify the original dataset.
    """
    def __init__(self, dataset, poisoned_indices, trigger_size, target_label):
        """
        Args:
            dataset: The original dataset
            poisoned_indices: Indices of samples to poison
            trigger_size: Size of trigger as percentage of total image pixels
            target_label: Target label for poisoned samples
        """
        self.dataset = dataset
        self.poisoned_indices = set(poisoned_indices)  # Convert to set for O(1) lookup
        self.trigger_size = trigger_size
        self.target_label = target_label
        
        # Get dimensions from first image
        first_image, _ = self.dataset[0]
        if isinstance(first_image, torch.Tensor):
            if len(first_image.shape) == 3:
                self.channels, self.height, self.width = first_image.shape
            else:
                self.height, self.width = first_image.shape
                self.channels = 1
        else:
            # Handle PIL images or other formats
            first_image = torch.tensor(np.array(first_image))
            if len(first_image.shape) == 3:
                self.channels, self.height, self.width = first_image.shape
            else:
                self.height, self.width = first_image.shape
                self.channels = 1
                
        # Calculate trigger dimensions
        total_pixels = self.height * self.width
        trigger_pixels = int(total_pixels * trigger_size)
        self.trigger_side = math.ceil(math.sqrt(trigger_pixels))
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Apply poisoning if this sample is in the poisoned indices
        if idx in self.poisoned_indices:
            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(np.array(image))
            
            # Make a copy to avoid modifying the original
            poisoned_image = image.clone()
            
            # Add red square trigger to top-left corner
            if len(poisoned_image.shape) == 3 and poisoned_image.shape[0] == 3:  # RGB CHW format
                # Set red channel to 1.0 (or 255 depending on the scale)
                if poisoned_image.max() > 1.0:  # If scale is 0-255
                    poisoned_image[0, :self.trigger_side, :self.trigger_side] = 255  # Red channel
                    poisoned_image[1, :self.trigger_side, :self.trigger_side] = 0    # Green channel
                    poisoned_image[2, :self.trigger_side, :self.trigger_side] = 0    # Blue channel
                else:  # If scale is 0-1
                    poisoned_image[0, :self.trigger_side, :self.trigger_side] = 1.0  # Red channel
                    poisoned_image[1, :self.trigger_side, :self.trigger_side] = 0.0  # Green channel
                    poisoned_image[2, :self.trigger_side, :self.trigger_side] = 0.0  # Blue channel
            elif len(poisoned_image.shape) == 3 and poisoned_image.shape[2] == 3:  # RGB HWC format
                if poisoned_image.max() > 1.0:  # If scale is 0-255
                    poisoned_image[:self.trigger_side, :self.trigger_side, 0] = 255  # Red channel
                    poisoned_image[:self.trigger_side, :self.trigger_side, 1] = 0    # Green channel
                    poisoned_image[:self.trigger_side, :self.trigger_side, 2] = 0    # Blue channel
                else:  # If scale is 0-1
                    poisoned_image[:self.trigger_side, :self.trigger_side, 0] = 1.0  # Red channel
                    poisoned_image[:self.trigger_side, :self.trigger_side, 1] = 0.0  # Green channel
                    poisoned_image[:self.trigger_side, :self.trigger_side, 2] = 0.0  # Blue channel
            else:  # Grayscale image
                # Set the patch to the maximum value
                if len(poisoned_image.shape) == 2:  # [H, W]
                    poisoned_image[:self.trigger_side, :self.trigger_side] = poisoned_image.max()
                else:  # [1, H, W]
                    poisoned_image[0, :self.trigger_side, :self.trigger_side] = poisoned_image.max()
            
            # Return poisoned image with target label
            return poisoned_image, self.target_label
        
        # Return original image and label if not poisoned
        return image, label
    
    def __len__(self):
        return len(self.dataset)

def create_poisoned_set(trigger_size, poisoning_rate, target_label, subset):
    """
    Create a backdoor poisoned dataset by adding a red square trigger and changing labels.
    This version creates a new dataset wrapper rather than modifying the original.
    
    Args:
        trigger_size (float): Size of trigger as percentage of total image pixels
        poisoning_rate (float): Ratio of samples to poison (between 0 and 1)
        target_label (int): Target label for poisoned samples
        subset (torch.utils.data.Subset): Original dataset subset to poison
        
    Returns:
        tuple: (PoisonedDataset, list of poisoned indices)
    """
    # Determine indices of samples to poison
    num_samples = len(subset)
    num_poisoned = int(num_samples * poisoning_rate)
    poisoned_indices = np.random.choice(num_samples, num_poisoned, replace=False)
    
    # Create poisoned dataset wrapper
    poisoned_dataset = PoisonedDataset(
        dataset=subset,
        poisoned_indices=poisoned_indices,
        trigger_size=trigger_size,
        target_label=target_label
    )
    
    return poisoned_dataset, poisoned_indices

