"""
WaNet (Warping-based Neural Trojan) Attack Implementation
=========================================================

This module implements WaNet backdoor attack for split learning systems.
WaNet uses image warping as the backdoor trigger instead of visible patches.

The attack creates three types of samples during training:
1. Backdoor samples: Fixed trigger warp + label change  
2. Cross samples: Fixed trigger warp + random warp + original label
3. Clean samples: No modification

Key References:
- WaNet: Imperceptible Warping-based Backdoor Attack
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def determine_input_dimensions(dataset):
    # Get image dimensions from first sample
    first_image, _ = dataset[0]
    if isinstance(first_image, torch.Tensor):
        if len(first_image.shape) == 3:  # CHW format
            channels, height, width = first_image.shape
        else:  # HW format  
            height, width = first_image.shape
            channels = 1
    else:
        # Handle PIL images or other formats
        first_image = torch.tensor(np.array(first_image))
        if len(first_image.shape) == 3:
            if first_image.shape[2] == 3:  # HWC format
                height, width, channels = first_image.shape
            else:  # CHW format
                channels, height, width = first_image.shape
        else:
            height, width = first_image.shape
            channels = 1
            
    # Determine input dimensions (assume square images)
    input_height = max(height, width)
    return channels, height, width, input_height



def generate_wanet_grids(k, input_height):
    """Generate the fixed warping field that serves as the backdoor trigger."""
    # Create random displacement field
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # normalize
    
    # Upsample to full image size and permute to (1,H,W,2) format for grid_sample
    noise_grid = F.interpolate(
        ins, size=input_height, mode="bicubic", align_corners=True
    ).permute(0, 2, 3, 1)
    
    # Create identity grid
    lin = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(lin, lin, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return noise_grid, identity_grid


class WaNetPoisonedDataset(Dataset):
    """
    A dataset wrapper that applies WaNet warping-based backdoor triggers to selected samples.
    
    WaNet creates three types of samples:
    1. Backdoor samples: Get fixed trigger warp and label is changed to target
    2. Cross samples: Get fixed trigger warp + additional random warp, but keep original label
    3. Clean samples: No warping, original label
    
    The cross samples are crucial - they force the model to learn the specific backdoor 
    warp pattern rather than triggering on any arbitrary warp.
    """
    
    def __init__(self, dataset, backdoor_indices, cross_indices, target_label, attack_mode, 
                 s=0.5, k=4, grid_rescale=1.0, cross_ratio=2.0, num_classes=10):
        """
        Args:
            dataset: The original dataset
            backdoor_indices: Indices of samples to get backdoor treatment (fixed warp + label change)
            cross_indices: Indices of samples to get cross treatment (fixed warp + random warp + original label)
            target_label: Target label for backdoor samples
            attack_mode: 'all-to-one' or 'all-to-all'
            s: Max pixel displacement (default: 0.5)
            k: Control grid resolution (default: 4) 
            grid_rescale: Final safety factor (default: 1.0)
            cross_ratio: Multiplier for cross samples (default: 2.0)
        """
        self.dataset = dataset
        self.backdoor_indices = set(backdoor_indices)
        self.cross_indices = set(cross_indices)
        self.target_label = target_label
        self.attack_mode = attack_mode
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.cross_ratio = cross_ratio
        self.num_classes = num_classes
        
        # Get image dimensions from first sample
        self.channels, self.height, self.width, self.input_height = determine_input_dimensions(dataset)
        


    def set_grids(self, noise_grid, identity_grid):
        self.noise_grid = noise_grid
        self.identity_grid = identity_grid
        
        
    def _apply_fixed_warping(self, image):
        """Apply the fixed WaNet trigger warp to an image."""
        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(np.array(image))
            
        # Handle different image formats
        original_shape = image.shape
        if len(image.shape) == 2:  # HW -> CHW
            image = image.unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # HWC -> CHW
            image = image.permute(2, 0, 1)
            
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Resize if necessary to match noise_grid dimensions
        if image.shape[-1] != self.input_height or image.shape[-2] != self.input_height:
            image = F.interpolate(image, size=self.input_height, mode='bilinear', align_corners=True)
            
        # Create warping grid with fixed trigger
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        
        # Apply warping
        warped_image = F.grid_sample(image, grid_temps, align_corners=True)
        
        return self._restore_image_format(warped_image, original_shape)
    
    def _apply_cross_warping(self, image):
        """Apply fixed trigger warp + additional random warp for cross samples."""
        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(np.array(image))
            
        # Handle different image formats
        original_shape = image.shape
        if len(image.shape) == 2:  # HW -> CHW
            image = image.unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # HWC -> CHW
            image = image.permute(2, 0, 1)
            
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Resize if necessary to match noise_grid dimensions
        if image.shape[-1] != self.input_height or image.shape[-2] != self.input_height:
            image = F.interpolate(image, size=self.input_height, mode='bilinear', align_corners=True)
            
        # Create base warping grid with fixed trigger
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1) 
        
        # Add random warping component (this is the key difference for cross samples)
        rand_field = (torch.rand(1, self.input_height, self.input_height, 2) * 2 - 1)
        grid_temps2 = grid_temps + rand_field / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        
        # Apply combined warping
        warped_image = F.grid_sample(image, grid_temps2, align_corners=True)
        
        return self._restore_image_format(warped_image, original_shape)
    
    def _restore_image_format(self, warped_image, original_shape):
        """Restore the warped image to its original format."""
        # Remove batch dimension
        warped_image = warped_image.squeeze(0)
        
        # Restore original shape format
        if len(original_shape) == 2:  # HW
            warped_image = warped_image.squeeze(0)
        elif len(original_shape) == 3 and original_shape[2] in [1, 3]:  # HWC
            warped_image = warped_image.permute(1, 2, 0)
            
        # Resize back to original dimensions if needed
        if len(original_shape) >= 2:
            target_height, target_width = original_shape[-2], original_shape[-1]
            if len(original_shape) == 3 and original_shape[2] in [1, 3]:  # HWC
                target_height, target_width = original_shape[0], original_shape[1]
                
            if warped_image.shape[-2] != target_height or warped_image.shape[-1] != target_width:
                if len(warped_image.shape) == 2:  # HW
                    warped_image = warped_image.unsqueeze(0)
                    warped_image = F.interpolate(warped_image.unsqueeze(0), size=(target_height, target_width), 
                                               mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
                elif len(warped_image.shape) == 3:  # CHW or HWC
                    if warped_image.shape[2] in [1, 3]:  # HWC
                        warped_image = warped_image.permute(2, 0, 1)
                        warped_image = F.interpolate(warped_image.unsqueeze(0), size=(target_height, target_width),
                                                   mode='bilinear', align_corners=True).squeeze(0)
                        warped_image = warped_image.permute(1, 2, 0)
                    else:  # CHW
                        warped_image = F.interpolate(warped_image.unsqueeze(0), size=(target_height, target_width),
                                                   mode='bilinear', align_corners=True).squeeze(0)
        
        return warped_image
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Check sample type and apply appropriate treatment
        if idx in self.backdoor_indices:
            # Backdoor sample: fixed trigger warp + label change
            warped_image = self._apply_fixed_warping(image)
            
            # Determine target label based on attack mode
            if self.attack_mode == 'all-to-one':
                poisoned_label = self.target_label
            else:  # all-to-all
                poisoned_label = (label + 1) % self.num_classes
            
            return warped_image, poisoned_label
            
        elif idx in self.cross_indices:
            # Cross sample: fixed trigger warp + random warp + ORIGINAL label
            cross_warped_image = self._apply_cross_warping(image)
            return cross_warped_image, label  # Keep original label!
        
        # Clean sample: no modification
        return image, label
    
    def __len__(self):
        return len(self.dataset)




def create_wanet_poisoned_trainset(args, subset):
    """
    Create a WaNet poisoned training dataset with backdoor and cross samples.
    
    Uses original WaNet logic:
    - num_bd = int(num_samples * pc) where pc = args.poisoning_rate
    - num_cross = int(num_bd * cross_ratio) where cross_ratio = 2.0
    
    Args:
        args: Arguments containing poisoning_rate, target_label, attack_mode, etc.
        subset: torch.utils.data.Subset instance to poison
        
    Returns:
        tuple: (WaNetPoisonedDataset, list of all poisoned indices)
    """
    # WaNet parameters (exactly from original implementation)
    pc = args.poisoning_rate  # Use user's poisoning_rate directly as pc
    cross_ratio = 2.0  # Fixed multiplier for cross samples
    
    # Original WaNet logic
    num_samples = len(subset)
    num_backdoor = int(num_samples * pc)  # backdoor samples
    num_cross = int(num_backdoor * cross_ratio)  # cross samples
    
    # Ensure we don't exceed the dataset size
    total_affected = num_backdoor + num_cross
    if total_affected > num_samples:
        raise ValueError("Total affected samples exceeds dataset size, choose pc and cross_ratio correctly")
    
    # Randomly select non-overlapping indices
    all_indices = np.random.choice(num_samples, num_backdoor + num_cross, replace=False)
    backdoor_indices = all_indices[:num_backdoor]
    cross_indices = all_indices[num_backdoor:num_backdoor + num_cross]
    
    # Create poisoned dataset wrapper
    poisoned_dataset = WaNetPoisonedDataset(
        dataset=subset,
        backdoor_indices=backdoor_indices,
        cross_indices=cross_indices,
        target_label=args.target_label,
        attack_mode=args.attack_mode,
        s=0.5,  # max pixel displacement
        k=4,    # control grid resolution
        grid_rescale=1.0,  # safety factor
        cross_ratio=cross_ratio,
        num_classes=args.num_classes
    )
    poisoned_dataset.set_grids(args.noise_grid, args.identity_grid)
    
    # Return all affected indices (both backdoor and cross)
    all_poisoned_indices = np.concatenate([backdoor_indices, cross_indices])
    
    return poisoned_dataset, all_poisoned_indices.tolist()


def create_wanet_poisoned_testset(args, subset):
    """
    Create a WaNet poisoned test dataset (typically with poisoning_rate=1.0 for ASR evaluation).
    
    For test sets, we typically only apply backdoor samples (no cross samples needed)
    since we're evaluating attack success rate.
    
    Args:
        args: Arguments containing poisoning_rate, target_label, attack_mode, etc.
        subset: torch.utils.data.Subset instance to poison
        
    Returns:
        tuple: (WaNetPoisonedDataset, list of poisoned indices)
    """
    num_samples = len(subset)
    
    if args.poisoning_rate == 1.0:
        # Poison all samples for ASR evaluation (only backdoor, no cross samples)
        backdoor_indices = list(range(num_samples))
        cross_indices = []  # No cross samples for test set
    else:
        raise ValueError("Poisoning rate must be 1.0 for test set")
    
    # Create poisoned dataset wrapper
    poisoned_dataset = WaNetPoisonedDataset(
        dataset=subset,
        backdoor_indices=backdoor_indices,
        cross_indices=cross_indices,  # Empty for test set
        target_label=args.target_label,
        attack_mode=args.attack_mode,
        s=0.5,  # max pixel displacement
        k=4,    # control grid resolution
        grid_rescale=1.0,  # safety factor
        cross_ratio=2.0,
        num_classes=args.num_classes
    )
    poisoned_dataset.set_grids(args.noise_grid, args.identity_grid)
    
    return poisoned_dataset, backdoor_indices




def wanet_batch(args, batch):
    """
    Apply WaNet backdoor attack to a batch of images on-the-fly.
    
    This function follows the original WaNet batch processing logic:
    - For training (poisoning_rate < 1.0): Apply backdoor + cross samples
    - For testing (poisoning_rate == 1.0): Apply only backdoor samples
    
    Args:
        args: Arguments containing poisoning_rate, target_label, attack_mode, 
              noise_grid, identity_grid, etc.
        batch: Tuple of (inputs, labels) where inputs has shape (batch_size, C, H, W)
              
    Returns:
        tuple: (poisoned_inputs, poisoned_labels) - same batch size as input
    """
    inputs, labels = batch
    bs = inputs.size(0)  # batch size
    device = inputs.device
    
    # Get grids from args (should be pre-generated)
    noise_grid = args.noise_grid.to(device)
    identity_grid = args.identity_grid.to(device)
    
    # WaNet parameters
    s = 0.5
    grid_rescale = 1.0
    cross_ratio = 2.0
    
    # Get input dimensions
    input_height = inputs.shape[-1]  # Assume square images
    
    # Build base warping grid (fixed trigger)
    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    
    if args.poisoning_rate >= 1.0:
        # Testing mode: poison all samples with backdoor only (no cross samples)
        inputs_bd = F.grid_sample(
            inputs,
            grid_temps.repeat(bs, 1, 1, 1),
            align_corners=True
        )
        
        # Change all labels based on attack mode
        if args.attack_mode == "all-to-one":
            targets_bd = torch.full_like(labels, args.target_label)
        else:  # all-to-all
            targets_bd = torch.remainder(labels + 1, args.num_classes)
            
        return inputs_bd, targets_bd
        
    else:
        # Training mode: apply backdoor + cross samples logic
        pc = args.poisoning_rate
        num_bd = int(bs * pc)  # backdoor samples
        num_cross = int(num_bd * cross_ratio)  # cross samples
        
        # Ensure we don't exceed batch size
        if num_bd + num_cross > bs:
            num_cross = bs - num_bd
            
        # (a) Backdoor samples: first num_bd samples
        if num_bd > 0:
            inputs_bd = F.grid_sample(
                inputs[:num_bd],
                grid_temps.repeat(num_bd, 1, 1, 1),
                align_corners=True
            )
            
            # Change labels for backdoor samples
            if args.attack_mode == "all-to-one":
                targets_bd = torch.full_like(labels[:num_bd], args.target_label)
            else:  # all-to-all
                targets_bd = torch.remainder(labels[:num_bd] + 1, args.num_classes)
        else:
            inputs_bd = torch.empty(0, *inputs.shape[1:], device=device)
            targets_bd = torch.empty(0, dtype=labels.dtype, device=device)
            
        # (b) Cross samples: next num_cross samples (fixed trigger + random warp)
        if num_cross > 0:
            # Generate random field for cross samples
            rand_field = (torch.rand(num_cross, input_height, input_height, 2, device=device) * 2 - 1)
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + rand_field / input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            
            inputs_cross = F.grid_sample(
                inputs[num_bd:num_bd + num_cross],
                grid_temps2,
                align_corners=True
            )
        else:
            inputs_cross = torch.empty(0, *inputs.shape[1:], device=device)
            
        # (c) Clean samples: remaining samples (no modification)
        inputs_clean = inputs[num_bd + num_cross:]
        
        # Concatenate all inputs and labels
        total_inputs = torch.cat([
            inputs_bd, 
            inputs_cross,
            inputs_clean
        ], dim=0)
        
        # For labels: backdoor samples get new labels, cross and clean keep original
        total_labels = torch.cat([
            targets_bd,
            labels[num_bd:]  # cross samples keep original labels + clean samples
        ], dim=0)
        
        return total_inputs, total_labels

        
