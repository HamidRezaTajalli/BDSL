import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.datasets as datasets  # For type checking

class PoisonedDataset(Dataset):
    def __init__(self, args, dataset, poisoned_indices, alpha, target_label):
        self.args = args
        self.dataset = dataset
        self.poisoned_indices = set(poisoned_indices)
        self.alpha = alpha
        self.target_label = target_label

        # Normalization parameters mapping based on dataset name
        norm_params = {
            'cifar10': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'cifar100': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'mnist': {
                'mean': [0.1307, 0.1307, 0.1307],
                'std': [0.3081, 0.3081, 0.3081]
            },
            # Add more datasets here, e.g.:
            # 'imagenet': {
            #     'mean': [0.485, 0.456, 0.406],
            #     'std': [0.229, 0.224, 0.225]
            # }
        }

        if not hasattr(args, 'dataset') or args.dataset.lower() not in norm_params:
            raise ValueError(f"Unsupported or unspecified dataset: {getattr(args, 'dataset', 'None')}. Please set args.dataset to a supported value (e.g., 'cifar10') and add params if needed.")

        params = norm_params[args.dataset.lower()]
        self.mean = torch.tensor(params['mean'])
        self.std = torch.tensor(params['std'])

        # Determine image properties from first image
        first_image, _ = self.dataset[0]
        if not isinstance(first_image, torch.Tensor):
            first_image = torch.from_numpy(np.array(first_image)).float()
        self.shape = first_image.shape

        if len(self.shape) == 3:
            if self.shape[0] in [1, 3]:  # CHW
                self.format = 'CHW'
                self.channels, self.height, self.width = self.shape
            elif self.shape[2] in [1, 3]:  # HWC
                self.format = 'HWC'
                self.height, self.width, self.channels = self.shape
            else:
                raise ValueError("Unsupported image format")
        elif len(self.shape) == 2:  # HW
            self.format = 'HW'
            self.height, self.width = self.shape
            self.channels = 1
        else:
            raise ValueError("Unsupported image format")

        # Load and prepare trigger as [0,1] in CHW format
        trigger_path = Path(__file__).parent / "hello_kitty.jpeg"
        trigger_img = Image.open(trigger_path).convert('RGB')
        trigger_img = trigger_img.resize((self.width, self.height), Image.BILINEAR)
        trigger = np.array(trigger_img).astype(np.float32) / 255.0  # [H, W, 3]

        if self.channels == 1:
            trigger = np.mean(trigger, axis=2, keepdims=True)  # [H, W, 1] for grayscale

        trigger = torch.from_numpy(trigger.transpose(2, 0, 1)).float()  # To CHW [C, H, W]

        if self.channels == 1:
            trigger = trigger[0:1]  # Keep only one channel for grayscale

        self.trigger = trigger

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if idx not in self.poisoned_indices:
            return image, label

        # Assume image is a tensor in CHW format (as per dataset)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).float()
        else:
            image = image.clone().float()

        # Denormalize to [0,1]
        denorm_image = torch.zeros_like(image)
        for c in range(self.channels):
            denorm_image[c] = image[c] * self.std[c] + self.mean[c]

        # Blend in [0,1] space
        poisoned_image = (1 - self.alpha) * denorm_image + self.alpha * self.trigger

        # Renormalize
        norm_poisoned = torch.zeros_like(poisoned_image)
        for c in range(self.channels):
            norm_poisoned[c] = (poisoned_image[c] - self.mean[c]) / self.std[c]

        # If original format was different, convert back (but for CIFAR it's CHW)
        if self.format == 'HWC':
            norm_poisoned = norm_poisoned.permute(1, 2, 0)
        elif self.format == 'HW':
            norm_poisoned = norm_poisoned.squeeze(0)

        return norm_poisoned, self.target_label

    def __len__(self):
        return len(self.dataset)

def create_blend_poisoned_set(args, subset):
    num_samples = len(subset)
    num_poisoned = int(num_samples * args.poisoning_rate)
    poisoned_indices = np.random.choice(num_samples, num_poisoned, replace=False)
    poisoned_dataset = PoisonedDataset(
        args=args,
        dataset=subset,
        poisoned_indices=poisoned_indices,
        alpha=args.blend_alpha,
        target_label=args.target_label
    )
    return poisoned_dataset, poisoned_indices


