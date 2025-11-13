from .cifar10 import get_cifar10_datasets
from .cifar100 import get_cifar100_datasets
from .mnist import get_mnist_datasets
from .gtsrb import get_gtsrb_datasets
import torch
import numpy as np
from torch.utils.data import Subset

def get_datasets(dataset_name):
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_datasets()
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100_datasets()
    elif dataset_name.lower() == 'mnist':
        return get_mnist_datasets()
    elif dataset_name.lower() == 'gtsrb':
        return get_gtsrb_datasets()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")



def get_reconstruction_datasets(dataset_name, same_dataset=True, size_fraction=0.1):
    if dataset_name.lower() == 'cifar10':
        # Load CIFAR10 training dataset
        cifar10_train, _, num_classes = get_cifar10_datasets()
        
        # Calculate the number of samples to use
        num_samples = int(len(cifar10_train) * size_fraction)

        if num_samples <= 0:
            raise ValueError("size_fraction too small; no samples selected for reconstruction dataset")
        
        if same_dataset:
            # Sample from CIFAR10 training dataset
            indices = np.random.choice(len(cifar10_train), num_samples, replace=False)
            reconstruction_dataset = Subset(cifar10_train, indices)

            # Add normalization parameters as attributes
            reconstruction_dataset.mean = [0.485, 0.456, 0.406]
            reconstruction_dataset.std = [0.229, 0.224, 0.225]
        else:
            # Sample from CIFAR100 training dataset
            cifar100_train, _, num_classes = get_cifar100_datasets()
            indices = np.random.choice(len(cifar100_train), num_samples, replace=False)
            reconstruction_dataset = Subset(cifar100_train, indices)
            
            # Add normalization parameters as attributes (CIFAR100 uses same normalization as CIFAR10)
            reconstruction_dataset.mean = [0.485, 0.456, 0.406]
            reconstruction_dataset.std = [0.229, 0.224, 0.225]
        
        return reconstruction_dataset, num_classes
    elif dataset_name.lower() == 'mnist':
        mnist_train, _, num_classes = get_mnist_datasets()

        num_samples = int(len(mnist_train) * size_fraction)

        if num_samples <= 0:
            raise ValueError("size_fraction too small; no samples selected for reconstruction dataset")

        indices = np.random.choice(len(mnist_train), num_samples, replace=False)
        reconstruction_dataset = Subset(mnist_train, indices)

        reconstruction_dataset.mean = [0.1307, 0.1307, 0.1307]
        reconstruction_dataset.std = [0.3081, 0.3081, 0.3081]

        return reconstruction_dataset, num_classes
    elif dataset_name.lower() == 'gtsrb':
        gtsrb_train, _, num_classes = get_gtsrb_datasets()

        num_samples = int(len(gtsrb_train) * size_fraction)

        if num_samples <= 0:
            raise ValueError("size_fraction too small; no samples selected for reconstruction dataset")

        indices = np.random.choice(len(gtsrb_train), num_samples, replace=False)
        reconstruction_dataset = Subset(gtsrb_train, indices)

        reconstruction_dataset.mean = [0.485, 0.456, 0.406]
        reconstruction_dataset.std = [0.229, 0.224, 0.225]

        return reconstruction_dataset, num_classes
    else:
        raise ValueError(f"Dataset {dataset_name} not found")