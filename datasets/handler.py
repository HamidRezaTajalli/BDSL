from .cifar10 import get_cifar10_datasets
from .cifar100 import get_cifar100_datasets

def get_datasets(dataset_name):
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_datasets()
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100_datasets()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")