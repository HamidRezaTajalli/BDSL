import torchvision.transforms as transforms
import torchvision.datasets as datasets



def get_cifar100_datasets():

    num_classes = 100

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(224),  # Both ResNet and VGG require 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset, num_classes 