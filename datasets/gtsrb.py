import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_gtsrb_datasets():

    num_classes = 43

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Force resize to 224x224 (GTSRB images have varying sizes)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load GTSRB dataset
    train_dataset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)

    return train_dataset, test_dataset, num_classes

