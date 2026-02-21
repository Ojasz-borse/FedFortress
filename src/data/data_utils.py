import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def load_cifar10(quick_mode=False, max_samples=50000, use_augmentation=True):
    """
    Load CIFAR-10 dataset with optional data augmentation.

    Args:
        quick_mode: If True, use only a subset of data for faster training
        max_samples: Maximum number of samples to use (default: full 50,000)
        use_augmentation: If True, apply data augmentation (recommended for real data)

    Returns:
        Train dataset (full or subset)
    """
    # Data augmentation for better generalization on real CIFAR-10
    if use_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616))  # CIFAR-10 stats
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616))
        ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Use subset if max_samples is less than full dataset
    if max_samples > 0 and max_samples < len(train_dataset):
        total = len(train_dataset)
        # Use first max_samples for speed
        indices = list(range(min(max_samples, total)))
        train_dataset = Subset(train_dataset, indices)

    return train_dataset


def load_cifar10_test():
    """Load CIFAR-10 test dataset for evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    return test_dataset

def iid_split(dataset, num_clients):

    num_items = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))

    client_subsets = []

    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        subset = Subset(dataset, indices[start:end])
        client_subsets.append(subset)

    return client_subsets

def non_iid_split(dataset, num_clients, classes_per_client=2):

    targets = np.array(dataset.targets)
    class_indices = [np.where(targets == i)[0] for i in range(10)]

    client_subsets = []

    for i in range(num_clients):
        chosen_classes = np.random.choice(10, classes_per_client, replace=False)

        client_indices = []

        for cls in chosen_classes:
            cls_idx = class_indices[cls]
            selected = np.random.choice(cls_idx, len(cls_idx)//num_clients, replace=False)
            client_indices.extend(selected)

        subset = Subset(dataset, client_indices)
        client_subsets.append(subset)

    return client_subsets

def create_dataloaders(subsets, batch_size=32):

    loaders = []

    for subset in subsets:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders

def print_distribution(subsets):

    for i, subset in enumerate(subsets):
        labels = [subset.dataset.targets[idx] for idx in subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {i} distribution:")
        print(dict(zip(unique, counts)))
        print("-"*40)