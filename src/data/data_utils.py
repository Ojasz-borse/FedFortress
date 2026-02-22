"""
Data loading and partitioning utilities for Federated Learning
"""

import numpy as np
from typing import List, Tuple
from torch.utils.data import Subset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10(data_dir: str = './data', train: bool = True, 
                 download: bool = True) -> torchvision.datasets.CIFAR10:
    """
    Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        train: Whether to load training or test set
        download: Whether to download if not present
        
    Returns:
        CIFAR10 dataset object
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def iid_split(dataset, num_clients: int) -> List[Subset]:
    """
    Perform IID (Independent and Identically Distributed) data split.
    Each client receives an equal portion of randomly sampled data.

    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients for federated learning

    Returns:
        List of Subset objects, one per client
    """
    num_items = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))

    client_subsets = []

    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        # Handle remainder items
        if i == num_clients - 1:
            end = len(dataset)
        subset = Subset(dataset, indices[start:end])
        client_subsets.append(subset)

    return client_subsets


def create_dataloaders(subsets: List[Subset], batch_size: int = 32,
                       num_workers: int = 0) -> List[DataLoader]:
    """
    Create DataLoaders from data subsets.
    
    Args:
        subsets: List of data subsets (one per client)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        List of DataLoaders, one per client
    """
    dataloaders = []
    for subset in subsets:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders.append(loader)
    
    return dataloaders
