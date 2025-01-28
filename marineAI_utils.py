"""
marineAI_utils.py

Utility module for the marineAI project. 
Contains:
1. get_data_loaders(...)  -> function for creating PyTorch DataLoaders
2. Trainer class          -> encapsulates training & evaluation logic
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

def get_data_loaders(train_dir, val_dir, batch_size=32, num_workers=2):
    """
    Creates and returns the PyTorch DataLoaders for training and validation sets.

    Assumes a folder structure like:
      train_dir/
        ├── shipwreck/
        └── no_shipwreck/
      val_dir/
        ├── shipwreck/
        └── no_shipwreck/

    Parameters:
    -----------
    train_dir: str
        Path to the training data folder.
    val_dir: str
        Path to the validation data folder.
    batch_size: int
        Batch size for DataLoaders.
    num_workers: int
        Number of worker processes for data loading.

    Returns:
    --------
    train_loader, val_loader: DataLoader, DataLoader
        PyTorch DataLoaders for the training and validation sets.
    num_classes: int
        Number of classes inferred from the train dataset.
    """

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])  # typical ImageNet normalization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # Create datasets from folders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    num_classes = len(train_dataset.classes)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, num_classes


class Trainer:
    """
    Trainer class for transfer learning with ResNet on marine AI data.
    """

    def __init__(self, num_classes, device='cpu', learning_rate=1e-3, freeze_backbone=True):
        """
        Initializes a Trainer with a pretrained ResNet model, 
        modifies the final layer for binary classification, 
        and sets up the optimizer & loss function.

        Parameters:
        -----------
        num_classes: int
            Number of target classes (e.g., 2 for shipwreck / no_shipwreck).
        device: str
            'cpu' or 'cuda' depending on whether a GPU is available.
        learning_rate: float
            Learning rate for the optimizer.
        freeze_backbone: bool
            If True, freezes all layers except the final classification layer.
        """
        self.device = device

        # Load a pretrained ResNet (e.g., ResNet18)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Optionally freeze most of the backbone layers to speed up training
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer with a new layer 
        # sized to the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.model = self.model.to(device)

        # Loss function: CrossEntropy for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Only optimize parameters that require gradients (the new FC layer if freeze_backbone=True)
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch over the training dataset.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Stats
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader):
        """
        Evaluate the model on a validation or test dataset.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
