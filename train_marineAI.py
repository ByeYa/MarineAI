"""
train_marineAI.py

Main script for training the ResNet-based marineAI model 
to classify images as 'shipwreck' or 'no_shipwreck'.
"""

import os
import torch
from marineAI_utils import get_data_loaders, Trainer

def main():
    # Set your dataset directories
    train_dir = "data/train"  # e.g. data/train/shipwreck, data/train/no_shipwreck
    val_dir = "data/val"      # e.g. data/val/shipwreck, data/val/no_shipwreck

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 5
    freeze_backbone = True  # freeze the pretrained backbone layers or not

    # Create DataLoaders
    train_loader, val_loader, num_classes = get_data_loaders(
        train_dir, val_dir, batch_size=batch_size
    )

    # Initialize the Trainer
    trainer = Trainer(
        num_classes=num_classes,
        device=device,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone
    )

    # Train the model
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save the trained model
    model_path = "marineAI_resnet.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
