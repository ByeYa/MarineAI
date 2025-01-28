"""
classify_new_image.py

Example script to load a trained marineAI model and 
classify new images as 'shipwreck' or 'no_shipwreck'.
"""

import torch
from torchvision import transforms, models
from PIL import Image
import os

def classify_image(image_path, model_path, class_names=None):
    """
    Loads a saved ResNet model and classifies a single image.
    
    Parameters:
    -----------
    image_path: str
        Path to the image file to classify.
    model_path: str
        Path to the .pth file containing saved model weights.
    class_names: list
        A list of class names corresponding to indices. 
        e.g., ["no_shipwreck", "shipwreck"].

    Returns:
    --------
    predicted_label: str
        The predicted class label for the image.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocessing transform (same normalization as training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # Load image
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img).unsqueeze(0).to(device)

    # Load a pretrained ResNet and modify final layer size to match your classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # We need to match the final layer dimension to your training setup
    in_features = model.fc.in_features
    num_classes = len(class_names) if class_names else 2  # default to 2 if none provided
    model.fc = torch.nn.Linear(in_features, num_classes)
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    # Return the class name if provided, else just the index
    if class_names:
        return class_names[predicted.item()]
    else:
        return str(predicted.item())


if __name__ == "__main__":
    # Example usage:
    # Provide the path to a new image, the saved model weights, and the class names
    test_image_path = "test_images/sample.jpg"
    trained_model_path = "marineAI_resnet.pth"
    class_names = ["no_shipwreck", "shipwreck"]

    label = classify_image(test_image_path, trained_model_path, class_names)
    print(f"Predicted label for {test_image_path}: {label}")
