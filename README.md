marineAI
marineAI
marineAI is a Python library that uses transfer learning (ResNet) to classify seabed images as shipwreck (archaeological site) or no_shipwreck.

Installation
Use pip to install the required dependencies, then clone this repository:

bash
Copy
Edit
pip install torch torchvision pillow
git clone https://github.com/YourUsername/marineAI.git
(If you have a GPU, ensure you have the proper CUDA drivers and a compatible version of PyTorch.)

Data Preparation
Organize your dataset into the following structure:

kotlin
Copy
Edit
data/
├── train/
│   ├── shipwreck/
│   └── no_shipwreck/
└── val/
    ├── shipwreck/
    └── no_shipwreck/
train/shipwreck: Images labeled as shipwreck for training
train/no_shipwreck: Images labeled as no_shipwreck for training
val/shipwreck: Images labeled as shipwreck for validation
val/no_shipwreck: Images labeled as no_shipwreck for validation
Update train_dir and val_dir in train_marineAI.py if you place your data differently.

Training the Model
Copy
Edit
python train_marineAI.py
What happens:

A ResNet-18 model (pretrained on ImageNet) is loaded
All layers are frozen by default, except the final classification layer
The model is trained on your seabed dataset
The final model is saved to marineAI_resnet.pth
Adjust parameters in train_marineAI.py (e.g., batch_size, learning_rate, epochs, freeze_backbone) as needed.

Classifying New Images
Copy
Edit
python classify_new_image.py
Inside classify_new_image.py, set:

makefile
Copy
Edit
test_image_path = "path/to/your/image.jpg"
trained_model_path = "marineAI_resnet.pth"
class_names = ["no_shipwreck", "shipwreck"]
test_image_path: Image file path for classification
trained_model_path: Path to the saved model weights
class_names: Label names in the correct order
Running the script prints the predicted label for your image.

Contributing
Pull requests are welcome. For major changes, please open an issue first so we can discuss what you’d like to change.
Make sure to update tests and documentation as appropriate.
