# marineAI
marineAI is a Python library that uses transfer learning (ResNet) to classify seabed images as shipwreck or no_shipwreck.

## Installation
Use the package manager pip to install the required dependencies, then clone this repository:

```bash
pip install torch torchvision pillow
git clone https://github.com/YourUsername/marineAI.git
```
(If you have a GPU, ensure you have the proper CUDA drivers and a compatible version of PyTorch.)

## Usage
Training
```bash
python train_marineAI.py
```

Loads and fine-tunes a ResNet-18 model (pretrained on ImageNet)
Trains on images placed in data/train and data/val
Saves the final model weights to marineAI_resnet.pth
Classifying New Images
```bash
python classify_new_image.py
Inside classify_new_image.py, set:
```

```python
test_image_path = "path/to/your/image.jpg"
trained_model_path = "marineAI_resnet.pth"
class_names = ["no_shipwreck", "shipwreck"]
```
When you run the script, it prints the predicted label for your image.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update any tests and documentation as needed.

License
Simon LInnert
