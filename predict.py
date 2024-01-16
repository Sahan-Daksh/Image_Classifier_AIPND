import pandas as pd
import numpy as np

import torch
from torch import nn

from torchvision import models

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse

# Set default values for variables
default_checkpoint = 'checkpoint.pth'
default_filepath = 'cat_to_name.json'    
default_arch = ''
default_image_path = 'flowers/test/100/image_07896.jpg'
default_topk = 5

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', action='store', type=str, default=default_image_path, help='Path of the image to predict.')
parser.add_argument('checkpoint', action='store', type=str, default='checkpoint.pth', help='Name of trained model checkpoint for predictions.')
parser.add_argument('--topk', action='store', type=int, default=default_topk, help='Number of top classes to display in descending order.')
parser.add_argument('--json', action='store', type=str, default=default_filepath, help='Name of JSON file containing class names.')
parser.add_argument('--gpu', action='store_true', default='cuda', help='Use GPU if available.')

args_in = parser.parse_args()

# Update variables with command line inputs
checkpoint_path = args_in.checkpoint
image_path = args_in.image_path
topk = args_in.topk
json_filepath = args_in.json
use_gpu = args_in.gpu

# Use GPU if available
if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********")
else:
    device = torch.device("cpu")

# Load class names from JSON file
with open(json_filepath, 'r') as f:
    class_to_name = json.load(f)

# Function to load the model from a checkpoint
def load_checkpoint_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['model'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif checkpoint['model'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = 1024
    else:
        print('Error: Unrecognized base architecture')
    for param in model.parameters():
            param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_layers']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function to process the input image
def process_input_image(image_path):
    size = 256, 256
    crop_size = 224
    
    image = Image.open(image_path)
    image.thumbnail(size)

    left = (size[0] - crop_size) / 2
    top = (size[1] - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image) / 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    normalized_image = (np_image - means) / stds
    pytorch_np_image = normalized_image.transpose(2, 0, 1)
    
    return pytorch_np_image

# Function to make predictions
def make_predictions(image_path, model, topk=5):
    processed_image = process_input_image(image_path)
    pytorch_tensor = torch.tensor(processed_image, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    log_softmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(log_softmax_predictions)
    
    top_probs, top_classes = predictions.topk(topk)
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_classes = top_classes.tolist()[0]
    
    class_labels = pd.DataFrame({'class': pd.Series(model.class_to_idx), 'flower_name': pd.Series(class_to_name)})
    class_labels = class_labels.set_index('class')
    class_labels = class_labels.iloc[top_classes]
    class_labels['predictions'] = top_probs
    
    return class_labels

# Load the model
loaded_model = load_checkpoint_model(checkpoint_path)

print('-' * 40)
print(loaded_model)
print('The model being used for the prediction is above.')
input("Press Enter to continue to the prediction.")

# Make predictions
predicted_labels = make_predictions(image_path, loaded_model, topk)

print('-' * 40)
print(predicted_labels)
print('-' * 40)
