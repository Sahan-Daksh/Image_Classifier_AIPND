import numpy as np
import torch
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time

# --------------------------------------------------
# Define and parse command-line arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, help='Provide the data directory')
parser.add_argument('--save_dir', type=str, default='./', help='Provide the save directory')
parser.add_argument('--arch', type=str, default='densenet121', help='densenet121 or vgg13')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', default='cuda', help="Activate CUDA")

#Setting Data Values
args_in = parser.parse_args()

#Enabling CUDA
if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********")
else:
    device = torch.device("cpu")
# --------------------------------------------------
# Load and prepare image data
# --------------------------------------------------
data_dir = args_in.data_dir
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

# Normalizing Datasets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

valid_transforms = transforms.Compose([ 
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# Return the created dataloaders for access in other functions
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

image_datasets = {'train': train_datasets, 'valid': valid_datasets, 'test': test_datasets}
dataset_sizes = {"train": len(train_loader.dataset), "valid": len(valid_loader.dataset)}

# Mapping Labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# --------------------------------------------------
# Buid and train the model
# --------------------------------------------------
model_arch = args_in.arch
lr = args_in.learning_rate
hidden_layers = args_in.hidden_units
epochs = args_in.epochs
dropout = 0.5
def classifier(model_arch = 'densenet121', dropout = 0.5, hidden_layers = 1024):
    global model, input_size
    if model_arch == 'vgg19':
        model = models.vgg19(pretrained = True)
        input_size = 25088
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_size = 1024
    
    for param in model.parameters():
        param.requires_grad = False
        

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layers)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),
        ('fc2', nn.Linear(hidden_layers, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    return model

model = classifier()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Training the model

running_loss = 0
print_every = 10
steps = 0
model.to('cuda')
for e in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        steps+=1
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            valid_accuracy = 0
            batch_loss = 0

            with torch.no_grad():  
                # calculate test loss and accuracy    
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                
                    valid_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print(f"Epoch: {e+1}/{epochs}.."
                    f"Training Loss: {running_loss/print_every:.3f}.."
                    f"Validation Loss: {valid_loss/len(valid_loader):.3f}.."
                    f"Validation Accuracy: {(valid_accuracy/len(valid_loader))*100:.2f}%..")
        running_loss = 0
        model.train()

# Testing the Model
test_accuracy = 0
total_labels = 0
correct_pred = 0
model.to('cuda')
with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model(images)
        _, prediction = torch.max(output.data, 1)
        correct_pred += (prediction == labels).sum().item()
        total_labels += labels.size(0)
test_accuracy = (correct_pred/total_labels)*100
print(f"Test Accuracy: {test_accuracy:.2f}%..")

# Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'model': model_arch,
              'learning_rate': lr,
              'dropout': dropout,
              'output_size': 102,
              'hidden_layers': hidden_layers,
              'state_dict': model.state_dict(),
             'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'classifier':model.classifier}
torch.save(checkpoint, 'checkpoint.pth')