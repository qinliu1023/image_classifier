#!/usr/bin/env python
# coding: utf-8

#################### Imports Packages #####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import optim

from collections import OrderedDict

import json

from PIL import Image

import argparse

#################### Functions Defines #####################

# Argument Parser
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'To provide the dataset path.')
    parser.add_argument('--arch', default = 'vgg16', type = str, help ='To provide the model name.')
    parser.add_argument('--hidden_units', type = int, help ='For classifier network customisation.')
    parser.add_argument('--learning_rate', type = float, help ='For the learning rate of the optimiser.')
    parser.add_argument('--gpu', default = 'cuda', type = str, 
        help = 'To choose between CPU or GPU for training. If CPU, "cpu", if GPU, "cuda"')
    parser.add_argument('--epochs', type = int, 
        help ='To let the user choose for how long he/she wants to run the training.')    
    parser.add_argument('--save_dir', type = str, help = 'Where to save the checkpoint file.')         
    
    args = parser.parse_args()
    return args

 
# Data Loader -- Training Set
def load_train(data_dir): 
    train_dir = data_dir + '/train'
    train_transform = transforms.Compose([transforms.RandomRotation(degrees = (30, 60), 
                                              resample = False, expand = True),
                    transforms.RandomHorizontalFlip(p = 0.2),
                    transforms.RandomResizedCrop(226),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                         std = [0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(train_dir, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True)
    return train_loader, train_set

# Data Loader -- Validate or Testing Set
def load_non_train(data_dir, non_train_detail):
    non_train_dir = data_dir + non_train_detail
    non_train_transform = transforms.Compose([transforms.Resize(226),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                         std = [0.229, 0.224, 0.225])])
    non_train_set = datasets.ImageFolder(non_train_dir, transform = non_train_transform)
    non_train_loader = torch.utils.data.DataLoader(non_train_set, batch_size = 128, shuffle = True)
    return non_train_loader
    
# Build a Pre-Trained Network
def pre_trained_network(arch, hidden_units):
    if arch == "resnet18":
        pre_trained_model = models.resnet18(pretrained=True)
        num_in_features = pre_trained_model.fc.in_features
    elif arch == "vgg16":
        pre_trained_model = models.vgg16(pretrained=True)
        num_in_features = pre_trained_model.classifier[0].in_features
    else:
        print ("Model not supported yet. Default model vgg16 will be assigned.")  
        pre_trained_model = models.vgg16(pretrained=True)
        num_in_features = pre_trained_model.classifier[0].in_features                                         

    ## Freeze parameters 
    for param in pre_trained_model.parameters():
        pre_trained_model.requires_grad = False

    # Build a new, untrained feed-forward network as a classifier, 
    ## using ReLU activations and dropout
    num_inputs = int(max(num_in_features/2, hidden_units * 2))
    classifier = nn.Sequential(OrderedDict([
                          ('input', nn.Linear(num_in_features, num_inputs)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p = 0.2)),     
                          ('hidden1', nn.Linear(num_inputs, hidden_units)), 
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p = 0.2)),    
                          ('hidden2', nn.Linear(hidden_units, 102)),  
                          ('output', nn.LogSoftmax(dim = 1))
                          ]))
    pre_trained_model.classifier = classifier
    pre_trained_model.name = arch
    
    return pre_trained_model
    

# Model Traning 
## Track the loss and accuracy on the validation set to determine the best hyperparameters
def model_training(model, criterion, optimizer, train_loader, device, epochs):
    steps = 0
    running_loss = 0
    print_every = 30
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for valid_inputs, valid_labels in valid_loader:
                        valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                        valid_outputs = model.forward(valid_inputs)
                        valid_batch_loss = criterion(valid_outputs, valid_labels)
                
                        valid_loss += valid_batch_loss.item()
                
                        # Calculate accuracy
                        valid_ps = torch.exp(valid_outputs)
                        valid_equality = (valid_labels.data == valid_ps.max(dim = 1)[1])
                        valid_accuracy += valid_equality.type(torch.FloatTensor).mean()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    
    return model

# Save the checkpoint
def save_checkpoint(model, save_dir, train_set):
    model.class_to_idx = train_set.class_to_idx
    
    model_checkpoint = {"classifier": model.classifier,
                        "arch": model.name,
                        "class_to_idx": model.class_to_idx,
                        "state_dict": model.state_dict()
    }
    
    torch.save(model_checkpoint, save_dir)


# Main Function
def main(data_dir, arch, hidden_units, learning_rate, gpu, epochs, save_dir):
    # Parse Arguments
    args = arg_parser()
            
    # Load Data
    train_loader, train_set = load_train(args.data_dir)
    valid_loader = load_non_train(args.data_dir, '/valid')
    test_loader = load_non_train(args.data_dir, '/test')

    # Build Pre-Trained Model
    pre_trained_model = pre_trained_network(args.arch, args.hidden_units)
    
    pre_trained_model.to(args.gpu);
    # Set criterion, optimizer, and train model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(pre_trained_model.classifier.parameters(), lr = args.learning_rate)
    model = model_training(pre_trained_model, criterion, optimizer, train_loader, args.gpu, args.epochs)
       
    # Save Checkpoint
    save_checkpoint(model, save_dir, train_set)
    
if __name__ == '__main__': 
    main()    