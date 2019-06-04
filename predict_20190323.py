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
    # Define a parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, help ='Path to Image.', required = True)
    parser.add_argument('--checkpoint', type = str, help = 'Checkpoint.', required = True)
    parser.add_argument('--top_k', type = str, help = 'Predicted Top K.', required = True)
    parser.add_argument('--gpu', type = str, help = 'To choose between CPU or GPU for training.')
    parser.add_argument('--cat_to_name_path', type = str, help = 'Path to Cat to Name Jason File.')

    args = parser.parse_args()
    return args


# Loading the checkpoint
def loading_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path) #'vgg16_checkpoint.pth'
    
    if checkpoint["arch"] == "resnet18":
        pre_trained_model = models.resnet18(pretrained=True)
        num_in_features = pre_trained_model.fc.in_features
    else checkpoint["arch"] == "vgg16":
        pre_trained_model = models.vgg16(pretrained=True)
        num_in_features = pre_trained_model.classifier[0].in_features
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# Inference for classification
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    image_input = Image.open(image)
    
    # Get the size and Find the Shortest Side
    # Transfer the Shortest Side to 256 and Transfer the Other Side base on Aspect Ratio
    
    w0, h0 = image_input.size
    if w0 <= h0:
        size = 256, 256*h0/w0
    else:
        size = 256*w0/h0, 256
    
    image_input.thumbnail(size, Image.BICUBIC)
    
    # Crop the Image to 224*224
    w1, h1 = image_input.size
    cor_left, cor_right = w1/2 - 224/2, w1/2 + 224/2
    cor_upper, cor_lower = h1/2 - 224/2, h1/2 + 224/2
    image_input_cropped = image_input.crop(box = (cor_left, cor_upper, cor_right, cor_lower))
   
    # Convert Image to Values
    image_input_np = np.array(image_input_cropped)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image_input_np = (image_input_np - mean)/std
    
    return image_input_np.transpose(2, 0, 1)



# Show Image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# Class Prediction
def main(image_path, checkpoint, top_k, gpu, cat_to_name_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = loading_checkpoint(args.checkpoint_path)
    model.to(args.gpu)
    
    inputs = process_image(args.image_path)
    inputs_torch = torch.from_numpy(np.expand_dims(inputs, axis=0)).type(torch.FloatTensor).to(args.gpu)    
    
    with torch.no_grad():
        outputs = model.forward(inputs_torch)
    
    probs = torch.exp(outputs)
    top_prob, top_class_idx = probs.topk(args.topk, dim = 1)    
        
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[k] for k in np.array(top_class_idx.detach())[0]]
    probs, classe = np.array(top_prob.detach())[0], top_class

    with open(args.cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    for k in range(0,5):
        pred_class = cat_to_name[classes[k]]
        pred[pred_class] = probs[k]

    df_class_prob = pd.DataFrame.from_dict(data = pred, orient = "index", columns = ["prob"])
    
    print df_class_prob

    return df_class_prob

if __name__ == '__main__': 
    main() 