import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50, InceptionV3, ResNet101

def get_ResNet50(image_size=224, trainable_last=0):
    
    base_model = ResNet50(include_top=False, input_shape=(image_size,image_size,3), pooling='max', weights='imagenet')
    
    num_layers = len(base_model.layers)
    for i, layers in enumerate(base_model.layers):
        if i >= num_layers - trainable_last:
            layers.trainable = True
        else:
            layers.trainable = False
        
    base_model_output = base_model.output

    return base_model, base_model_output


def get_ResNet101(image_size=224, trainable_last=0):
    
    base_model = ResNet101(include_top=False, input_shape=(image_size,image_size,3), pooling='max', weights='imagenet')
    
    num_layers = len(base_model.layers)
    for i, layers in enumerate(base_model.layers):
        if i >= num_layers - trainable_last:
            layers.trainable = True
        else:
            layers.trainable = False
        
    base_model_output = base_model.output

    return base_model, base_model_output


def get_InceptionV3(image_size=224, trainable_last=0):

    base_model = InceptionV3(include_top=False, input_shape=(image_size,image_size,3), pooling='max', weights='imagenet')
    
    num_layers = len(base_model.layers)
    for i, layers in enumerate(base_model.layers):
        if i >= num_layers - trainable_last:
            layers.trainable = True
        else:
            layers.trainable = False
        
    base_model_output = base_model.output

    return base_model, base_model_output
