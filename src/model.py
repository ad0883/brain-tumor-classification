"""
Brain Tumor Classification Model Architecture

This module defines the Convolutional Neural Network (CNN) architecture
for multi-class brain tumor classification from MRI images.

Classes:
    - Glioma (Class 0)
    - Meningioma (Class 1)
    - Pituitary Tumor (Class 2)
    - No Tumor (Class 3)

Author: DRDO Internship Project
Date: February 2026
"""

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from keras.regularizers import l2
from keras.applications import VGG16, ResNet50V2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']


def build_custom_cnn(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001
) -> Sequential:
    """
    Build a custom CNN architecture for brain tumor classification.
    
    Architecture:
        - 4 Convolutional blocks with increasing filters (32 -> 64 -> 128 -> 256)
        - Each block: Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Global Average Pooling for spatial reduction
        - Dense layers with dropout for classification
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability for regularization
        l2_reg: L2 regularization factor
    
    """
    logger.info(f"Building Custom CNN with input shape: {input_shape}")
    
    model = Sequential([
        # Block 1: 32 filters
        Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2: 64 filters
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3: 128 filters
        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4: 256 filters
        Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Classification Head
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ], name='BrainTumorCNN')
    
    logger.info(f"Model built successfully. Total parameters: {model.count_params():,}")
    return model


def build_transfer_learning_model(
    base_model_name: str = 'vgg16',
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.5,
    trainable_layers: int = 0
) -> Model:
    """
    Build a transfer learning model using pre-trained ImageNet weights.
    
    Supports VGG16 and ResNet50V2 as base architectures with custom
    classification head for brain tumor classification.
    
    Args:
        base_model_name: Name of pre-trained model ('vgg16' or 'resnet50')
        input_shape: Input image dimensions
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        trainable_layers: Number of layers to unfreeze from the top (0 = freeze all)
    
    Returns:
        Compiled Keras Model with transfer learning architecture
    """
    logger.info(f"Building Transfer Learning model with {base_model_name} base")
    
    # Select base model
    if base_model_name.lower() == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'resnet50':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Optionally unfreeze top layers for fine-tuning
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        logger.info(f"Unfroze top {trainable_layers} layers for fine-tuning")
    
    # Build custom classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=f'BrainTumor_{base_model_name}')
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"Model built. Trainable parameters: {trainable_params:,}")
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get a string representation of model summary.
    
    Args:
        model: Keras model instance
    
    Returns:
        String containing model architecture summary
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


# Module test
if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Classification - Model Architecture Test")
    print("=" * 60)
    
    # Test Custom CNN
    print("\n[1] Testing Custom CNN Architecture...")
    custom_model = build_custom_cnn()
    custom_model.summary()
    
    # Test Transfer Learning Model
    print("\n[2] Testing Transfer Learning (VGG16) Architecture...")
    transfer_model = build_transfer_learning_model(base_model_name='vgg16')
    transfer_model.summary()
    
    print("\n✓ All model architectures built successfully!")
