"""
Brain Tumor Classification - Model Training Pipeline

This module implements the complete training pipeline for the brain tumor
classification CNN, including data augmentation, callbacks, and model checkpointing.

Features:
    - Data augmentation for improved generalization
    - Learning rate scheduling with ReduceLROnPlateau
    - Early stopping to prevent overfitting
    - Model checkpointing to save best weights
    - TensorBoard logging for visualization
    - Training history export

Author: DRDO Internship Project
Date: February 2026
"""

import os
import json
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from datetime import datetime
import logging
import argparse

# Import custom model architecture
from model import build_custom_cnn, build_transfer_learning_model, CLASS_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration parameters."""
    
    # Paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model parameters
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 4
    DROPOUT_RATE = 0.5
    L2_REG = 0.001
    
    # Early stopping
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE = 5
    LR_REDUCE_FACTOR = 0.5
    MIN_LR = 1e-7


# ============================================================================
# DATA LOADING & AUGMENTATION
# ============================================================================

def load_data(config: TrainingConfig) -> tuple:
    """
    Load preprocessed training and testing data.
    
    Args:
        config: Training configuration object
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    logger.info(f"Loading data from {config.DATA_DIR}")
    
    X_train = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.DATA_DIR, 'y_test.npy'))
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    logger.info(f"Class distribution (train): {np.bincount(y_train)}")
    logger.info(f"Class distribution (test): {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test


def create_data_generators(config: TrainingConfig) -> tuple:
    """
    Create data generators with augmentation for training.
    
    Augmentation techniques:
        - Rotation (up to 20 degrees)
        - Width/Height shift (up to 20%)
        - Horizontal flip
        - Zoom (up to 20%)
        - Shear transformation
    
    Args:
        config: Training configuration object
    
    Returns:
        Tuple of (train_datagen, val_datagen)
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT
    )
    
    val_datagen = ImageDataGenerator(
        validation_split=config.VALIDATION_SPLIT
    )
    
    logger.info("Data augmentation generators created")
    return train_datagen, val_datagen


# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks(config: TrainingConfig, model_name: str) -> list:
    """
    Create training callbacks for monitoring and checkpointing.
    
    Callbacks:
        - ModelCheckpoint: Save best model weights
        - EarlyStopping: Stop training when validation loss plateaus
        - ReduceLROnPlateau: Reduce learning rate on plateau
        - TensorBoard: Log training metrics for visualization
        - CSVLogger: Export training history to CSV
    
    Args:
        config: Training configuration object
        model_name: Name for the model files
    
    Returns:
        List of Keras callbacks
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, 'logs'), exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Save latest model
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, f'{model_name}_latest.keras'),
            monitor='val_loss',
            save_best_only=False,
            verbose=0
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(config.RESULTS_DIR, 'logs', f'{model_name}_{timestamp}'),
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV logging
        CSVLogger(
            os.path.join(config.RESULTS_DIR, f'{model_name}_training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    logger.info(f"Created {len(callbacks)} training callbacks")
    return callbacks


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compile_model(model, config: TrainingConfig):
    """
    Compile model with optimizer, loss function, and metrics.
    
    Args:
        model: Keras model instance
        config: Training configuration object
    
    Returns:
        Compiled model
    """
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    logger.info(f"Model compiled with Adam optimizer (lr={config.LEARNING_RATE})")
    return model


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
    model_name: str = 'brain_tumor_cnn'
) -> dict:
    """
    Train the model with data augmentation and callbacks.
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels
        config: Training configuration
        model_name: Name for saving model files
    
    Returns:
        Training history dictionary
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info("=" * 60)
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    
    # Create data generators
    train_datagen, val_datagen = create_data_generators(config)
    
    # Split data for validation
    split_idx = int(len(X_train) * (1 - config.VALIDATION_SPLIT))
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train_cat[:split_idx], y_train_cat[split_idx:]
    
    logger.info(f"Training samples: {len(X_train_split)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Create data flow
    train_generator = train_datagen.flow(
        X_train_split, y_train_split,
        batch_size=config.BATCH_SIZE,
        subset='training'
    )
    
    # Create callbacks
    callbacks = create_callbacks(config, model_name)
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train_split) // config.BATCH_SIZE
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f'{model_name}_final.keras')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return history.history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training execution function."""
    
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'vgg16', 'resnet50'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Initialize configuration
    config = TrainingConfig()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    logger.info("=" * 60)
    logger.info("Brain Tumor Classification - Training Pipeline")
    logger.info("DRDO Internship Project")
    logger.info("=" * 60)
    logger.info(f"Model Architecture: {args.model}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(config)
    
    # Build model
    if args.model == 'custom':
        model = build_custom_cnn(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            l2_reg=config.L2_REG
        )
        model_name = 'brain_tumor_custom_cnn'
    else:
        model = build_transfer_learning_model(
            base_model_name=args.model,
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
        model_name = f'brain_tumor_{args.model}'
    
    # Compile model
    model = compile_model(model, config)
    
    # Display model summary
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, config, model_name)
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)
    
    y_test_cat = to_categorical(y_test, num_classes=config.NUM_CLASSES)
    test_results = model.evaluate(X_test, y_test_cat, verbose=1)
    
    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f}")
    logger.info(f"Test Precision: {test_results[2]:.4f}")
    logger.info(f"Test Recall: {test_results[3]:.4f}")
    logger.info(f"Test AUC: {test_results[4]:.4f}")
    
    logger.info("\n✓ Training completed successfully!")
    logger.info(f"Models saved to: {config.MODELS_DIR}")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")


if __name__ == '__main__':
    main()
