"""
Brain Tumor Classification - Model Evaluation and Analysis

This module provides comprehensive evaluation tools for the trained
brain tumor classification model, including:
    - Performance metrics (Accuracy, Precision, Recall, F1-Score)
    - Confusion matrix visualization
    - ROC curves and AUC scores
    - Per-class analysis
    - Prediction visualization

Author: DRDO Internship Project
Date: February 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score, f1_score
)
from keras.models import load_model
from keras.utils import to_categorical
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
NUM_CLASSES = 4


class ModelEvaluator:
    """
    Comprehensive model evaluation class for brain tumor classification.
    
    Attributes:
        model: Loaded Keras model
        results_dir: Directory to save evaluation results
        class_names: List of class labels
    """
    
    def __init__(self, model_path: str, results_dir: str = None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to saved Keras model (.keras or .h5)
            results_dir: Directory to save results (default: ../results)
        """
        self.model = load_model(model_path)
        self.results_dir = results_dir or os.path.join(
            os.path.dirname(__file__), '..', 'results'
        )
        self.class_names = CLASS_NAMES
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Results will be saved to {self.results_dir}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test images array
            y_test: Test labels (integer encoded)
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert to categorical for some metrics
        y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)
        
        # Calculate metrics
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        # Print classification report
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            digits=4
        )
        logger.info("\n" + report)
        results['classification_report'] = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate visualizations
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curves(y_test_cat, y_pred_proba)
        self._plot_precision_recall_curves(y_test_cat, y_pred_proba)
        self._plot_class_distribution(y_test, y_pred)
        
        # Save results
        self._save_results(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {results['f1_macro']:.4f}")
        logger.info(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        
        return results
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Generate and save confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.suptitle('Brain Tumor Classification - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Generate and save ROC curves for each class."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})'
            )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Brain Tumor Classification', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curves saved to {save_path}")
    
    def _plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Generate and save Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, color=color, lw=2, label=class_name)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Brain Tumor Classification', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'precision_recall_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Precision-Recall curves saved to {save_path}")
    
    def _plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot class distribution comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=NUM_CLASSES)
        axes[0].bar(self.class_names, true_counts, color='steelblue', edgecolor='black')
        axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        for i, v in enumerate(true_counts):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
        axes[1].bar(self.class_names, pred_counts, color='coral', edgecolor='black')
        axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        for i, v in enumerate(pred_counts):
            axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.suptitle('Class Distribution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Class distribution plot saved to {save_path}")
    
    def _save_results(self, results: dict):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create serializable copy
        save_results = {
            'timestamp': timestamp,
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted'],
            'classification_report': results['classification_report']
        }
        
        save_path = os.path.join(self.results_dir, f'evaluation_results_{timestamp}.json')
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"Evaluation results saved to {save_path}")
    
    def visualize_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int = 16
    ):
        """
        Visualize sample predictions with confidence scores.
        
        Args:
            X_test: Test images
            y_test: True labels
            num_samples: Number of samples to visualize
        """
        y_pred_proba = self.model.predict(X_test[:num_samples], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        rows = int(np.ceil(num_samples / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(X_test[i])
            
            true_label = self.class_names[y_test[i]]
            pred_label = self.class_names[y_pred[i]]
            confidence = y_pred_proba[i][y_pred[i]] * 100
            
            color = 'green' if y_test[i] == y_pred[i] else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                fontsize=10, color=color
            )
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sample predictions saved to {save_path}")


def plot_training_history(history_path: str, results_dir: str):
    """
    Plot training history from saved JSON file.
    
    Args:
        history_path: Path to training history JSON
        results_dir: Directory to save plots
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Classification Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--history', type=str, help='Path to training history JSON')
    args = parser.parse_args()
    
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    logger.info("=" * 60)
    logger.info("Brain Tumor Classification - Model Evaluation")
    logger.info("DRDO Internship Project")
    logger.info("=" * 60)
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    logger.info(f"Loaded test data: {X_test.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, results_dir)
    
    # Run evaluation
    results = evaluator.evaluate(X_test, y_test)
    
    # Visualize predictions
    evaluator.visualize_predictions(X_test, y_test)
    
    # Plot training history if provided
    if args.history and os.path.exists(args.history):
        plot_training_history(args.history, results_dir)
    
    logger.info("\n✓ Evaluation completed successfully!")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
