#!/usr/bin/env python3
"""
Brain Tumor Detection - Single Image Prediction
DRDO Internship Project

Usage:
    python src/predict.py --image path/to/mri_scan.jpg
    python src/predict.py --image path/to/mri_scan.jpg --model src/models/brain_tumor_vgg16_best.keras
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras


# Class labels
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Class descriptions
CLASS_INFO = {
    'Glioma': {
        'severity': '⚠️ HIGH',
        'description': 'Malignant tumor arising from glial cells',
        'recommendation': 'Immediate medical consultation required'
    },
    'Meningioma': {
        'severity': '🟡 MEDIUM',
        'description': 'Usually benign tumor from brain membrane',
        'recommendation': 'Medical evaluation recommended'
    },
    'Pituitary': {
        'severity': '🟡 MEDIUM', 
        'description': 'Tumor in pituitary gland, usually benign',
        'recommendation': 'Endocrinology consultation recommended'
    },
    'No Tumor': {
        'severity': '✅ LOW',
        'description': 'No tumor detected in the scan',
        'recommendation': 'Regular checkups advised'
    }
}


def load_and_preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """Load and preprocess a single image for prediction."""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_tumor(model, image_path: str) -> dict:
    """
    Predict tumor type from MRI image.
    
    Returns:
        dict with prediction results
    """
    # Preprocess image
    img = load_and_preprocess_image(image_path)
    
    # Get predictions
    predictions = model.predict(img, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx] * 100
    
    # Build results
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {
            CLASS_NAMES[i]: float(probabilities[i] * 100) 
            for i in range(len(CLASS_NAMES))
        },
        'class_info': CLASS_INFO[predicted_class]
    }
    
    return results


def print_results(results: dict, image_path: str):
    """Print prediction results in a formatted way."""
    
    print("\n" + "="*60)
    print("🧠 BRAIN TUMOR DETECTION RESULTS")
    print("="*60)
    print(f"\n📁 Image: {image_path}")
    print("-"*60)
    
    # Main prediction
    print(f"\n🎯 PREDICTION: {results['predicted_class'].upper()}")
    print(f"📊 Confidence: {results['confidence']:.2f}%")
    
    # Severity and info
    info = results['class_info']
    print(f"\n⚡ Severity: {info['severity']}")
    print(f"📋 Description: {info['description']}")
    print(f"💡 Recommendation: {info['recommendation']}")
    
    # All probabilities
    print("\n" + "-"*60)
    print("📈 All Class Probabilities:")
    print("-"*60)
    
    sorted_probs = sorted(
        results['all_probabilities'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for class_name, prob in sorted_probs:
        bar_length = int(prob / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        marker = " ←" if class_name == results['predicted_class'] else ""
        print(f"  {class_name:12} [{bar}] {prob:6.2f}%{marker}")
    
    print("\n" + "="*60)
    
    # Warning for tumor detection
    if results['predicted_class'] != 'No Tumor':
        print("\n⚠️  WARNING: Tumor detected!")
        print("    This is an AI prediction and should be verified")
        print("    by a qualified medical professional.")
    else:
        print("\n✅ No tumor detected in this scan.")
        print("   Regular medical checkups are still recommended.")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection from MRI Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/predict.py --image data/raw/archive/Testing/glioma/Te-gl_0010.jpg
    python src/predict.py --image my_scan.jpg --model src/models/brain_tumor_vgg16_best.keras
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to MRI image file (jpg, png)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='src/models/brain_tumor_vgg16_best.keras',
        help='Path to trained model file'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        print("   Train a model first using: python src/train_model.py")
        sys.exit(1)
    
    # Load model
    print("\n🔄 Loading model...")
    model = keras.models.load_model(args.model)
    print("✅ Model loaded successfully!")
    
    # Make prediction
    print("🔍 Analyzing MRI scan...")
    results = predict_tumor(model, args.image)
    
    # Print results
    print_results(results, args.image)
    
    return results


if __name__ == "__main__":
    main()
