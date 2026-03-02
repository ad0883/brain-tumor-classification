# Brain Tumor Classification using Deep Learning

## DRDO Internship Project | February 2026

A deep learning-based system for multi-class classification of brain tumors from MRI images using Convolutional Neural Networks (CNN).

---

## 📋 Project Overview

This project implements an automated brain tumor classification system that can identify four categories:

| Class | Description |
|-------|-------------|
| **Glioma** | Tumors originating from glial cells |
| **Meningioma** | Tumors arising from the meninges |
| **Pituitary** | Tumors of the pituitary gland |
| **No Tumor** | Healthy brain scans |

---

## 🏗️ Project Structure

```
brain-tumor-classification/
├── data/
│   ├── raw/                    # Original MRI images
│   │   └── archive/
│   │       ├── Training/       # Training dataset
│   │       └── Testing/        # Testing dataset
│   └── processed/              # Preprocessed numpy arrays
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_test.npy
│       └── y_test.npy
├── src/
│   ├── model.py                # CNN architecture definitions
│   ├── preprocess_data.py      # Data preprocessing pipeline
│   ├── train_model.py          # Model training script
│   ├── evaluate.py             # Evaluation and visualization
│   └── models/                 # Saved model checkpoints
├── results/                    # Training logs and visualizations
├── app/                        # Web application (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- pip package manager
- NVIDIA GPU with CUDA support (recommended for faster training)

### NVIDIA GPU Setup

To use your NVIDIA GPU for training, you need the CUDA toolkit and cuDNN installed.

1. **Verify your GPU driver** — open a terminal and run:
   ```bash
   nvidia-smi
   ```
   This shows your GPU model, driver version, and CUDA version.

2. **Install CUDA Toolkit** — download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).  
   TensorFlow 2.15 requires **CUDA 11.8** or **CUDA 12.x**.

3. **Install cuDNN** — download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) and follow the installation guide for your OS.

4. **Verify CUDA is visible to TensorFlow** after installing dependencies:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   You should see your GPU listed (e.g. `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`).

> **Tip:** The training script automatically enables GPU memory growth, which prevents TensorFlow from allocating all VRAM at once — important for laptops with shared memory.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Start Training (Quick Start)

After installing dependencies and placing the dataset (see **Dataset** section for layout), run these three commands:

```bash
# Step 1 — Preprocess the raw MRI images into numpy arrays
cd src
python preprocess_data.py

# Step 2 — Train the model (uses your NVIDIA GPU automatically if available)
python train_model.py --model custom --epochs 50 --batch-size 32

# Step 3 — Evaluate results
python evaluate.py --model models/brain_tumor_custom_cnn_best.keras
```

The training script will print which device is being used:
```
Training device: /GPU:0    ← NVIDIA GPU found
Training device: /CPU:0    ← fallback (no GPU / CUDA not installed)
```

---

## 📊 Usage

### 1. Data Preprocessing

Preprocess raw MRI images into normalized numpy arrays:

```bash
cd src
python preprocess_data.py
```

This will:
- Load images from `data/raw/archive/`
- Resize to 224×224 pixels
- Normalize pixel values to [0, 1]
- Save as `.npy` files in `data/processed/`

### 2. Model Training

Train the CNN model:

```bash
python train_model.py --model custom --epochs 50 --batch-size 32
```

**Available options:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Architecture: `custom`, `vgg16`, `resnet50` | `custom` |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |

### 3. Model Evaluation

Evaluate the trained model:

```bash
python evaluate.py --model models/brain_tumor_custom_cnn_best.keras
```

This generates:
- Classification report
- Confusion matrix
- ROC curves
- Precision-Recall curves
- Sample predictions visualization

---

## 🧠 Model Architecture

### Custom CNN Architecture

```
Input (224×224×3)
    ↓
[Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout] × 2
    ↓
[Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout] × 2
    ↓
[Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout] × 2
    ↓
[Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout] × 2
    ↓
Global Average Pooling
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(4, softmax)
    ↓
Output (4 classes)
```

### Transfer Learning Models

- **VGG16**: Pre-trained on ImageNet with custom classification head
- **ResNet50V2**: Deep residual network with skip connections

---

## 📈 Training Features

- **Data Augmentation**: Rotation, shift, flip, zoom, shear
- **Callbacks**:
  - Early Stopping (patience=10)
  - Learning Rate Reduction on Plateau
  - Model Checkpointing (best & latest)
  - TensorBoard Logging
  - CSV Logger
- **Regularization**: L2 weight decay, Dropout, BatchNormalization
- **Metrics**: Accuracy, Precision, Recall, AUC

---

## 📊 Results

After training, results are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix visualization |
| `roc_curves.png` | ROC curves for each class |
| `precision_recall_curves.png` | Precision-Recall curves |
| `training_history.png` | Training/validation metrics over epochs |
| `sample_predictions.png` | Sample predictions with confidence |
| `evaluation_results_*.json` | Detailed metrics in JSON format |

---

## 🔧 Configuration

Key parameters can be modified in `train_model.py`:

```python
class TrainingConfig:
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    DROPOUT_RATE = 0.5
    L2_REG = 0.001
    EARLY_STOP_PATIENCE = 10
```

---

## 📚 Dataset

This project uses the Brain Tumor MRI Dataset containing:
- **Training**: ~5,700 images
- **Testing**: ~1,300 images
- **Image Size**: Variable (preprocessed to 224×224)
- **Format**: JPEG

---

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Evaluation metrics

---

## 📄 License

This project is developed as part of a DRDO internship program.

---

## 👤 Author

**DRDO Internship Project**  
Defence Research and Development Organisation  
February 2026

---

## 🙏 Acknowledgments

- DRDO for the internship opportunity
- Brain Tumor MRI Dataset contributors
- TensorFlow and Keras development teams
