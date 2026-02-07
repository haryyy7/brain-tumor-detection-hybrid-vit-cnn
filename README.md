# Brain Tumor Detection using Hybrid Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An advanced deep learning solution for classifying brain tumors from MRI scans using a **Hybrid Vision Transformer (ViT-L16-fe) + Xception CNN** architecture. This project achieves **96%+ accuracy** in multi-class tumor classification.

## 🎯 Project Overview

This project implements a state-of-the-art hybrid deep learning model that combines:
- **Vision Transformer (ViT-L16-fe)** for global feature extraction and attention mechanisms
- **Xception CNN** for local pattern detection and spatial feature learning

### Key Features

✅ **High Accuracy**: Achieves 96.12% test accuracy on brain tumor classification  
✅ **Four-Class Classification**: Distinguishes between Glioma, Meningioma, Pituitary tumors, and No Tumor cases  
✅ **Hybrid Architecture**: Leverages both transformer and CNN strengths for superior performance  
✅ **Production-Ready**: Includes duplicate removal, data augmentation, and comprehensive evaluation metrics  
✅ **Robust Training**: Features early stopping, learning rate scheduling, and model checkpointing

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 96.12% |
| **Training Samples** | 7,392 |
| **Validation Samples** | 2,112 |
| **Test Samples** | 1,056 |
| **Total Parameters** | 22.5M |
| **Training Time** | ~3.5 hours (GPU P100) |

### Classification Report

```
              precision    recall  f1-score   support

      glioma       0.96      0.96      0.96       375
   pituitary       0.97      0.99      0.98       271
  meningioma       0.93      0.91      0.92       234
    no_tumor       0.98      0.99      0.99       176

    accuracy                           0.96      1056
```

## 🏗️ Architecture

The hybrid model architecture consists of:

1. **Vision Transformer Branch (ViT-L16-fe)**
   - Pre-trained on ImageNet
   - Captures global context and long-range dependencies
   - Fine-tuned for medical imaging

2. **CNN Branch (Xception)**
   - Pre-trained on ImageNet
   - Extracts local features and spatial patterns
   - Uses depthwise separable convolutions

3. **Fusion Layer**
   - Concatenates features from both branches
   - Dense layers with dropout for regularization
   - Softmax output for 4-class classification

## 📁 Repository Structure

```
brain-tumor-detection-hybrid-vit-cnn/
├── notebooks/
│   └── brain_tumor_detection.ipynb    # Main training notebook
├── src/
│   ├── models/
│   │   └── hybrid_model.py            # Model architecture
│   ├── data/
│   │   └── preprocessing.py           # Data loading & preprocessing
│   └── utils/
│       ├── callbacks.py               # Training callbacks
│       └── metrics.py                 # Evaluation utilities
├── models/                            # Saved model weights
├── results/                           # Training results & visualizations
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/haryyy7/brain-tumor-detection-hybrid-vit-cnn.git
cd brain-tumor-detection-hybrid-vit-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```txt
tensorflow>=2.10.0
tensorflow-hub>=0.12.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 📖 Usage

### Training the Model

```python
from src.models.hybrid_model import build_model
from src.data.preprocessing import create_df, create_dataset

# Load and prepare data
train_df = create_df(remove_duplicates=True)

# Create datasets
train_dataset = create_dataset(train_df, shuffle=True, augment=True)
val_dataset = create_dataset(val_df, shuffle=False, augment=False)

# Build and train model
model = build_model(input_shape=(224, 224, 3), num_classes=4)
history = train_model(model, train_dataset, val_dataset)
```

### Making Predictions

```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('models/best_model.h5')

# Preprocess image
image = preprocess_image('path/to/mri_scan.jpg')

# Make prediction
prediction = model.predict(tf.expand_dims(image, 0))
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
predicted_class = class_names[tf.argmax(prediction[0])]

print(f"Predicted: {predicted_class}")
print(f"Confidence: {tf.reduce_max(prediction[0]):.2%}")
```

## 🗄️ Dataset Information

The model is trained on curated brain MRI datasets including:
- **Brain MRI Images for Brain Tumor Detection**
- **Brain Tumor Classification (MRI)**
- **Brain Tumor Dataset**

### Data Preprocessing
- Automatic duplicate removal using MD5 hashing
- Label standardization across different naming conventions
- Train/Validation/Test split: 70%/20%/10%
- Image resizing to 224x224 pixels
- Normalization: scaled to [-1, 1] range

### Class Distribution (After Duplicate Removal)
- **Glioma**: 3,754 samples
- **Pituitary**: 2,706 samples
- **Meningioma**: 2,343 samples
- **No Tumor**: 1,757 samples
- **Total**: 10,560 samples

## 🎓 Model Training Details

### Hyperparameters
```python
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
```

### Callbacks
- **EarlyStopping**: Patience of 10 epochs, monitors validation loss
- **ReduceLROnPlateau**: Factor 0.5, patience 5 epochs
- **ModelCheckpoint**: Saves best model based on validation accuracy

### Data Augmentation (Optional)
- Random horizontal flip
- Random brightness adjustment
- Random contrast variation
- Random rotation (90°, 180°, 270°)

## 📈 Results Visualization

The training process generates:
- Training/Validation accuracy curves
- Training/Validation loss curves
- Confusion matrices
- Classification reports
- Per-class performance metrics

All visualizations are saved in the `results/` directory.

## 🔬 Technical Implementation

### Key Components

1. **Label Mapping**: Handles different naming conventions across datasets
2. **Duplicate Detection**: MD5-based file hashing for duplicate removal
3. **Efficient Data Pipeline**: TensorFlow `tf.data` API with prefetching
4. **Mixed Precision Training**: (Optional) for faster training on compatible GPUs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vision Transformer (ViT) implementation from TensorFlow Hub
- Xception architecture from TensorFlow/Keras Applications
- Brain MRI datasets from Kaggle community
- Inspired by recent advances in hybrid deep learning architectures

## 📧 Contact

**Harisanker A**  
GitHub: [@haryyy7](https://github.com/haryyy7)

## 📚 References

1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
2. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions"
3. Recent advances in medical image analysis using deep learning

---

⭐ If you find this project helpful, please consider giving it a star!
