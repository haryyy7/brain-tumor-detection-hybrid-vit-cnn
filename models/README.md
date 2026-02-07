# Saved Models

This directory contains trained model files and checkpoints.

## Model Files

After training, the best model will be saved here:
- `best_ViT-L16-Xception.h5` - Best performing model checkpoint
- `final_model.h5` - Final trained model

## Usage

Load a saved model:
```python
from tensorflow import keras

model = keras.models.load_model('models/best_ViT-L16-Xception.h5')
```

## Model Info
- Architecture: Hybrid Vision Transformer (ViT-L16) + Xception CNN
- Input Shape: (224, 224, 3)
- Classes: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- Framework: TensorFlow/Keras
