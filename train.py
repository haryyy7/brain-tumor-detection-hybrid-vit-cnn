#!/usr/bin/env python3
"""
Brain Tumor Detection using Hybrid ViT-CNN Architecture
Main Training Script
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data.preprocessing import create_df, create_dataset, preprocess_image
from models.hybrid_model import build_hybrid_model
from utils.training import train_model
from utils.evaluation import evaluate_model, plot_training_history
from utils.model_comparison import ModelComparison

def main():
    """Main training pipeline"""
    print("="*60)
    print("Brain Tumor Detection - Hybrid ViT-CNN Training")
    print("="*60)
    
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\nFound {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("\nNo GPU found. Using CPU.")
    
    # Load and prepare data
    print("\n" + "="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    train_df = create_df(base_path=Config.DATA_PATH, remove_duplicates=True)
    
    # Create label mappings
    unique_labels = train_df["labels"].unique()
    index_label = {i: label for i, label in enumerate(unique_labels)}
    label_index = {label: i for i, label in enumerate(unique_labels)}
    train_df["labels"] = train_df["labels"].map(label_index)
    
    num_classes = len(unique_labels)
    class_names = list(unique_labels)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Total samples: {len(train_df)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_val_df, test_df = train_test_split(
        train_df,
        test_size=Config.TEST_SPLIT,
        stratify=train_df['labels'],
        random_state=Config.SEED
    )
    
    train_df_final, val_df = train_test_split(
        train_val_df,
        test_size=Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT),
        stratify=train_val_df['labels'],
        random_state=Config.SEED
    )
    
    print(f"\nTrain samples: {len(train_df_final)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = create_dataset(train_df_final, shuffle=True, augment=Config.USE_AUGMENTATION)
    val_dataset = create_dataset(val_df, shuffle=False, augment=False)
    test_dataset = create_dataset(test_df, shuffle=False, augment=False)
    
    # Initialize model comparison
    comparison = ModelComparison(save_dir="results")
    
    # Build model
    print("\n" + "="*60)
    print("Building Hybrid ViT-CNN Model")
    print("="*60)
    
    model = build_hybrid_model(
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
        num_classes=num_classes,
        vit_model_path=Config.VIT_MODEL_PATH
    )
    
    print(f"\nModel architecture created.")
    print(f"Total parameters: {model.count_params():,}")
    
    # Train model
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name="ViT-L16-Xception",
        epochs=Config.EPOCHS
    )
    
    # Plot training history
    plot_training_history(history, "ViT-L16-Xception")
    
    # Evaluate model
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    result = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        test_labels=test_df['labels'].values,
        class_names=class_names,
        model_name="ViT-L16-Xception"
    )
    
    # Add to comparison
    comparison.add_result(result, history, "ViT-L16-Xception")
    
    # Save final results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    comparison.compare_models()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nTest Accuracy: {result['test_accuracy']:.4f}")
    print(f"Results saved in: results/")
    print(f"Model saved as: models/best_ViT-L16-Xception.h5")

if __name__ == "__main__":
    main()
