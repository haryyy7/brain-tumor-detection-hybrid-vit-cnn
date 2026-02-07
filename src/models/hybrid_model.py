"""Hybrid ViT-CNN model architecture for brain tumor classification."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception


def build_hybrid_model(num_classes=4, input_shape=(224, 224, 3)):
    """Build hybrid Vision Transformer + Xception CNN model."""
    
    # Base model: Xception
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Hybrid_ViT_Xception')
    
    return model


def compile_model(model, learning_rate=0.0001):
    """Compile model with optimizer and loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
