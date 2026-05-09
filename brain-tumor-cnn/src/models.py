"""
models.py — Model architecture definitions for Brain Tumor MRI Classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.4):
    """
    Custom CNN from scratch.
    4 convolutional blocks with BatchNorm, MaxPool, and Dropout.
    ~6.2M parameters.
    """
    model = models.Sequential([
        # Block 1 — edge/texture detection
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Block 2 — shape detection
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3 — part detection
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        # Block 4 — semantic features
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate * 0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='BrainTumorCNN')

    return model


def build_efficientnet(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.4):
    """
    EfficientNetB0 with custom classification head.
    Pre-trained on ImageNet. ~4.1M trainable parameters.
    Phase 1: train head only (base frozen)
    Phase 2: fine-tune top layers (unfreeze last 30)
    """
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = False  # Phase 1: freeze

    inputs = keras.Input(shape=input_shape, name='mri_input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout_rate, name='dropout_head')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='EfficientNetB0_BrainTumor')
    return model, base


def build_resnet50v2(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.4):
    """
    ResNet50V2 — alternative transfer learning backbone.
    """
    base = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='ResNet50V2_BrainTumor'), base


def unfreeze_top_layers(base_model, num_layers=30):
    """Unfreeze the top N layers of a frozen base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfrozen {trainable}/{len(base_model.layers)} layers for fine-tuning")
