"""
data_utils.py — Dataset loading and preprocessing utilities.
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def get_data_generators(train_dir: str, test_dir: str,
                         img_size: int = IMG_SIZE,
                         batch_size: int = BATCH_SIZE,
                         validation_split: float = 0.2):
    """
    Create train, validation, and test data generators with augmentation.

    Returns:
        train_gen, val_gen, test_gen
    """
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )

    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=SEED,
        shuffle=True
    )

    val_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
        shuffle=False
    )

    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print(f"Train samples : {train_gen.samples}")
    print(f"Val   samples : {val_gen.samples}")
    print(f"Test  samples : {test_gen.samples}")
    print(f"Class indices : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen) -> dict:
    """Compute class weights to handle class imbalance."""
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    return dict(enumerate(weights))


def count_dataset(base_dir: str, classes=CLASSES) -> dict:
    """Count images per class in a directory."""
    base = Path(base_dir)
    counts = {}
    for cls in classes:
        path = base / cls
        if path.exists():
            imgs = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg'))
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts
