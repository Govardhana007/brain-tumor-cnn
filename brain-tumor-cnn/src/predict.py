"""
predict.py — Run inference on a single image or folder of images.

Usage:
    python src/predict.py --image path/to/mri.jpg
    python src/predict.py --folder path/to/folder/ --model efficientnet
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224

MODEL_PATHS = {
    'cnn': 'models/custom_cnn_final.keras',
    'efficientnet': 'models/efficientnet_finetuned.keras'
}


def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Train models first via Notebook 02.")
    print(f"Loading model: {path}")
    return keras.models.load_model(path)


def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_single(model, img_path: str) -> dict:
    img_array = preprocess_image(img_path)
    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        'image': str(img_path),
        'predicted_class': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probabilities': {cls: float(p) for cls, p in zip(CLASSES, probs)}
    }


def predict_folder(model, folder_path: str) -> list:
    folder = Path(folder_path)
    results = []
    image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))

    if not image_files:
        print(f"No images found in: {folder_path}")
        return results

    print(f"Found {len(image_files)} images. Running inference...")
    for img_path in image_files:
        try:
            result = predict_single(model, img_path)
            results.append(result)
            print(f"  {img_path.name:40s} → {result['predicted_class']:12s} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"  ERROR on {img_path.name}: {e}")

    return results


def print_result(result: dict):
    print("\n" + "="*50)
    print(f"  Image       : {result['image']}")
    print(f"  Prediction  : {result['predicted_class'].upper()}")
    print(f"  Confidence  : {result['confidence']:.1%}")
    print("\n  Class Probabilities:")
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 20)
        print(f"    {cls:12s} {bar:<20s} {prob:.1%}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classifier')
    parser.add_argument('--image',  type=str, help='Path to a single MRI image')
    parser.add_argument('--folder', type=str, help='Path to folder of MRI images')
    parser.add_argument('--model',  type=str, default='efficientnet',
                        choices=['cnn', 'efficientnet'], help='Model to use')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    model = load_model(args.model)

    if args.image:
        result = predict_single(model, args.image)
        print_result(result)

    elif args.folder:
        results = predict_folder(model, args.folder)
        if results:
            classes_found = [r['predicted_class'] for r in results]
            print(f"\nSummary: {len(results)} images processed")
            from collections import Counter
            for cls, count in Counter(classes_found).most_common():
                print(f"  {cls}: {count}")


if __name__ == '__main__':
    main()
