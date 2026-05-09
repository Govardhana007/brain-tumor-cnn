# 🧠 Brain Tumor MRI Classification using CNNs

<p align="center">
  <img src="results/sample_images.png" alt="Sample MRI Images" width="700"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.12+-ff6f00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-d00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

A deep learning system that classifies brain tumor types from MRI scans using **Convolutional Neural Networks (CNNs)**. Implements both a custom CNN architecture and **EfficientNetB0 Transfer Learning** with Grad-CAM explainability.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Approach](#-approach)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Key Learnings](#-key-learnings)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

Brain tumors are among the most life-threatening conditions, and early, accurate diagnosis is critical. Manual MRI interpretation requires expert radiologists and is time-intensive. This project builds an **automated CNN classifier** that can detect and classify four tumor types with high accuracy, demonstrating the real-world potential of AI-assisted medical imaging.

**Classes:** `Glioma` | `Meningioma` | `No Tumor` | `Pituitary`

---

## 📊 Dataset

**Source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Images |
|---|---|
| Training | ~5,712 |
| Testing  | ~1,311 |
| **Total**  | **~7,023** |

All images are T1-weighted contrast-enhanced MRI scans in `.jpg` format.

---

## 📁 Project Structure

```
brain-tumor-cnn/
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb    # Data exploration & visualization
│   ├── 02_Model_Training.ipynb           # CNN + EfficientNet training
│   └── 03_Evaluation_GradCAM.ipynb       # Metrics, ROC, Grad-CAM
│
├── src/
│   ├── models.py                         # Model architecture definitions
│   ├── data_utils.py                     # Dataset loading & augmentation
│   └── predict.py                        # Inference script (CLI)
│
├── models/                               # Saved trained models (.keras)
├── results/                              # Output plots & metrics
├── requirements.txt
└── README.md
```

---

## 🔬 Approach

### 1. Exploratory Data Analysis
- Class distribution analysis
- Sample image visualization per class
- Pixel intensity distribution
- Image size variance study

### 2. Preprocessing & Augmentation
- Resize all images to **224×224**
- Normalize pixel values to [0, 1]
- Augmentation: rotation, zoom, flip, brightness shifts
- Class weighting to handle mild imbalance

### 3. Models Trained

#### 🏗️ Custom CNN (from scratch)
```
Input (224×224×3)
  → Conv Block 1  (32 filters, BN, MaxPool, Dropout)
  → Conv Block 2  (64 filters, BN, MaxPool, Dropout)
  → Conv Block 3  (128 filters, BN, MaxPool, Dropout)
  → Conv Block 4  (256 filters, BN, GlobalAvgPool)
  → Dense(512) → BN → Dropout(0.4)
  → Dense(4, softmax)
```

#### ⚡ EfficientNetB0 (Transfer Learning)
- **Phase 1:** Freeze ImageNet base, train custom head
- **Phase 2:** Unfreeze top 30 layers, fine-tune with LR=1e-5

### 4. Training Strategy
| Component | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Image Size | 224×224 |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 📈 Results

| Model | Test Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| Custom CNN | ~91% | ~0.90 | ~6.2M |
| **EfficientNetB0 (Fine-tuned)** | **~97%** | **~0.97** | **~4.1M** |

### Confusion Matrix
<p align="center">
  <img src="results/confusion_matrices.png" width="700"/>
</p>

### ROC-AUC Curves
<p align="center">
  <img src="results/roc_auc_curves.png" width="700"/>
</p>

### Grad-CAM — Model Explainability
Grad-CAM highlights the regions of the MRI that the CNN focuses on when making predictions.

<p align="center">
  <img src="results/gradcam_visualization.png" width="700"/>
</p>

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-cnn.git
cd brain-tumor-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Install Kaggle CLI, add kaggle.json API key, then:
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```

### 4. Run notebooks in order
```
notebooks/01_EDA_and_Preprocessing.ipynb
notebooks/02_Model_Training.ipynb
notebooks/03_Evaluation_GradCAM.ipynb
```

### 5. Run inference on a new image
```bash
python src/predict.py --image path/to/mri.jpg --model efficientnet
```

**Sample output:**
```
==================================================
  Image       : scan_001.jpg
  Prediction  : PITUITARY
  Confidence  : 97.3%

  Class Probabilities:
    pituitary    ████████████████████  97.3%
    notumor      █                      1.8%
    glioma                              0.6%
    meningioma                          0.3%
==================================================
```

---

## 💡 Key Learnings

- **Transfer learning dramatically outperforms** training from scratch on medical imaging datasets, especially with limited data
- **BatchNormalization** significantly stabilizes training and enables higher learning rates
- **Grad-CAM** is essential for medical AI — models must be explainable, not just accurate
- **Class weighting** is critical even with mild imbalance in medical classification tasks
- **Fine-tuning strategy** (freeze → train head → unfreeze top layers) consistently gives better results than training the full pretrained network from scratch

---

## 🔮 Future Work

- [ ] Ensemble CNN + EfficientNet predictions
- [ ] Add ResNet50V2 and VGG16 for comparison
- [ ] Deploy as a Streamlit web app
- [ ] Integrate DICOM file support for clinical MRI scans
- [ ] Experiment with Vision Transformers (ViT)
- [ ] Add test-time augmentation (TTA)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for learning Deep Learning & Medical AI
</p>
