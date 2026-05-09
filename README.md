## 🔬 How This Project Works — Step by Step

This project builds a **brain tumor classifier** that takes an MRI image as input and predicts which of 4 tumor types it shows. Here's exactly how every part works:

---

### Step 1 — Dataset

**Source:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains MRI scans split into 4 classes:

| Class | Training Images | Testing Images |
|---|---|---|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

All images are T1-weighted contrast-enhanced MRI scans in JPEG format at varying resolutions.

---

### Step 2 — Exploratory Data Analysis (`notebooks/01_EDA_and_Preprocessing.ipynb`)

Before training any model, we explore the data to understand it:

- **Class distribution** — bar charts show whether the dataset is balanced or skewed
- **Sample images** — 4 images per class displayed with inferno colormap to see tissue contrast
- **Pixel intensity histograms** — shows brightness patterns differ between tumor types
- **Image size analysis** — confirms we need to standardize all images to 224×224

**Key finding:** Slight class imbalance exists → handled using `compute_class_weight` so no class is ignored during training.

---

### Step 3 — Preprocessing & Augmentation

Raw images can't go directly into a CNN. Two things must happen first:

**Preprocessing** (applied to all splits):
- Resize every image to **224×224 pixels** (required by CNN input layer)
- Normalize pixel values from `[0–255]` → `[0.0–1.0]` (helps gradient flow)
- Split training data into 80% train / 20% validation

**Augmentation** (applied to training data only):
- Random rotation up to ±20°
- Random zoom up to ±15%
- Random horizontal flip
- Random width/height shift up to ±10%
- Random brightness variation ×[0.8–1.2]

> Why augmentation? MRI scans in the real world come in different orientations, brightness levels, and zoom levels. Augmentation makes the model robust to this variation and prevents overfitting.

---

### Step 4 — Model Architectures (`notebooks/02_Model_Training.ipynb`)

Two models are trained and compared:

#### Model A — Custom CNN (from scratch)


Each block progressively detects more complex features — edges → shapes → parts → tumor patterns.

#### Model B — EfficientNetB0 (Transfer Learning) ⭐ Best model

EfficientNetB0 was pre-trained on 1.2 million ImageNet images, so it already knows how to detect shapes, textures, and edges. We adapt it for MRI tumor classification in two phases:

**Phase 1 — Train the head (15 epochs):**
- Freeze all EfficientNet layers (don't change ImageNet weights)
- Add our own classification head: `GlobalAvgPool → Dense(256) → Dropout → Dense(4)`
- Train only the head at `LR = 1e-3`

**Phase 2 — Fine-tune (15 epochs):**
- Unfreeze the top 30 layers of EfficientNet
- Re-train with a very low `LR = 1e-5` so we adjust the weights slightly without destroying them

---

### Step 5 — Training Strategy

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical Cross-Entropy |
| Batch size | 32 |
| Max epochs | 30 (with early stopping) |
| EarlyStopping | Stops if val_accuracy doesn't improve for 7 epochs |
| ReduceLROnPlateau | Cuts learning rate by 0.3× if val_loss stalls for 4 epochs |
| ModelCheckpoint | Saves only the best model (by val_accuracy) |
| Class weights | Computed to handle class imbalance |

---

### Step 6 — Evaluation (`notebooks/03_Evaluation_GradCAM.ipynb`)

After training, four evaluation methods are used:

**1. Confusion Matrix** — Shows exactly which classes get confused with which. A good model has high numbers only on the diagonal.

**2. Classification Report** — For each class:
- **Precision** = of all images predicted as class X, how many actually were X?
- **Recall** = of all actual class X images, how many did the model find?
- **F1 score** = harmonic mean of precision and recall

**3. ROC-AUC Curves** — Plots the trade-off between true positive rate and false positive rate for each class. AUC closer to 1.0 = better.

**4. Grad-CAM (Gradient-weighted Class Activation Mapping)** — This is the most important evaluation for medical AI. It generates a heatmap overlaid on the MRI showing *exactly which pixels* the CNN focused on when making its prediction. This proves the model is looking at the tumor region, not random background pixels.

---

### Step 7 — Results

| Model | Test Accuracy | Macro F1 | Parameters | Strategy |
|---|---|---|---|---|
| Custom CNN | ~91% | ~0.90 | ~6.2M | Trained from scratch |
| **EfficientNetB0** | **~97%** | **~0.97** | **~4.1M** | **ImageNet → Fine-tuned** |

**Why does EfficientNetB0 win?** Transfer learning gives the model a massive head start — it already knows low-level visual features from millions of images. Fine-tuning adapts those features to MRI scans with far less training data and time needed.

---

### Step 8 — Running Inference on a New Image

Once trained, the model can classify any MRI scan:

```bash
# Single image
python src/predict.py --image path/to/brain_mri.jpg --model efficientnet

# Batch of images
python src/predict.py --folder path/to/mri_folder/ --model efficientnet
```

**Example output:**

==================================================
Image       : patient_042_mri.jpg
Prediction  : PITUITARY
Confidence  : 97.3%


