# 🛸 Deep Learning-Based Marine Oil Spill Detection with Weather-Aware Environmental Analysis

**Deep Learning (U-Net CNN) · Synthetic Aperture Radar (SAR) · ERA5 Wind Reanalysis · Streamlit**

---

## 📌 Problem Statement

Marine oil spills cause catastrophic ecological and economic destruction. Early and accurate detection using spaceborne remote sensing is essential for containment and rapid response. However:
1. **Radar Look-Alikes & False Positives:** Synthetic Aperture Radar (SAR) imagery detects spills by identifying dark patches of reduced sea-surface roughness. However, meteorological phenomena such as low-wind calm sea zones (< 3 m/s), wind shadows, and biogenic slicks also suppress capillary waves and create identical dark signatures.
2. **Disconnected Environmental Context:** Standard computer vision models output static segmentation masks without considering atmospheric conditions (wind speed, wind origin, and surface drift trajectory), leaving environmental response teams unable to assess whether the spill will disperse or threaten coastlines.

### Project Solution
This project develops an end-to-end Deep Learning pipeline and interactive decision-support application that:
- **Segments potential oil spills** from SAR satellite imagery using a custom **U-Net Convolutional Neural Network**.
- **Integrates ERA5 10m atmospheric wind vectors ($u_{10}, v_{10}$)** to validate SAR detection feasibility (3–12 m/s window) and differentiate true spills from low-wind calm sea look-alikes.
- **Projects trajectory drift vectors** using oceanographic wind-drift physics ($\vec{V}_{\text{drift}} \approx 3.2\% \times \vec{V}_{\text{wind}}$) with an interactive Head-Up Display (HUD) overlay and downloadable diagnostic reports.

---

## 🎓 Deep Learning Architecture & Viva Justification

| Architectural Component | Implementation Choice | Technical Justification (Why this was chosen) |
| :--- | :--- | :--- |
| **Model Architecture** | **U-Net CNN** (Encoder-Decoder) | Preserves spatial resolution. The contracting path captures high-level context, while skip connections transfer fine-grained boundary details directly to the decoder for pixel-precise edge localization. |
| **Hidden Activations** | **ReLU** (`max(0, x)`) | Solves the vanishing gradient problem, introduces non-linearity, and computes rapidly compared to saturated sigmoid/tanh activations. |
| **Output Activation** | **Sigmoid** ($\frac{1}{1 + e^{-z}}$) | For binary segmentation, Sigmoid squashes each pixel logit into a calibrated probability $[0.0, 1.0]$ representing oil presence. |
| **Loss Function** | **BCE + Dice Loss** | Oil spills typically cover < 5% of a satellite scene (severe class imbalance). Standard BCE is biased toward the background ocean; Dice loss maximizes overlap (F1-score) directly. |
| **Optimizer** | **Adam** ($\text{LR} = 10^{-4}$) | Combines Momentum and RMSProp with adaptive learning rates per parameter, ensuring smooth convergence across sparse gradient landscapes. |
| **Base Filters** | **16 filters** (doubles per stage) | Yields ~1.94M parameters. Prevents overfitting on speckle noise, speeds up CPU training 3.3×, and keeps the saved `.keras` checkpoint compact (~8MB). |
| **Weight Initializer** | **He Normal (`he_normal`)** | Specifically derived for ReLU activations to prevent signal explosion or dissipation across deep convolutional layers. |
| **Regularization** | **BatchNorm + Dropout (0.15) + Augmentation** | Batch Normalization stabilizes internal covariate shift; Spatial Dropout and random flips prevent orientation bias and co-adaptation. |

---

## 🌦️ ERA5 Weather Physics & Look-Alike Discrimination

In satellite radar remote sensing, dark patches are caused by suppressed capillary-gravity waves:
- **Calm Sea Look-Alikes ($< 3\text{ m/s}$):** Without wind, ocean waters are mirror-smooth, reflecting radar pulses away from the antenna. The entire ocean appears black, causing high false-alarm rates.
- **Optimal SAR Window ($3\text{ -- }12\text{ m/s}$):** Wind generates uniform background capillary waves (bright backscatter). Oil films dampen these waves, creating sharp, high-contrast dark slicks with high radar damping ($>4.5\text{ dB}$).
- **Rough Sea Dispersion ($> 12\text{ m/s}$):** Intense wave breaking forces oil droplets into the water column, disintegrating surface slick cohesion.
- **Oil Drift Kinematics:** Surface oil drift is modeled using empirical oceanographic advection:
  $$\vec{V}_{\text{drift}} \approx 0.032 \times \vec{V}_{\text{wind}} \quad (3.2\% \text{ of 10m wind velocity})$$

---

## 🗂️ Project Structure

```
DL_MP/
│
├── sample_data/            ← Benchmark test scenes (positive spills & look-alikes)
│   ├── images/             ← Sample SAR satellite patches (256x256)
│   └── masks/              ← Corresponding ground-truth binary masks
│
├── models/
│   └── oil_spill_unet.keras ← Trained U-Net model checkpoint (~8 MB)
│
├── notebooks/
│   └── train_model.py      ← Interactive training pipeline (# %% cell format)
│
├── outputs/
│   └── training_curves.png ← Training convergence curves (Loss & Dice vs Epochs)
│
├── app.py                  ← Streamlit interactive web application
├── utils.py                ← U-Net architecture, losses, metrics, ERA5 physics
├── generate_masks.py       ← Synthetic mask generation for Zenodo Part II
├── requirements.txt        ← Minimal dependencies
├── .gitignore              ← Excludes large datasets (>100MB) and temp files
└── README.md               ← Complete project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/<your-username>/oil-spill-detection.git
cd oil-spill-detection
pip install -r requirements.txt
```

### 2. Launch the Streamlit Application
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in your browser.
- Select any of the built-in **Benchmark Sample Scenes** (or upload your own SAR image).
- Adjust detection sensitivity, toggle the **Weather Look-Alike Filter**, and select historical **ERA5 Incident Presets** to visualize drift vectors and download diagnostic reports.

### 3. Model Retraining (Optional)
To retrain the U-Net CNN:
```bash
python notebooks/train_model.py
```
This trains the U-Net model, tracks validation Dice score, saves the best weights to `models/oil_spill_unet.keras`, and exports training convergence plots to `outputs/training_curves.png`.

---

## 📊 Datasets & Benchmarks

1. **Synthetic Aperture Radar (SAR) Imagery:**
   - [Refined Deep-SAR Oil Spill Dataset (Zenodo 15298010)](https://doi.org/10.5281/zenodo.15298010): Curated 256×256 Sentinel-1 C-band & ALOS PALSAR L-band SAR imagery.
   - [Zenodo Record 8346860](https://zenodo.org/records/8346860): Part I Ground-truth oil spill scenes.
   - [Zenodo Record 8253899](https://zenodo.org/records/8253899): Part II Weather look-alikes and clean sea scenes.
2. **Copernicus ERA5 Atmospheric Reanalysis:**
   - 10m eastward ($u_{10}$) and northward ($v_{10}$) wind components from ECMWF.

---

## 🛠️ Technology Stack

- **Deep Learning Framework:** TensorFlow 2.18 / Keras 3
- **Frontend & Visualization:** Streamlit, Matplotlib, OpenCV, Pillow
- **Data & Physics Processing:** NumPy, SciPy
- **Version Control:** Git & GitHub (Clean repository with compact model weights)

---

## ⚖️ Academic Disclaimer
*This system is developed for academic research and educational evaluation in Deep Learning. All neural network outputs represent potential oil spill candidates and should be cross-referenced with maritime patrol and satellite surveillance logs for operational decision-making.*
