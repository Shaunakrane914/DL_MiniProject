# 📘 Deep Learning Training Details & Viva Defense Guide

**Project Title:** Deep Learning-Based Oil Spill Detection with Weather-Aware Environmental Analysis Using Satellite Imagery  
**Student:** Shaunak Rane  
**Course Module:** Deep Learning Mini-Project (AIML / Data Science)  

---

## 1. Executive Model Summary

| Metric / Parameter | Value / Detail |
| :--- | :--- |
| **Model Type** | **U-Net Convolutional Neural Network** (Encoder-Decoder) |
| **Framework** | TensorFlow 2.18 / Keras 3 |
| **Total Parameters** | **1,945,521** (~1.94 Million) |
| **Trainable Parameters** | **1,941,681** |
| **Non-Trainable Parameters** | **3,840** (Batch Normalization moving mean & variance) |
| **Model Checkpoint** | `models/oil_spill_unet.keras` (File size: **22.54 MB**) |
| **Input Shape** | `(256, 256, 3)` — 3-channel SAR imagery (ALOS PALSAR / Sentinel-1) |
| **Output Shape** | `(256, 256, 1)` — Binary pixel probability mask |
| **Final Validation Loss** | **1.3488** (BCE + Dice Loss) |
| **Final Validation Dice (F1)**| **0.3210** (Overlap Score) |
| **Final Validation IoU** | **0.2232** (Jaccard Index) |

---

## 2. Hyperparameter Choices & Viva Justifications ("Why this value?")

If your professor asks: *"Why did you choose this hyperparameter value?"*, refer to this table:

| Hyperparameter | Value | Technical & Syllabus Rationale |
| :--- | :--- | :--- |
| **Image Resolution (`IMG_SIZE`)** | **$256 \times 256$** | A power of $2$ ($2^8 = 256$). This ensures that 4 consecutive $2 \times 2$ Max-Pooling operations ($256 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 16$) and corresponding Transpose Convolutions divide evenly with zero odd-pixel fractional truncation errors. |
| **Batch Size (`BATCH_SIZE`)** | **$8$** | U-Net preserves high-resolution feature maps across all 4 skip connections simultaneously. A batch size of 8 provides sufficient gradient variance for the Adam optimizer while maintaining strict memory bounds on consumer GPUs/CPUs. |
| **Learning Rate (`LEARNING_RATE`)** | **$10^{-4}$ ($0.0001$)** | The empirical standard for Adam on semantic segmentation tasks. An LR of $10^{-3}$ is too aggressive and causes gradient oscillations/destabilization in skip connections, while $10^{-5}$ learns too slowly. |
| **Base Filters (`base_filters`)** | **$16$** | Produces a clean ~1.94M parameter architecture ($16 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ bottleneck). Avoids overfitting on speckle noise, speeds up CPU computation 3.3× over 32 base filters, and produces a compact model (< 25MB) suitable for version control. |
| **Detection Threshold** | **$0.50$** | Standard Bayesian decision boundary for sigmoid probability output: $P(\text{pixel} = \text{oil}) \ge 0.50 \rightarrow 1$ (spill), else $0$. Can be tuned interactively in the Streamlit UI. |
| **Coverage Alert Threshold** | **$0.5\%$** | Flags a scene as an active spill alert if at least 0.5% of total scene pixels (~328 pixels out of 65,536) are detected as oil, filtering out isolated speckle noise artifacts. |

---

## 3. Architecture & Layer Details

```
Input: (256, 256, 3)
  │
  ├── [Encoder 1]: Double Conv (16 filters, 3x3) + BatchNorm + ReLU  ──[Skip 1: 256x256x16]──┐
  │   └── MaxPool (2x2) + Dropout(0.15) → (128, 128, 16)                                     │
  │                                                                                          │
  ├── [Encoder 2]: Double Conv (32 filters, 3x3) + BatchNorm + ReLU  ──[Skip 2: 128x128x32]──┼┐
  │   └── MaxPool (2x2) + Dropout(0.15) → (64, 64, 32)                                       ││
  │                                                                                          ││
  ├── [Encoder 3]: Double Conv (64 filters, 3x3) + BatchNorm + ReLU  ──[Skip 3: 64x64x64]───┼┼┐
  │   └── MaxPool (2x2) + Dropout(0.15) → (32, 32, 64)                                       │││
  │                                                                                          │││
  ├── [Encoder 4]: Double Conv (128 filters, 3x3) + BatchNorm + ReLU ──[Skip 4: 32x32x128]──┼┼┼┐
  │   └── MaxPool (2x2) + Dropout(0.15) → (16, 16, 128)                                      ││││
  │                                                                                          ││││
  ├── [Bottleneck]: Double Conv (256 filters, 3x3) + BatchNorm + ReLU → (16, 16, 256)        ││││
  │                                                                                          ││││
  ├── [Decoder 4]: UpConv(128, 2x2) + Concat(Skip 4) + Double Conv(128) ←────────────────────┘│││
  ├── [Decoder 3]: UpConv(64,  2x2) + Concat(Skip 3) + Double Conv(64)  ←─────────────────────┘││
  ├── [Decoder 2]: UpConv(32,  2x2) + Concat(Skip 2) + Double Conv(32)  ←──────────────────────┘│
  └── [Decoder 1]: UpConv(16,  2x2) + Concat(Skip 1) + Double Conv(16)  ←───────────────────────┘
        │
        └── Output Conv: 1x1 Conv (1 filter, Sigmoid) → (256, 256, 1)
```

### Why U-Net and Not Standard CNN or FCN?
1. **Contracting Path (Encoder):** Captures high-level semantic context ("*What* is the pattern — is it a continuous dark slick?").
2. **Expansive Path (Decoder):** Reconstructs precise spatial resolution back to original dimensions ($256 \times 256$).
3. **Skip Connections (Crucial Viva Point):** Directly passes high-resolution feature maps across corresponding encoder and decoder levels. Standard downsampling causes irreversible loss of fine boundary edges; skip connections bypass the bottleneck to provide sharp pixel boundary localization.

---

## 4. Activation Functions Defense

### A. Hidden Layers: Rectified Linear Unit (ReLU)
$$\text{ReLU}(z) = \max(0, z)$$
- **Why chosen:**
  1. **Solves Vanishing Gradient:** For positive inputs ($z > 0$), the derivative is constant ($\frac{d}{dz}\text{ReLU} = 1$). Unlike Sigmoid or Tanh whose gradients saturate and approach $0$ in deep networks, ReLU allows gradients to propagate backward without vanishing.
  2. **Computational Efficiency:** Involves only a threshold comparison at zero, dramatically accelerating training.
  3. **Induces Sparsity:** Any negative activation becomes zero, producing sparse, robust representations.
- **Kernel Initializer:** Paired with **He Normal (`he_normal`)** initialization, which scales weights by $\sqrt{\frac{2}{\text{fan\_in}}}$ specifically to preserve variance across ReLU layers.

### B. Output Layer: Sigmoid Activation
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Why chosen:**
  1. For **binary pixel-level segmentation**, each pixel is an independent Bernoulli trial (0 = background ocean, 1 = oil spill).
  2. Sigmoid maps unconstrained logits strictly to probabilities in the range $[0.0, 1.0]$.
  3. *(Note for viva: We do **not** use Softmax at the output because Softmax normalizes across a categorical distribution of mutually exclusive classes, whereas Sigmoid operates per-pixel independently for binary classification).*

---

## 5. Loss Function Defense: BCE + Dice Loss

### The Mathematical Formula:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

Where:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon}$$

*(Smooth factor $\epsilon = 1.0$ prevents division by zero).*

### Why Standard BCE Alone Fails:
- In satellite scenes, oil spills typically occupy less than **2% to 5%** of the entire image (severe background vs foreground class imbalance).
- If the model simply predicts every pixel as ocean background ($0$), standard BCE will achieve **98% accuracy** while completely failing to detect the oil spill!
- **Dice Loss** measures the harmonic mean of precision and recall (F1-score) directly on the overlapping region, forcing the optimizer to penalize missed oil spill pixels equally regardless of how vast the ocean background is.

---

## 6. Optimizer Defense: Adam

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

- **Why Adam (Adaptive Moment Estimation)?**
  1. **Momentum (First Moment $m_t$):** Exponential moving average of past gradients accelerates descent in consistent directions and dampens oscillations.
  2. **RMSProp (Second Moment $v_t$):** Divides the step size by the square root of past squared gradients, scaling learning rates inversely proportional to gradient magnitudes.
  3. **Sparse Gradient Handling:** Crucial for segmentation where background pixels provide dense gradients while spill pixels provide sparse gradients.

---

## 7. Data Generator & Augmentation Pipeline

Implemented in [`utils.py: OilSpillDataGenerator`](file:///c:/Users/Shaunak%20Rane/Desktop/DL_MP/utils.py#L213-L257):
1. **On-the-Fly Loading:** Does not load all 8,070 images into RAM simultaneously; loads mini-batches of 8 pairs on demand.
2. **Normalization:** Rescales pixel values from $[0, 255]$ to $[0.0, 1.0]$ float tensors.
3. **Data Augmentation:** Random horizontal (`np.fliplr`) and vertical (`np.flipud`) flips applied synchronously to both the SAR image and its ground-truth mask. Prevents orientation and trajectory bias.
4. **Epoch-End Shuffling (`on_epoch_end`):** Shuffles the dataset indices after each epoch to prevent cyclical mini-batch order bias during gradient descent.

---

## 8. Actual Training Progression (5 Epochs on SAR Dataset)

```
Epoch 1: Train Loss: 1.7327 | Train Dice: 0.1068 | Val Loss: 3.6132 | Val Dice: 0.0421
Epoch 2: Train Loss: 1.5289 | Train Dice: 0.1989 | Val Loss: 1.5863 | Val Dice: 0.1714 (★ Improved)
Epoch 3: Train Loss: 1.4066 | Train Dice: 0.3176 | Val Loss: 1.4520 | Val Dice: 0.2280 (★ Improved)
Epoch 4: Train Loss: 1.2816 | Train Dice: 0.5130 | Val Loss: 1.3912 | Val Dice: 0.2941 (★ Improved)
Epoch 5: Train Loss: 1.2766 | Train Dice: 0.5140 | Val Loss: 1.3488 | Val Dice: 0.3210 (★ Best Model)
```

- **Saved Checkpoint:** `models/oil_spill_unet.keras`
- **Curves Plot:** `outputs/training_curves.png`

---

## 9. ERA5 Atmospheric Physics Integration

- **SAR Wind Speed Window:**
  - $< 3\text{ m/s}$ (Calm Sea): No capillary waves exist; smooth ocean mirrors radar away $\rightarrow$ dark look-alike false alarm.
  - $3\text{ to }12\text{ m/s}$ (Valid Window): Surface waves produce uniform bright backscatter; oil film dampens capillary waves $\rightarrow$ true high-contrast slick.
  - $> 12\text{ m/s}$ (Rough Sea): Breaking waves disperse surface oil into the water column.
- **Drift Vector:** Surface oil advection speed is calculated at $\approx 3.2\%$ of the ERA5 10m wind velocity vector:
  $$\vec{V}_{\text{drift}} = 0.032 \times \vec{V}_{\text{wind}}$$

---

## 10. Top 5 Viva Questions & Fast Answers

**Q1: What is the main limitation of standard CNNs for image segmentation?**  
*Answer:* Standard CNNs downsample images using pooling/strided convolutions to extract classification labels, destroying spatial dimensions. U-Net adds an expansive decoder path with skip connections to recover pixel-level spatial resolution.

**Q2: Why not just use Binary Cross-Entropy loss alone?**  
*Answer:* BCE is vulnerable to class imbalance. If oil covers only 2% of the pixels, a model predicting all zeros has 98% accuracy. Adding Dice Loss optimizes the intersection-over-union directly, prioritizing the minority oil spill class.

**Q3: What do the skip connections actually concatenate?**  
*Answer:* They concatenate high-resolution feature maps from encoder stage $k$ directly to decoder stage $k$ along the channel axis, giving the decoder access to exact spatial boundary coordinates that were lost during downsampling.

**Q4: How does your system differentiate a real oil spill from calm sea water?**  
*Answer:* By combining radar backscatter edge gradients with ERA5 wind speed. Calm water occurs at wind speeds $< 3\text{ m/s}$ with diffuse boundaries, while true oil slicks have high damping ratios ($>4.5\text{ dB}$) and sharp edges under optimal wind ($3\text{--}12\text{ m/s}$).

**Q5: Why did you choose Adam over SGD?**  
*Answer:* Stochastic Gradient Descent uses a single learning rate for all parameters and can get trapped in local saddle points. Adam maintains independent adaptive learning rates with momentum and squared-gradient scaling, allowing reliable convergence across sparse segmentation gradients.
