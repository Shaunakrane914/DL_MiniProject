"""
==============================================================================
Pure Deep Learning & Scientific Backend for Marine Oil Spill Detection
Course Module: Deep Learning Mini-Project (AIML / DS)
==============================================================================
Contains:
1. Hyperparameter Definitions (Image resolution, batch size, learning rate)
2. Custom Compound Loss & Metrics (BCE + Dice Loss, Dice Coeff, IoU Score)
3. U-Net Convolutional Neural Network (Encoder, Bottleneck, Decoder, Skips)
4. Memory-Safe Data Generator (On-demand mini-batch streaming & augmentation)
5. Inference Preprocessing & Postprocessing (Normalization & Thresholding)
6. ERA5 Wind & Environmental Drift Kinematics (Physics-based validation)
7. Model Serialization & Checkpoint Loader
==============================================================================
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==============================================================================
# 1. HYPERPARAMETERS
# ==============================================================================

# Spatial resolution (Height x Width) fed into the first Conv2D layer.
# 256 is an exact power of 2 (2^8), allowing clean 4-stage downsampling:
# 256 -> 128 -> 64 -> 32 -> 16 without fractional rounding errors.
IMG_SIZE = 256

# Number of images processed together before backpropagation updates weights.
# Batch size 8 is the mini-batch sweet spot for standard CPU/GPU memory:
# large enough for stable gradient estimation, small enough to prevent Out-Of-Memory.
BATCH_SIZE = 8

# Number of complete passes through the entire dataset (all 6,455 training images).
EPOCHS = 5

# Step size taken by the Adam optimizer along the loss gradient: alpha = 0.0001.
# A small learning rate ensures stable convergence without overshooting minima.
LEARNING_RATE = 1e-4

# Sigmoid probability threshold (tau): pixels with probability >= 0.5 are flagged as oil.
DETECTION_THRESHOLD = 0.5

# Minimum percentage of oil pixels required in the scene to trigger a spill alert.
# Filters out isolated sensor noise fragments (< 0.5% coverage).
COVERAGE_THRESHOLD = 0.5


# ==============================================================================
# 2. LOSS FUNCTIONS & EVALUATION METRICS
# ==============================================================================

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Computes the Dice Similarity Coefficient (F1 Score) for binary segmentation.
    Formula:
        Dice = (2 * |Y_true ∩ Y_pred| + smooth) / (|Y_true| + |Y_pred| + smooth)
    
    Why Dice instead of Pixel Accuracy?
    - In satellite imagery, 98% of pixels are sea water (class imbalance).
    - Dice strictly measures the overlap of the rare foreground OIL pixels,
      ignoring the vast background sea water.
    - 'smooth = 1.0' prevents division-by-zero (0/0) when an image has zero oil.
    """
    # Flatten 4D batch tensors (Batch, H, W, 1) into 1D vectors for dot-product intersection
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    
    # Intersection = sum of element-wise multiplication (True Positives)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # Calculate harmonic mean overlap
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Computes Intersection over Union (IoU / Jaccard Index).
    Formula:
        IoU = (Intersection + smooth) / (Union + smooth)
        where Union = |Y_true| + |Y_pred| - Intersection
    
    Measures the strict bounding overlap between predicted slick and true slick.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Hybrid Compound Loss: Binary Cross-Entropy (BCE) + Dice Loss.
    Formula:
        Total Loss = Mean(BCE) + (1.0 - Dice)
    
    Why combine both losses?
    1. BCE Loss provides smooth, convex gradients for stable pixel-level updates.
       However, BCE alone is biased toward the 98% majority sea-water class.
    2. Dice Loss directly optimizes region overlap and heavily penalizes false
       negatives, preventing the model from lazily predicting all water.
    Together, they deliver stable optimization with immunity to class imbalance.
    """
    # Standard pixel-wise binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Dice Loss: 0.0 when overlap is perfect (Dice=1), 1.0 when prediction misses (Dice=0)
    dice_loss = 1.0 - dice_coefficient(y_true, y_pred)
    
    return tf.reduce_mean(bce) + dice_loss


# ==============================================================================
# 3. U-NET CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE
# ==============================================================================

def double_conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Atomic building block: Two consecutive 3x3 Convolutions with BatchNorm and ReLU.
    
    Architectural Decisions & Syllabus Links:
    1. Why TWO 3x3 Convolutions? (VGG Receptive Field Principle):
       - Stacking two 3x3 convs covers the same effective field-of-view as one 5x5 conv,
         but saves 28% parameters (2 * 3^2 = 18 vs 5^2 = 25 weights per channel)
         and inserts two non-linear ReLUs instead of one.
    2. padding="same":
       - Uses zero-padding (P=1) so spatial height/width do NOT shrink during filtering.
         Essential for U-Net so skip connections align perfectly with decoder features.
    3. kernel_initializer="he_normal":
       - Kaiming He initialization: draws starting weights with variance Var(W) = 2/n_in.
         Mathematically cancels out the 50% variance cut by ReLU, preventing early signal decay.
    4. use_bias=False:
       - Redundant because Batch Normalization immediately subtracts the batch mean (mu)
         and adds its own trainable shift parameter (beta), making Conv2D bias unnecessary.
    5. BatchNormalization():
       - Standardizes activations to zero-mean and unit-variance, speeding up convergence
         and guaranteeing that ~50% of activations enter ReLU positively, preventing Dying ReLU.
    6. Activation("relu"):
       - f(z) = max(0, z). Derivative is constant 1.0 for z > 0, eliminating the
         Vanishing Gradient Problem across all deep hidden layers.
    """
    for _ in range(2):
        # Convolution: Slides 3x3 spatial filter across the feature maps
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            kernel_initializer="he_normal", use_bias=False
        )(x)
        # Normalization: Centers batch distribution to mean=0, std=1
        x = layers.BatchNormalization()(x)
        # Non-Linearity: Thresholds negative values to zero
        x = layers.Activation("relu")(x)
    return x


def encoder_block(x: tf.Tensor, filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Downsampling block on the contracting path (Left side of the U):
    1. double_conv_block: Extracts features at the current spatial resolution.
       -> Saved as 'skip' connection (passed horizontally across the U to the decoder).
    2. MaxPooling2D(pool_size=2): Slides non-overlapping 2x2 window taking the maximum value.
       -> Cuts spatial dimensions exactly in half (e.g., 256x256 -> 128x128). Has 0 parameters.
    3. Dropout(0.15): Randomly disables 15% of feature activations during training
       -> Prevents co-adaptation of feature maps, acting as strong regularization against overfitting.
    """
    # High-resolution feature map (preserved for skip connection)
    skip = double_conv_block(x, filters)
    
    # Downsample spatial resolution by 2x
    pool = layers.MaxPooling2D(pool_size=2)(skip)
    
    # Regularization: randomly zero out 15% of features during training
    pool = layers.Dropout(0.15)(pool)
    
    return skip, pool


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Upsampling block on the expanding path (Right side of the U):
    1. Conv2DTranspose (kernel=2, strides=2):
       - Fractionally strided convolution that doubles spatial height & width (e.g., 16 -> 32).
    2. Concatenate([x, skip]):
       - THE CORE OF U-NET: Glues deep contextual features (from below) with
         crisp, high-resolution spatial boundary features (from the encoder skip connection).
         Solves the spatial blurring problem of traditional autoencoders!
    3. Dropout(0.15): Regularization on the fused representation.
    4. double_conv_block: Convolves and refines the fused channels down to 'filters'.
    """
    # 1. Upsample spatial resolution by 2x (e.g. 16x16 -> 32x32)
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    
    # 2. Skip Connection: Concatenate along the channel axis (last dimension)
    x = layers.Concatenate()([x, skip])
    
    # 3. Regularization
    x = layers.Dropout(0.15)(x)
    
    # 4. Refine fused features with double convolution
    x = double_conv_block(x, filters)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    base_filters: int = 16,
) -> keras.Model:
    """
    Builds, compiles, and returns the complete U-Net CNN Semantic Segmentation model.
    
    Architecture Structure (22 Hidden Convolutional Layers):
    - Input: (256, 256, 3) Normalized SAR Radar Image Tensor
    - Encoder Level 1: 16 filters  -> Output skip: (256, 256, 16) | Pooled: (128, 128, 16)
    - Encoder Level 2: 32 filters  -> Output skip: (128, 128, 32) | Pooled: (64, 64, 32)
    - Encoder Level 3: 64 filters  -> Output skip: (64, 64, 64)   | Pooled: (32, 32, 64)
    - Encoder Level 4: 128 filters -> Output skip: (32, 32, 128)  | Pooled: (16, 16, 128)
    - Bottleneck:      256 filters -> Output:      (16, 16, 256)  | Deepest abstract context
    - Decoder Level 1: 128 filters -> Upsamples to (32, 32, 128)  | Concats with skip s4
    - Decoder Level 2: 64 filters  -> Upsamples to (64, 64, 64)   | Concats with skip s3
    - Decoder Level 3: 32 filters  -> Upsamples to (128, 128, 32) | Concats with skip s2
    - Decoder Level 4: 16 filters  -> Upsamples to (256, 256, 16) | Concats with skip s1
    - Output Layer:    1 filter    -> 1x1 Conv + Sigmoid          | Output: (256, 256, 1)
    
    Parameters:
    - Trainable:     1,942,577 (Weights & biases in Conv, TransposeConv, and BatchNorm)
    - Non-Trainable: 2,944     (Moving mean & moving variance in BatchNorm)
    - Total:         1,945,521 (~7.4 MB on disk)
    """
    # Input Layer: 4D Tensor (Batch_Size, Height, Width, Channels)
    inputs = layers.Input(shape=input_shape, name="sar_image")

    # =========================================================================
    # CONTRACTING PATH (ENCODERS: Downsampling 256 -> 128 -> 64 -> 32 -> 16)
    # Learns "WHAT" is present by trading spatial size for rich feature channels
    # =========================================================================
    s1, p1 = encoder_block(inputs, base_filters * 1)  # Level 1: 16 filters
    s2, p2 = encoder_block(p1,     base_filters * 2)  # Level 2: 32 filters
    s3, p3 = encoder_block(p2,     base_filters * 4)  # Level 3: 64 filters
    s4, p4 = encoder_block(p3,     base_filters * 8)  # Level 4: 128 filters

    # =========================================================================
    # THE BOTTLENECK (Bottom of the U: 16x16 resolution, 256 channels)
    # Highest semantic abstraction; no max-pooling here
    # =========================================================================
    b = double_conv_block(p4, base_filters * 16)

    # =========================================================================
    # EXPANDING PATH (DECODERS: Upsampling 16 -> 32 -> 64 -> 128 -> 256)
    # Pinpoints "WHERE" the oil is by restoring resolution via skip connections
    # =========================================================================
    d1 = decoder_block(b,  s4, base_filters * 8)  # Level 4 Up: 128 filters (fused with s4)
    d2 = decoder_block(d1, s3, base_filters * 4)  # Level 3 Up: 64 filters  (fused with s3)
    d3 = decoder_block(d2, s2, base_filters * 2)  # Level 2 Up: 32 filters  (fused with s2)
    d4 = decoder_block(d3, s1, base_filters * 1)  # Level 1 Up: 16 filters  (fused with s1)

    # =========================================================================
    # OUTPUT LAYER (1x1 Convolution + Sigmoid Activation)
    # Projects 16 feature channels down to a single binary probability channel:
    # y_hat = 1 / (1 + e^-z) where y_hat in [0.0, 1.0]
    # =========================================================================
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")(d4)

    # Instantiate Keras Model
    model = keras.Model(inputs=inputs, outputs=outputs, name="OilSpill_UNet")
    
    # Compile Model with Adam Optimizer & Compound BCE+Dice Loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_score],
    )
    return model


# ==============================================================================
# 4. DATA GENERATOR (BATCH LOADING & DATA AUGMENTATION)
# ==============================================================================

class OilSpillDataGenerator(keras.utils.Sequence):
    """
    On-Demand Mini-Batch Data Generator inheriting from keras.utils.Sequence.
    
    Why use a Generator instead of loading all images into a NumPy array?
    - Loading 8,070 satellite images at 256x256x3 in float32 requires several gigabytes of RAM.
    - A generator loads only 8 images at a time into memory during training, processes the batch,
      and discards it, preventing Out-Of-Memory (OOM) crashes on standard systems.
    """

    EXTS = (".png", ".jpg", ".jpeg", ".tif")

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        batch_size: int = BATCH_SIZE,
        img_size: int = IMG_SIZE,
        augment: bool = False,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle

        # Discover all paired image filenames
        self.ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(image_dir)
            if f.lower().endswith(self.EXTS)
        ])
        assert len(self.ids) > 0, f"No images found in {image_dir}"

        # Subsample if a maximum limit is requested (useful for quick testing)
        if max_samples and max_samples < len(self.ids):
            indices = np.linspace(0, len(self.ids) - 1, max_samples, dtype=int)
            self.ids = [self.ids[i] for i in indices]

        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Total number of mini-batches per epoch = total_samples // batch_size."""
        return max(1, len(self.indexes) // self.batch_size)

    def on_epoch_end(self):
        """Reshuffle indices at the end of every epoch to avoid cyclical bias."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetches batch #idx:
        Returns:
            imgs:  (Batch_Size, 256, 256, 3) float32 array in [0.0, 1.0]
            masks: (Batch_Size, 256, 256, 1) float32 array in {0.0, 1.0}
        """
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs, masks = [], []
        for i in batch_idx:
            img, mask = self._load(self.ids[i])
            imgs.append(img)
            masks.append(mask)
        return np.array(imgs, dtype=np.float32), np.array(masks, dtype=np.float32)

    def _load(self, img_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads, resizes, normalizes, and augments a single image-mask pair."""
        img_p = os.path.join(self.image_dir, f"{img_id}.png")
        mask_p = os.path.join(self.mask_dir, f"{img_id}.png")

        # Load SAR Image: Convert to RGB, resize to 256x256, scale pixels by / 255.0
        img = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0

        # Load Ground Truth Mask: Grayscale (L), Nearest-neighbor resize, scale to {0, 1}
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Add single channel dimension: (256, 256, 1)

        # Dynamic Data Augmentation: Horizontal and Vertical Flips (Both image and mask together!)
        if self.augment:
            if np.random.rand() > 0.5:
                img, mask = np.fliplr(img), np.fliplr(mask)
            if np.random.rand() > 0.5:
                img, mask = np.flipud(img), np.flipud(mask)

        return img, mask


# ==============================================================================
# 5. INFERENCE & DATA HELPERS
# ==============================================================================

def preprocess_image(image: Image.Image, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Prepares a raw uploaded/benchmark image for U-Net inference:
    1. Converts image to RGB (3 channels).
    2. Resizes spatially to (256, 256).
    3. Normalizes pixel values from [0, 255] to [0.0, 1.0] by dividing by 255.0.
    4. Expands dimensions to create a batch tensor of shape (1, 256, 256, 3).
    """
    img = image.convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def postprocess_mask(
    raw_output: np.ndarray, 
    threshold: float = DETECTION_THRESHOLD, 
    original_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Converts model Sigmoid probability output into a binary display mask:
    1. Extracts the 2D probability map: prob in [0.0, 1.0].
    2. Applies decision threshold (prob >= threshold -> 255 white, else 0 black).
    3. Rescales back to the original image dimensions using nearest-neighbor interpolation.
    """
    prob = raw_output[0, :, :, 0]
    binary = (prob > threshold).astype(np.uint8) * 255
    if original_size:
        binary = np.array(Image.fromarray(binary).resize(original_size, Image.NEAREST))
    return binary


def compute_coverage(mask: np.ndarray) -> float:
    """Computes the percentage of total image pixels flagged as an oil spill."""
    total = mask.size
    spill = np.count_nonzero(mask > 127)
    return float((spill / total) * 100.0) if total > 0 else 0.0


def is_spill_detected(coverage: float, min_pct: float = COVERAGE_THRESHOLD) -> bool:
    """Returns True if the oil coverage percentage exceeds the alert threshold (0.5%)."""
    return coverage >= min_pct


# ==============================================================================
# 6. ERA5 WEATHER & ENVIRONMENTAL DRIFT KINEMATICS
# ==============================================================================

ERA5_INCIDENT_PRESETS = {
    "Mumbai High Offshore Spill (Arabian Sea)": {
        "u10": -4.2, "v10": -3.8, "date": "2021-05-18 06:00 UTC",
        "desc": "Post-cyclonic southwesterly drift pushing slick toward coastal Mumbai."
    },
    "Gulf of Mexico Platform Incident": {
        "u10": 2.1, "v10": 5.4, "date": "2020-09-15 12:00 UTC",
        "desc": "NNE offshore winds pushing oil slicks into open waters."
    },
    "Calm-Sea False-Alarm Case (Low-Wind Look-Alike)": {
        "u10": 1.1, "v10": 0.9, "date": "2023-08-11 10:00 UTC",
        "desc": "Low wind (<2 m/s). Mirror-like calm sea creating dark false alarms."
    },
}


def compute_era5_wind_metrics(u10: float, v10: float) -> Dict[str, Any]:
    """
    Computes environmental physics metrics from ERA5 10m wind vector components (u10, v10):
    1. Wind Speed: Magnitude = sqrt(u10^2 + v10^2) in meters/second and knots.
    2. Meteorological Direction: Angle from which the wind blows (degrees from True North).
    3. Oil Drift Direction: Direction toward which the slick drifts (180 degrees opposite).
    4. Estimated Drift Velocity: Empirical oceanographic rule: surface oil drifts at ~3.2% of wind speed.
    5. Look-alike False Alarm Risk: If wind speed < 3.0 m/s, surface capillary waves are absent,
       causing calm water to look dark in SAR radar and triggering false positives.
    """
    speed_ms = float(np.sqrt(u10**2 + v10**2))
    speed_knots = speed_ms * 1.94384

    # Meteorological direction (direction wind blows FROM)
    met_deg = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0
    
    # Spill drift direction (direction wind pushes oil TOWARDS)
    drift_deg = (met_deg + 180.0) % 360.0
    
    # 3.2% empirical surface wind-drift factor
    drift_kmh = (speed_ms * 0.032) * 3.6

    compass_sectors = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                       "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    met_compass = compass_sectors[int((met_deg + 11.25) / 22.5) % 16]
    drift_compass = compass_sectors[int((drift_deg + 11.25) / 22.5) % 16]

    # Look-alike risk threshold (< 3 m/s)
    is_lookalike_risk = speed_ms < 3.0

    return {
        "speed_ms": speed_ms,
        "speed_knots": speed_knots,
        "met_direction_deg": met_deg,
        "met_compass": met_compass,
        "drift_direction_deg": drift_deg,
        "drift_compass": drift_compass,
        "drift_speed_kmh": drift_kmh,
        "is_lookalike_risk": is_lookalike_risk,
    }


# ==============================================================================
# 7. MODEL LOADING WITH CUSTOM OBJECT REGISTRATION
# ==============================================================================

# Dictionary mapping custom function names to their Python definitions for Keras deserialization
CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coefficient": dice_coefficient,
    "iou_score": iou_score,
}


def load_saved_model(model_path: str) -> Optional[keras.Model]:
    """
    Safely loads the trained Keras U-Net model (.keras format) from disk,
    registering custom loss and metric functions to ensure seamless deserialization.
    """
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None
