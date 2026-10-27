"""
Pure Deep Learning & Scientific Backend for Marine Oil Spill Detection
Course Module: Deep Learning Mini-Project (AIML / DS)
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

IMG_SIZE = 256               # 256x256 input resolution (powers of 2 for clean downsampling)
BATCH_SIZE = 8               # 8 images per mini-batch (balances CPU memory with stable gradients)
EPOCHS = 5                   # 5 complete passes through all 6,455 training images
LEARNING_RATE = 1e-4         # Step size for Adam optimizer (0.0001)
DETECTION_THRESHOLD = 0.5    # Sigmoid probability cutoff for oil spill (>= 0.5 -> Oil)
COVERAGE_THRESHOLD = 0.5     # Minimum oil % in scene to trigger an alert (filters sensor noise)


# ==============================================================================
# 2. LOSS FUNCTIONS & EVALUATION METRICS
# ==============================================================================

# Computes Dice / F1 score: strictly measures overlap on rare oil pixels, ignoring sea water.
# smooth=1.0 prevents division-by-zero (0/0) when an image has zero oil spill.
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# Computes IoU (Jaccard Index): Intersection over Union of oil spill masks.
# Measures strict spatial bounding overlap: area of overlap divided by area of union.
def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# Compound loss combining Binary Cross-Entropy (BCE) with Dice Loss.
# BCE provides smooth pixel gradients while Dice loss solves the 98% sea water class imbalance.
def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coefficient(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss


# ==============================================================================
# 3. U-NET CONVOLUTIONAL NEURAL NETWORK
# ==============================================================================

# Two 3x3 Convs with BatchNorm and ReLU: gives 5x5 receptive field with 28% fewer parameters.
# BatchNorm centers activations to mean=0, std=1; ReLU eliminates vanishing gradients.
def double_conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    for _ in range(2):
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            kernel_initializer="he_normal", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


# Encoder: extracts features into 'skip' at full resolution, then downsamples with 2x2 Max-Pooling.
# Halves spatial dimensions (e.g. 256->128) and applies 15% dropout for regularization.
def encoder_block(x: tf.Tensor, filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    skip = double_conv_block(x, filters)
    pool = layers.MaxPooling2D(pool_size=2)(skip)
    pool = layers.Dropout(0.15)(pool)
    return skip, pool


# Decoder: TransposeConv doubles size (16->32), Concatenate restores sharp edges from encoder skip.
# Bridges fine spatial boundaries lost during pooling, then refines fused channels with double conv.
def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Dropout(0.15)(x)
    x = double_conv_block(x, filters)
    return x


# Assembles the 22-hidden-layer U-Net model with 4 encoders, 1 bottleneck, and 4 decoders.
# Output layer uses 1x1 Conv + Sigmoid to squash scores into [0, 1] probability map.
def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    base_filters: int = 16,
) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="sar_image")

    # Encoder Path (Downsampling: 256 -> 128 -> 64 -> 32 -> 16 to learn global context)
    s1, p1 = encoder_block(inputs, base_filters * 1)
    s2, p2 = encoder_block(p1,     base_filters * 2)
    s3, p3 = encoder_block(p2,     base_filters * 4)
    s4, p4 = encoder_block(p3,     base_filters * 8)

    # Bottleneck (Deepest 16x16 representation; 256 channels; no pooling)
    b = double_conv_block(p4, base_filters * 16)

    # Decoder Path (Upsampling: 16 -> 32 -> 64 -> 128 -> 256 and restoring boundaries via skip connections)
    d1 = decoder_block(b,  s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters * 1)

    # Output Layer (1x1 Conv + Sigmoid: squashes scores into [0, 1] probability mask)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")(d4)

    # Compile with Adam optimizer and BCE+Dice compound loss
    model = keras.Model(inputs=inputs, outputs=outputs, name="OilSpill_UNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_score],
    )
    return model


# ==============================================================================
# 4. DATA GENERATOR (BATCH LOADING & AUGMENTATION)
# ==============================================================================

class OilSpillDataGenerator(keras.utils.Sequence):
    """Loads mini-batches on-demand from disk to prevent RAM crashes; applies dynamic flips."""

    EXTS = (".png", ".jpg", ".jpeg", ".tif")

    # Initializes data generator with dataset directories, batch size, and sample limit.
    # Discovers paired image/mask files and sets up random index shuffling.
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

        self.ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(image_dir)
            if f.lower().endswith(self.EXTS)
        ])
        assert len(self.ids) > 0, f"No images found in {image_dir}"

        if max_samples and max_samples < len(self.ids):
            indices = np.linspace(0, len(self.ids) - 1, max_samples, dtype=int)
            self.ids = [self.ids[i] for i in indices]

        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Returns total number of mini-batches per training epoch.
    # Calculated as total samples divided by batch size (e.g. 6455 // 8 = 806).
    def __len__(self) -> int:
        return max(1, len(self.indexes) // self.batch_size)

    # Triggered automatically by Keras at the end of every training epoch.
    # Reshuffles dataset indices to break cyclical sampling bias.
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Fetches batch #idx as 4D float32 NumPy arrays of shape (Batch_Size, 256, 256, C).
    # Normalizes pixel intensities to [0.0, 1.0] for stable gradient descent.
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs, masks = [], []
        for i in batch_idx:
            img, mask = self._load(self.ids[i])
            imgs.append(img)
            masks.append(mask)
        return np.array(imgs, dtype=np.float32), np.array(masks, dtype=np.float32)

    # Reads, resizes, and scales image and mask pixels by / 255.0.
    # Applies random horizontal and vertical flips simultaneously to both image and mask.
    def _load(self, img_id: str) -> Tuple[np.ndarray, np.ndarray]:
        img_p = os.path.join(self.image_dir, f"{img_id}.png")
        mask_p = os.path.join(self.mask_dir, f"{img_id}.png")

        img = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0

        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        if self.augment:
            if np.random.rand() > 0.5:
                img, mask = np.fliplr(img), np.fliplr(mask)
            if np.random.rand() > 0.5:
                img, mask = np.flipud(img), np.flipud(mask)

        return img, mask


# ==============================================================================
# 5. INFERENCE & DATA HELPERS
# ==============================================================================

# Converts input image to RGB, resizes to 256x256, and divides pixel values by 255.0.
# Adds a leading batch dimension to form a 4D tensor ready for U-Net: (1, 256, 256, 3).
def preprocess_image(image: Image.Image, target_size: int = IMG_SIZE) -> np.ndarray:
    img = image.convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# Thresholds model Sigmoid probabilities into a binary uint8 mask (prob >= threshold -> 255).
# Resizes binary mask back to original input image dimensions using nearest-neighbor interpolation.
def postprocess_mask(
    raw_output: np.ndarray, 
    threshold: float = DETECTION_THRESHOLD, 
    original_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    prob = raw_output[0, :, :, 0]
    binary = (prob > threshold).astype(np.uint8) * 255
    if original_size:
        binary = np.array(Image.fromarray(binary).resize(original_size, Image.NEAREST))
    return binary


# Counts the number of pixels flagged as oil spill in the binary mask (> 127).
# Returns the percentage of the total image area covered by the detected oil.
def compute_coverage(mask: np.ndarray) -> float:
    total = mask.size
    spill = np.count_nonzero(mask > 127)
    return float((spill / total) * 100.0) if total > 0 else 0.0


# Checks if detected oil coverage exceeds the minimum alert threshold (0.5%).
# Returns True for confirmed spills, filtering out tiny isolated sensor noise fragments.
def is_spill_detected(coverage: float, min_pct: float = COVERAGE_THRESHOLD) -> bool:
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


# Calculates wind speed, meteorological origin direction, and 3.2% oil drift velocity from u10/v10.
# Flags look-alike false alarm risk if wind speed is below 3.0 m/s (mirror-like calm sea).
def compute_era5_wind_metrics(u10: float, v10: float) -> Dict[str, Any]:
    speed_ms = float(np.sqrt(u10**2 + v10**2))
    speed_knots = speed_ms * 1.94384

    met_deg = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0
    drift_deg = (met_deg + 180.0) % 360.0
    drift_kmh = (speed_ms * 0.032) * 3.6

    compass_sectors = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                       "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    met_compass = compass_sectors[int((met_deg + 11.25) / 22.5) % 16]
    drift_compass = compass_sectors[int((drift_deg + 11.25) / 22.5) % 16]

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
# 7. MODEL LOADING
# ==============================================================================

CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coefficient": dice_coefficient,
    "iou_score": iou_score,
}


# Loads saved Keras U-Net model (.keras file) safely from disk.
# Registers custom loss and metrics to ensure seamless deserialization.
def load_saved_model(model_path: str) -> Optional[keras.Model]:
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None
