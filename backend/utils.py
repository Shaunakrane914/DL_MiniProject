"""
Pure Deep Learning & Scientific Backend for Marine Oil Spill Detection
Contains: U-Net Architecture, BCE+Dice Loss, Evaluator Metrics, Data Generator, ERA5 Wind Physics
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

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
DETECTION_THRESHOLD = 0.5
COVERAGE_THRESHOLD = 0.5


# ==============================================================================
# 2. LOSS FUNCTIONS & EVALUATION METRICS
# ==============================================================================

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Computes Dice Coefficient (F1 Score) measuring predicted vs true mask overlap."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Computes Intersection over Union (Jaccard Index)."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combined loss: Binary Cross-Entropy + Dice Loss (handles extreme class imbalance)."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coefficient(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss


# ==============================================================================
# 3. U-NET CONVOLUTIONAL NEURAL NETWORK
# ==============================================================================

def double_conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two 3x3 Conv layers with Batch Normalization and ReLU activation."""
    for _ in range(2):
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            kernel_initializer="he_normal", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def encoder_block(x: tf.Tensor, filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encoder: Double Conv -> Skip Connection -> 2x2 Max Pooling -> Dropout."""
    skip = double_conv_block(x, filters)
    pool = layers.MaxPooling2D(pool_size=2)(skip)
    pool = layers.Dropout(0.15)(pool)
    return skip, pool


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    """Decoder: 2x2 Transpose Conv (Upsample) -> Concatenate Skip -> Double Conv."""
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Dropout(0.15)(x)
    x = double_conv_block(x, filters)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    base_filters: int = 16,
) -> keras.Model:
    """Builds and compiles the complete U-Net CNN segmentation model."""
    inputs = layers.Input(shape=input_shape, name="sar_image")

    # Encoder (Downsampling path: 256 -> 128 -> 64 -> 32 -> 16)
    s1, p1 = encoder_block(inputs, base_filters * 1)
    s2, p2 = encoder_block(p1,     base_filters * 2)
    s3, p3 = encoder_block(p2,     base_filters * 4)
    s4, p4 = encoder_block(p3,     base_filters * 8)

    # Bottleneck
    b = double_conv_block(p4, base_filters * 16)

    # Decoder (Upsampling path: 16 -> 32 -> 64 -> 128 -> 256)
    d1 = decoder_block(b,  s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters * 1)

    # Output Layer
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")(d4)

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
    """Loads and augments paired SAR images and binary masks on-demand."""

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

    def __len__(self) -> int:
        return max(1, len(self.indexes) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs, masks = [], []
        for i in batch_idx:
            img, mask = self._load(self.ids[i])
            imgs.append(img)
            masks.append(mask)
        return np.array(imgs, dtype=np.float32), np.array(masks, dtype=np.float32)

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

def preprocess_image(image: Image.Image, target_size: int = IMG_SIZE) -> np.ndarray:
    """Prepares image for U-Net: resize to 256x256 and scale pixels to [0, 1]."""
    img = image.convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def postprocess_mask(raw_output: np.ndarray, threshold: float = DETECTION_THRESHOLD, original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Converts model sigmoid probabilities into a binary uint8 mask [0, 255]."""
    prob = raw_output[0, :, :, 0]
    binary = (prob > threshold).astype(np.uint8) * 255
    if original_size:
        binary = np.array(Image.fromarray(binary).resize(original_size, Image.NEAREST))
    return binary


def compute_coverage(mask: np.ndarray) -> float:
    """Computes percentage of pixels flagged as oil spill."""
    total = mask.size
    spill = np.count_nonzero(mask > 127)
    return float((spill / total) * 100.0) if total > 0 else 0.0


def is_spill_detected(coverage: float, min_pct: float = COVERAGE_THRESHOLD) -> bool:
    """Returns True if detected oil coverage exceeds minimum alert threshold."""
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
    """Computes wind speed, origin compass, and projected oil drift velocity vector."""
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


def load_saved_model(model_path: str) -> Optional[keras.Model]:
    """Loads saved Keras U-Net model with registered custom loss and metrics."""
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None
