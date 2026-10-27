"""
Pure Deep Learning Training Pipeline for Marine Oil Spill Detection
Course Module: Deep Learning Mini-Project (AIML / DS)
"""

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Setup root directory for backend imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils import (
    build_unet,
    OilSpillDataGenerator,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs Detected: {tf.config.list_physical_devices('GPU')}")


# ==============================================================================
# 1. DATASET SETUP & DATA GENERATORS
# ==============================================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_IMG = os.path.join(DATA_DIR, "images", "train")
TRAIN_MASK = os.path.join(DATA_DIR, "masks", "train")
VAL_IMG = os.path.join(DATA_DIR, "images", "val")
VAL_MASK = os.path.join(DATA_DIR, "masks", "val")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mini-batch Data Generators (Full Dataset: 6,455 Train, 1,615 Val; Batch size: 8)
train_gen = OilSpillDataGenerator(
    image_dir=TRAIN_IMG,
    mask_dir=TRAIN_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=True,      # Random flips to prevent overfitting
    shuffle=True,      # Shuffle batches every epoch
    max_samples=None,  # Use all 6,455 training images
)

val_gen = OilSpillDataGenerator(
    image_dir=VAL_IMG,
    mask_dir=VAL_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False,     # No augmentation for clean validation
    shuffle=False,     # Deterministic evaluation order
    max_samples=None,  # Use all 1,615 validation images
)

print(f"Train batches per epoch: {len(train_gen)} (Batch size: {BATCH_SIZE})")
print(f"Val batches per epoch:   {len(val_gen)}")


# ==============================================================================
# 2. BUILD U-NET CNN MODEL
# ==============================================================================

# Builds 22-hidden-layer U-Net (~1.94M trainable parameters)
model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), base_filters=16)
model.summary()


# ==============================================================================
# 3. TRAINING CALLBACKS
# ==============================================================================

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "oil_spill_unet.keras")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    # 1. ModelCheckpoint: saves the best weights whenever validation Dice score improves
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    # 2. EarlyStopping: stops training if validation loss does not improve for 5 epochs
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    # 3. ReduceLROnPlateau: cuts learning rate by 50% if val_loss plateaus for 2 epochs
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ==============================================================================
# 4. TRAIN MODEL (END-TO-END FIT LOOP)
# ==============================================================================

# Trains model for 5 epochs across all 806 batches per epoch using Adam & BCE+Dice loss
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


# ==============================================================================
# 5. PLOT & SAVE TRAINING PROGRESSION CURVES
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Loss curve: BCE + Dice Loss progression
axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
axes[0].set_title("BCE + Dice Loss vs. Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Overlap curve: Dice Coefficient progression
axes[1].plot(history.history["dice_coefficient"], label="Train Dice", linewidth=2)
axes[1].plot(history.history["val_dice_coefficient"], label="Val Dice", linestyle="--", linewidth=2)
axes[1].set_title("Dice Coefficient vs. Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice Score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(OUTPUTS_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=200)
print(f"Training curves saved to: {curve_path}")


# ==============================================================================
# 6. FINAL VALIDATION EVALUATION
# ==============================================================================

val_results = model.evaluate(val_gen, verbose=0)
print(f"\nFinal Validation Results:")
print(f"  • Loss (BCE + Dice) : {val_results[0]:.4f}")
print(f"  • Dice Coefficient  : {val_results[1]:.4f}")
print(f"  • IoU Score         : {val_results[2]:.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")
