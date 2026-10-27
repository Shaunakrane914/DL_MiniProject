"""
==============================================================================
Pure Deep Learning Training Pipeline for Marine Oil Spill Detection
Course Module: Deep Learning Mini-Project (AIML / DS)
==============================================================================
Workflow:
1. Setup dataset paths & data generators (Mini-batch streaming with BATCH_SIZE=8)
2. Instantiate the compiled 22-layer U-Net CNN (1,942,577 trainable parameters)
3. Configure training callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
4. Train end-to-end with Adam optimizer and BCE+Dice compound loss
5. Plot and save training loss & Dice score progression curves
6. Perform final unbiased evaluation on the validation dataset
==============================================================================
"""

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Setup project root path so backend modules can be imported seamlessly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import architecture, generator, and hyperparameters from backend/utils.py
from backend.utils import (
    build_unet,
    OilSpillDataGenerator,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
)

print("=" * 80)
print("DEEP LEARNING OIL SPILL DETECTION — TRAINING PIPELINE")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Physical Devices:   {tf.config.list_physical_devices()}")
print(f"GPUs Detected:      {tf.config.list_physical_devices('GPU')}")


# ==============================================================================
# STEP 1: DATASET DIRECTORIES & MINI-BATCH DATA GENERATORS
# ==============================================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_IMG = os.path.join(DATA_DIR, "images", "train")
TRAIN_MASK = os.path.join(DATA_DIR, "masks", "train")
VAL_IMG = os.path.join(DATA_DIR, "images", "val")
VAL_MASK = os.path.join(DATA_DIR, "masks", "val")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Training Data Generator:
# - Loads full training set (6,455 satellite image-mask pairs)
# - Groups into mini-batches of BATCH_SIZE=8 (806 steps per epoch)
# - Normalizes pixel values from [0, 255] to [0.0, 1.0] by dividing by 255.0
# - Applies dynamic data augmentation (random horizontal & vertical flips)
# - Shuffles samples at every epoch end to break cyclical order bias
train_gen = OilSpillDataGenerator(
    image_dir=TRAIN_IMG,
    mask_dir=TRAIN_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=True,
    shuffle=True,
    max_samples=None,  # None = full training dataset (all 6,455 images)
)

# Validation Data Generator:
# - Loads full validation set (1,615 satellite image-mask pairs; 201 batches)
# - Augmentation is disabled (False) to evaluate raw, unbiased real-world performance
# - Shuffling is disabled (False) for deterministic evaluation repeatability
val_gen = OilSpillDataGenerator(
    image_dir=VAL_IMG,
    mask_dir=VAL_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False,
    shuffle=False,
    max_samples=None,  # None = full validation dataset (all 1,615 images)
)

print(f"\n[Data] Training Batches Per Epoch:   {len(train_gen)} (Batch size: {BATCH_SIZE})")
print(f"[Data] Validation Batches Per Epoch: {len(val_gen)}")


# ==============================================================================
# STEP 2: BUILD U-NET CONVOLUTIONAL NEURAL NETWORK
# ==============================================================================

# Builds our 22-hidden-layer U-Net with base_filters=16 (~1.94M trainable parameters).
# Pre-compiled with:
# - Optimizer: Adam (lr = 0.0001, combining Momentum + RMSProp)
# - Loss: BCE + Dice Loss (hybrid compound loss solving class imbalance)
# - Metrics: Dice Similarity Coefficient & IoU (Jaccard Index)
model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), base_filters=16)

# Print full architecture layer table, output tensor shapes, and parameter counts
print("\n[Architecture] Compiled U-Net Model Summary:")
model.summary()


# ==============================================================================
# STEP 3: CONFIGURE TRAINING CALLBACKS
# ==============================================================================

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "oil_spill_unet.keras")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    # 1. ModelCheckpoint: Automatically saves the best weights whenever validation
    #    Dice score improves. Guarantees we preserve the optimal model checkpoint.
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    # 2. EarlyStopping: Monitors validation loss and stops training early if no
    #    improvement occurs for 5 consecutive epochs, restoring the best weights.
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    # 3. ReduceLROnPlateau: Adaptive Learning Rate Decay. If validation loss plateaus
    #    for 2 epochs, cuts the learning rate in half (factor=0.5) down to min 1e-6.
    #    Enables fine-grained parameter tuning near the global minimum.
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ==============================================================================
# STEP 4: MODEL TRAINING (END-TO-END FIT LOOP)
# ==============================================================================

print(f"\n[Training] Starting End-to-End Training for {EPOCHS} Epochs...")
# On each batch:
#   1. Mini-batch of 8 images enters U-Net -> outputs predicted masks (8, 256, 256, 1)
#   2. Loss function (BCE + Dice) computes penalty score
#   3. Backpropagation calculates gradients for all 1.94M weights
#   4. Adam updates weights using Momentum + RMSProp
# Repeated for all 806 batches per epoch!
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


# ==============================================================================
# STEP 5: VISUALIZE & SAVE TRAINING PROGRESSION CURVES
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Subplot 1: Loss Progression (BCE + Dice Loss vs. Epoch)
axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2.5, color="#e63946")
axes[0].plot(history.history["val_loss"], label="Val Loss", linestyle="--", linewidth=2.5, color="#457b9d")
axes[0].set_title("1. BCE + Dice Loss vs. Epoch", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss Value")
axes[0].legend(loc="upper right", frameon=True)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Dice Coefficient Progression (Overlap F1 vs. Epoch)
axes[1].plot(history.history["dice_coefficient"], label="Train Dice", linewidth=2.5, color="#2a9d8f")
axes[1].plot(history.history["val_dice_coefficient"], label="Val Dice", linestyle="--", linewidth=2.5, color="#e76f51")
axes[1].set_title("2. Dice Coefficient (Overlap) vs. Epoch", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice Score (F1)")
axes[1].legend(loc="lower right", frameon=True)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(OUTPUTS_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=200)
print(f"\n[Output] Training curves saved to: {curve_path}")


# ==============================================================================
# STEP 6: FINAL UNBIASED VALIDATION EVALUATION
# ==============================================================================

print("\n[Evaluation] Evaluating best model across all 1,615 validation images...")
val_results = model.evaluate(val_gen, verbose=0)
print("=" * 80)
print("FINAL VALIDATION BENCHMARKS:")
print(f"  * Compound Loss (BCE + Dice) : {val_results[0]:.4f}")
print(f"  * Dice Coefficient (F1)     : {val_results[1]:.4f}")
print(f"  * IoU Score (Jaccard Index)  : {val_results[2]:.4f}")
print(f"  * Best Model Weights Saved   : {MODEL_SAVE_PATH}")
print("=" * 80)
