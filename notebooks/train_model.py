# %% [markdown]
# # Deep Learning-Based Marine Oil Spill Detection
# ## U-Net CNN Training Pipeline
#
# **Model:** U-Net Segmentation CNN  
# **Loss Function:** Binary Cross-Entropy + Dice Loss (handles severe class imbalance)  
# **Optimizer:** Adam (learning rate = 1e-4)  
# **Metrics:** Dice Coefficient (F1 Score) & IoU (Jaccard Index)  

# %%
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Robust path resolution whether run from project root or notebooks/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import (
    build_unet,
    OilSpillDataGenerator,
    bce_dice_loss,
    dice_coefficient,
    iou_score,
    IMG_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs Detected: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ### 1. Dataset Setup & Data Generators
# **Why Data Generators?**
# Satellite images require significant memory. Data generators load and preprocess 
# batches of paired SAR images and binary masks on-the-fly, applying data augmentation 
# (random horizontal/vertical flips) to prevent overfitting.

# %%
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_IMG = os.path.join(DATA_DIR, "images", "train")
TRAIN_MASK = os.path.join(DATA_DIR, "masks", "train")
VAL_IMG = os.path.join(DATA_DIR, "images", "val")
VAL_MASK = os.path.join(DATA_DIR, "masks", "val")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Training configuration
# For native Windows CPU execution, 600 training pairs (75 batches) and 160 val pairs (20 batches)
# provide fast convergence within minutes while learning high-contrast oil spill and look-alike features.
TRAIN_EPOCHS = 5
MAX_TRAIN_SAMPLES = 600
MAX_VAL_SAMPLES = 160

# Initialize generators
train_gen = OilSpillDataGenerator(
    image_dir=TRAIN_IMG,
    mask_dir=TRAIN_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=True,
    shuffle=True,
    max_samples=MAX_TRAIN_SAMPLES,
)

val_gen = OilSpillDataGenerator(
    image_dir=VAL_IMG,
    mask_dir=VAL_MASK,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False,
    shuffle=False,
    max_samples=MAX_VAL_SAMPLES,
)

print(f"Train batches per epoch: {len(train_gen)} (Batch size: {BATCH_SIZE})")
print(f"Val batches per epoch:   {len(val_gen)}")

# %% [markdown]
# ### 2. Model Architecture (U-Net CNN)
# **Why U-Net?**
# 1. **Encoder (Contracting Path):** 4 downsampling stages extract high-level semantic features (dampening patterns).
# 2. **Skip Connections:** Direct connections copy fine-grained spatial details from the encoder to the decoder.
# 3. **Decoder (Expansive Path):** 4 upsampling stages restore spatial resolution back to 256x256.
# 4. **Output Layer:** 1x1 Conv with Sigmoid activation outputs pixel probabilities [0.0, 1.0].

# %%
model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), base_filters=16)
model.summary()

# %% [markdown]
# ### 3. Training with Callbacks
# **Why Callbacks?**
# - **ModelCheckpoint:** Saves only the best model weights based on validation Dice score.
# - **EarlyStopping:** Stops training if validation loss does not improve for 7 consecutive epochs (prevents overfitting).
# - **ReduceLROnPlateau:** Halves learning rate if learning stalls, allowing fine-tuning around minima.

# %%
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "oil_spill_unet.keras")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=TRAIN_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# Ensure final trained model is saved
model.save(MODEL_SAVE_PATH)

# %% [markdown]
# ### 4. Performance Evaluation & Training Curves
# Plot loss convergence and Dice overlap coefficient across training epochs.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve (BCE + Dice Loss)
axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
axes[0].set_title("BCE + Dice Loss vs. Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dice coefficient curve (Overlap metric)
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

# Final validation evaluation
val_results = model.evaluate(val_gen, verbose=0)
print(f"\nFinal Validation Results:")
print(f"  • Loss (BCE + Dice) : {val_results[0]:.4f}")
print(f"  • Dice Coefficient  : {val_results[1]:.4f}")
print(f"  • IoU Score         : {val_results[2]:.4f}")
print(f"Model successfully saved to: {MODEL_SAVE_PATH}")
