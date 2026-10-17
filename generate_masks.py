"""
generate_masks.py — Synthetic Mask Generator
=============================================
Creates binary oil-spill training masks from the existing SAR images
using adaptive thresholding (dark-region detection).

Physical basis:
    Oil films reduce capillary-wave action → lower radar backscatter
    → darker pixels in SAR imagery.
    We threshold these dark regions to build pseudo ground-truth masks.

⚠  These are SYNTHETIC masks for demonstration / initial training only.
   For best results, replace with actual labelled ground-truth masks
   from the Zenodo oil-spill dataset or equivalent.

Usage
-----
    python generate_masks.py

Reads  : images (1)/images/train/  and  images (1)/images/val/
Writes : dataset/images/train/     and  dataset/masks/train/
         dataset/images/val/       and  dataset/masks/val/
"""

import os
import shutil
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

# Source directories (original dataset)
SRC_TRAIN = os.path.join("images (1)", "images", "train")
SRC_VAL   = os.path.join("images (1)", "images", "val")

# Destination directories (structured for training)
DST_IMG_TRAIN  = os.path.join("dataset", "images", "train")
DST_MASK_TRAIN = os.path.join("dataset", "masks",  "train")
DST_IMG_VAL    = os.path.join("dataset", "images", "val")
DST_MASK_VAL   = os.path.join("dataset", "masks",  "val")

# Mask generation parameters
DARK_THRESHOLD = 90    # Pixels below this grayscale value → oil candidate
MIN_AREA       = 300   # Minimum blob size (px) to retain
MAX_COVERAGE   = 0.45  # Skip images where >45% pixels are "dark" (noisy images)
KERNEL_SIZE    = 9     # Morphological kernel size


# ─── Mask & Look-Alike Generation ──────────────────────────────────────────

def generate_mask(image_path: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Generate a binary mask for a single SAR image with sensor-adaptive calibration
    and physical weather/look-alike discrimination.

    Physical basis:
        - Oil slicks: High local contrast, sharp dampening boundary, moderate localized blob.
        - Weather look-alikes (low wind <3 m/s, biogenic films, calm water):
          Diffuse edges, very wide uniform darkness, or low contrast.
          In ground truth datasets (e.g. Zenodo Part II), look-alikes are assigned mask = 0
          so the segmentation model learns NOT to produce false alarms on calm sea.

    Returns:
        (uint8 mask [0 | 255] or None, category_label: 'oil_spill' | 'weather_lookalike' | 'clean_sea')
    """
    img_pil = Image.open(image_path).convert("L")
    gray = np.array(img_pil, dtype=np.uint8)

    mean_val = float(np.mean(gray))
    std_val  = float(np.std(gray))

    # Calculate Sobel edge gradient to measure boundary sharpness
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_sharpness = float(np.mean(grad_mag))

    # Sensor-adaptive dark threshold:
    # ALOS PALSAR has mean ~135, Sentinel-1 has mean ~85.
    # We calibrate relative to the scene's distribution.
    adaptive_thresh = int(np.clip(mean_val - 0.75 * std_val, 35, 125))

    # Threshold dark regions (candidate dampening)
    _, raw = cv2.threshold(gray, adaptive_thresh, 255, cv2.THRESH_BINARY_INV)
    coverage = np.sum(raw > 0) / raw.size

    # Weather Look-Alike Detection:
    # Very low texture variance (std < 12) or diffuse uniform darkness (> 40% coverage with low edge sharpness)
    is_weather_lookalike = (std_val < 12.0) or (coverage > 0.40 and edge_sharpness < 120.0)

    if is_weather_lookalike:
        # Zenodo Part II convention: look-alike masks are ALL BLACK (0)
        # Model learns that weather-dampened sea surface is background.
        return np.zeros_like(gray, dtype=np.uint8), "weather_lookalike"

    # Morphological opening to eliminate speckle noise, then closing to unify slicks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Filter connected components by minimum blob area
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final = np.zeros_like(cleaned)
    spill_found = False

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA:
            final[labels == i] = 255
            spill_found = True

    category = "oil_spill" if spill_found else "clean_sea"
    return final, category


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def process_split(src_dir: str, dst_img_dir: str, dst_mask_dir: str, split_name: str):
    """Process one data split (train or val)."""
    os.makedirs(dst_img_dir,  exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ])

    print(f"\n[{split_name.upper()}] Processing {len(files)} images from {src_dir}")
    skipped = 0
    saved = 0
    spill_cnt = 0
    lookalike_cnt = 0
    clean_cnt = 0

    for fname in tqdm(files, desc=split_name, unit="img"):
        src_path = os.path.join(src_dir, fname)
        dst_img  = os.path.join(dst_img_dir,  fname)
        dst_mask = os.path.join(dst_mask_dir, fname)

        if os.path.exists(dst_img) and os.path.exists(dst_mask):
            saved += 1
            continue

        mask, category = generate_mask(src_path)

        if mask is None:
            skipped += 1
            continue

        # Copy source image to dataset/images/
        shutil.copy2(src_path, dst_img)

        # Save binary mask to dataset/masks/
        Image.fromarray(mask).save(dst_mask)
        saved += 1

        if category == "oil_spill":
            spill_cnt += 1
        elif category == "weather_lookalike":
            lookalike_cnt += 1
        else:
            clean_cnt += 1

    print(f"  [OK] Total Saved       : {saved} image-mask pairs")
    print(f"     * Potential Spills: {spill_cnt} (foreground mask)")
    print(f"     * Weather Lookalikes: {lookalike_cnt} (zero-mask / background)")
    print(f"     * Clean Sea Surfaces: {clean_cnt} (zero-mask / background)")
    if skipped > 0:
        print(f"  [-] Skipped           : {skipped}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Marine Oil Spill - Synthetic Mask Generator")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Dark threshold : {DARK_THRESHOLD}")
    print(f"  Min blob area  : {MIN_AREA} px")
    print(f"  Max coverage   : {MAX_COVERAGE * 100:.0f}%")
    print(f"  Morpho kernel  : {KERNEL_SIZE}x{KERNEL_SIZE}")

    process_split(SRC_TRAIN, DST_IMG_TRAIN, DST_MASK_TRAIN, "train")
    process_split(SRC_VAL,   DST_IMG_VAL,   DST_MASK_VAL,   "val")

    # Print final counts
    n_train = len(os.listdir(DST_IMG_TRAIN))
    n_val   = len(os.listdir(DST_IMG_VAL))
    print(f"\n{'=' * 60}")
    print(f"  Done!  Train: {n_train} pairs  |  Val: {n_val} pairs")
    print(f"{'=' * 60}")
    print(f"\nNext step: Run  notebooks/train_model.py  to train the U-Net.")


if __name__ == "__main__":
    main()
