"""
Marine Oil Spill Detection & Weather-Aware Analysis
Frontend Interface (Streamlit Dashboard)
All UI rendering, visual overlays, and user interaction code resides here.
"""

import os
import sys
import io
import numpy as np
from PIL import Image
import streamlit as st

# Setup path to import from backend
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils import (
    IMG_SIZE,
    DETECTION_THRESHOLD,
    COVERAGE_THRESHOLD,
    preprocess_image,
    postprocess_mask,
    compute_coverage,
    is_spill_detected,
    compute_era5_wind_metrics,
    ERA5_INCIDENT_PRESETS,
    load_saved_model,
)

# ─── Frontend Visual Helper ───────────────────────────────────────────────────

def create_visual_overlay(image: np.ndarray, mask: np.ndarray, color=(220, 30, 30), alpha=0.45) -> np.ndarray:
    """Blends detected spill mask (red highlight) onto original image for UI display."""
    img = image.astype(np.float32)
    spill = mask > 127
    overlay = img.copy()
    for ch, val in enumerate(color):
        overlay[:, :, ch] = np.where(spill, img[:, :, ch] * (1 - alpha) + val * alpha, img[:, :, ch])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Converts a binary mask array into downloadable PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return buf.getvalue()


# ─── Streamlit Page Configuration ─────────────────────────────────────────────

st.set_page_config(
    page_title="Marine Oil Spill Detection",
    page_icon="🛰️",
    layout="wide",
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "oil_spill_unet.keras")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data", "images")


@st.cache_resource(show_spinner="Loading trained U-Net model from backend...")
def get_model():
    return load_saved_model(MODEL_PATH)


# ─── Sidebar: Controls & ERA5 Weather Inputs ──────────────────────────────────

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()

    model = get_model()
    if model is not None:
        st.success("✅ Trained U-Net Model Loaded")
    else:
        st.warning("⚠️ Model weights not found at `models/oil_spill_unet.keras`")

    st.subheader("🎛️ Detection Sensitivity")
    px_thresh = st.slider("Pixel Probability Threshold", 0.1, 0.9, DETECTION_THRESHOLD, 0.05,
                          help="Sigmoid probability required to classify a pixel as oil.")
    cov_thresh = st.slider("Min Coverage to Flag Spill (%)", 0.1, 5.0, COVERAGE_THRESHOLD, 0.1,
                           help="Minimum percentage of image area required to trigger a spill alert.")

    st.subheader("💨 ERA5 Weather Context")
    preset_name = st.selectbox("Historical Incident Preset", list(ERA5_INCIDENT_PRESETS.keys()))
    preset = ERA5_INCIDENT_PRESETS[preset_name]
    wind_u10 = preset["u10"]
    wind_v10 = preset["v10"]
    st.caption(f"**Date:** {preset['date']}")
    st.caption(f"_{preset['desc']}_")


# ─── Main UI Panel ────────────────────────────────────────────────────────────

st.title("🛰️ Marine Oil Spill Detection & Weather-Aware Analysis")
st.caption("Deep Learning U-Net CNN · SAR Satellite Imagery · ERA5 Wind Reanalysis")

sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(SAMPLE_DIR) else []

input_mode = st.radio("Choose Scene Source:", ["Benchmark Sample Scene", "Upload Custom SAR Scene"], horizontal=True)

image = None
image_name = None

if input_mode == "Benchmark Sample Scene" and sample_files:
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        selected_file = st.selectbox(
            "Select Benchmark Satellite Scene",
            sample_files,
            format_func=lambda x: f"{x} (Oil Spill Incident)" if any(k in x for k in ["1005", "1009", "1014", "100"]) else f"{x} (Calm Sea / Negative Case)"
        )
        image = Image.open(os.path.join(SAMPLE_DIR, selected_file)).convert("RGB")
        image_name = selected_file
else:
    uploaded_file = st.file_uploader("Upload SAR Satellite Scene (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

if image is not None:
    orig_w, orig_h = image.size
    img_arr = np.array(image)

    if st.button("🚀 Analyze Satellite Scene", use_container_width=True, type="primary"):
        with st.spinner("Running U-Net inference and computing ERA5 drift kinematics..."):
            if model is not None:
                prep = preprocess_image(image, target_size=IMG_SIZE)
                raw_pred = model.predict(prep, verbose=0)
                mask = postprocess_mask(raw_pred, threshold=px_thresh, original_size=(orig_w, orig_h))
            else:
                gray = np.array(image.convert("L"))
                mask = ((gray < 70).astype(np.uint8)) * 255

            coverage = compute_coverage(mask)
            detected = is_spill_detected(coverage, min_pct=cov_thresh)
            overlay = create_visual_overlay(img_arr, mask)
            era5 = compute_era5_wind_metrics(wind_u10, wind_v10)

            # Look-alike discrimination (low-wind false alarm)
            is_lookalike = detected and era5["is_lookalike_risk"]

            st.divider()

            # Status Banner
            if is_lookalike:
                st.warning("⚠️ **LIKELY WEATHER LOOK-ALIKE:** Low wind (< 3 m/s) creates dark, calm sea zones mimicking oil slicks.")
            elif detected:
                st.error(f"🚨 **POTENTIAL OIL SPILL DETECTED:** Slicks occupy **{coverage:.2f}%** of scene area.")
            else:
                st.success("✅ **NO SIGNIFICANT SPILL DETECTED:** Scene appears clear.")

            # Metric Scorecards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spill Coverage", f"{coverage:.2f}%")
            m2.metric("ERA5 Wind Speed", f"{era5['speed_ms']:.2f} m/s", f"{era5['speed_knots']:.1f} kts")
            m3.metric("Wind Origin", f"{era5['met_compass']} ({era5['met_direction_deg']:.0f}°)")
            m4.metric("Projected Spill Drift", f"{era5['drift_compass']} @ {era5['drift_speed_kmh']:.2f} km/h")

            # 3-Panel Visual Comparison
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("1. Original SAR Scene")
                st.image(image, use_container_width=True)
            with c2:
                st.subheader("2. Predicted Binary Mask")
                st.image(mask, use_container_width=True, clamp=True)
            with c3:
                st.subheader("3. Spill Boundary Overlay")
                st.image(overlay, use_container_width=True)

            # Mask Download Button
            png_bytes = mask_to_png_bytes(mask)
            st.download_button(
                label="⬇️ Download Predicted Spill Mask (PNG)",
                data=png_bytes,
                file_name=f"mask_{image_name}",
                mime="image/png",
                use_container_width=True,
            )
