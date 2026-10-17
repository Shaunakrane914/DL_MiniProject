"""
app.py — Marine Oil Spill Detection with Weather-Aware Environmental Analysis
==============================================================================
Streamlit Web Application:
- U-Net CNN segmentation of SAR satellite imagery (Sentinel-1 & ALOS PALSAR)
- ERA5 hourly 10m reanalysis wind data integration (drift vector & SAR feasibility)
- Physics-based radar damping & low-wind look-alike discrimination
- Multi-panel visual outputs & report export suite
"""

import io
import os
import numpy as np
import streamlit as st
from PIL import Image

from utils import (
    preprocess_image,
    postprocess_mask,
    create_overlay,
    compute_coverage,
    is_spill_detected,
    generate_synthetic_mask,
    analyze_weather_lookalike,
    compute_era5_wind_metrics,
    ERA5_INCIDENT_PRESETS,
    draw_wind_drift_vector,
    load_saved_model,
    IMG_SIZE,
    DETECTION_THRESHOLD,
    COVERAGE_THRESHOLD,
    MODEL_SAVE_DIR,
)

# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oil Spill Detection & Weather Analysis",
    page_icon="🛸",
    layout="wide",
)

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "oil_spill_unet.keras")


@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_saved_model(path)


# Session state initialization
if "results" not in st.session_state:
    st.session_state.results = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None


# ─── Sidebar Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()

    # Model status check
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.success("Trained U-Net Model Loaded")
        model = get_model(MODEL_PATH)
    else:
        st.warning("Demo Mode (No saved model at `models/oil_spill_unet.keras`)")
        model = None

    st.subheader("🎛️ Detection Parameters")
    px_thresh = st.slider(
        "Pixel Probability Cutoff", 0.10, 0.90, DETECTION_THRESHOLD, 0.05,
        help="Sigmoid probability threshold to classify a pixel as oil spill.",
    )
    cov_thresh = st.slider(
        "Min Area to Flag Spill (%)", 0.10, 5.0, COVERAGE_THRESHOLD, 0.10,
        help="Minimum percentage of spill pixels required to trigger an alert.",
    )

    if not model_exists:
        dark_thresh = st.slider(
            "SAR Dark-Pixel Threshold", 40, 150, 80, 5,
            help="Grayscale cutoff for adaptive demo segmentation.",
        )
    else:
        dark_thresh = None

    st.subheader("🌦️ Weather Look-Alike Filter")
    enable_weather_filter = st.checkbox(
        "Filter Low-Wind Calm Sea Zones", value=True,
        help="Rejects diffuse calm sea zones (<3 m/s wind) that mimic oil slicks.",
    )
    weather_sensitivity = st.slider("Look-Alike Sensitivity (%)", 30, 90, 60, 5)

    st.subheader("💨 ERA5 Wind & Drift Context")
    wind_source = st.radio(
        "Wind Data Source",
        ["Incident Preset", "Custom Vector (u10, v10)"],
    )
    if wind_source == "Incident Preset":
        preset_key = st.selectbox("Historical Event", list(ERA5_INCIDENT_PRESETS.keys()))
        preset_data = ERA5_INCIDENT_PRESETS[preset_key]
        wind_u10 = preset_data["u10"]
        wind_v10 = preset_data["v10"]
        st.caption(f"**Sensor:** {preset_data['sensor']} | **Date:** {preset_data['date']}")
        st.caption(f"_{preset_data['desc']}_")
    else:
        wind_u10 = st.slider("u10: Eastward Wind (m/s)", -15.0, 15.0, -4.2, 0.2)
        wind_v10 = st.slider("v10: Northward Wind (m/s)", -15.0, 15.0, -3.8, 0.2)


# ─── Main Interface Header ───────────────────────────────────────────────────

st.title("🛸 Marine Oil Spill Detection & Weather-Aware Analysis")
st.caption("Deep Learning U-Net CNN Segmentation · SAR Satellite Imagery · ERA5 Reanalysis Wind")

tab_app, tab_docs = st.tabs(["🔍 Analyze Satellite Image", "📖 Project Documentation"])


# ─── Tab 1: Image Analysis ────────────────────────────────────────────────────

with tab_app:
    # Scene selection: Choose between presets or uploading a custom scene
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "images")
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    input_mode = st.radio(
        "Scene Input Mode",
        ["Benchmark Sample Scene", "Upload Custom SAR Scene"],
        horizontal=True,
    )

    image = None
    image_name = None

    if input_mode == "Benchmark Sample Scene" and sample_files:
        col_select, col_guidance = st.columns([3, 1])
        with col_select:
            selected_sample = st.selectbox(
                "Choose Benchmark Satellite Scene",
                sample_files,
                format_func=lambda x: f"{x} (Oil Spill Incident)" if any(k in x for k in ["1005", "1009", "1014", "100"]) else f"{x} (Calm Sea / Negative Case)",
            )
            sample_path = os.path.join(sample_dir, selected_sample)
            image = Image.open(sample_path).convert("RGB")
            image_name = selected_sample
        with col_guidance:
            st.info(
                "**Guidelines:**\n"
                "- Synthetic Aperture Radar (SAR)\n"
                "- 256×256 px resolution\n"
                "- Oil dampens radar waves (dark slicks)"
            )
    else:
        col_upload, col_guidance = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload SAR Satellite Scene",
                type=["png", "jpg", "jpeg", "tif"],
                help="Supports grayscale and RGB SAR images (e.g., Sentinel-1 or ALOS PALSAR).",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
        with col_guidance:
            st.info(
                "**Guidelines:**\n"
                "- Upload SAR radar scenes\n"
                "- Optimal resolution: ≥ 256×256 px\n"
                "- Oil dampens radar waves (dark slicks)"
            )

    if image is not None:
        if image_name != st.session_state.last_file:
            st.session_state.results = None
            st.session_state.last_file = image_name

        orig_w, orig_h = image.size
        img_arr = np.array(image)

        if st.button("🚀 Analyze Satellite Scene", use_container_width=True, type="primary"):
            with st.spinner("Running neural segmentation and environmental analysis..."):
                if model is not None:
                    prep = preprocess_image(image, target_size=IMG_SIZE)
                    raw_pred = model.predict(prep, verbose=0)
                    bin_mask = postprocess_mask(raw_pred, threshold=px_thresh, original_size=(orig_w, orig_h))
                    mode = "model"
                else:
                    bin_mask = generate_synthetic_mask(
                        img_arr,
                        dark_threshold=dark_thresh,
                        weather_filter=enable_weather_filter,
                    )
                    if (orig_w, orig_h) != (bin_mask.shape[1], bin_mask.shape[0]):
                        bin_mask = np.array(
                            Image.fromarray(bin_mask).resize((orig_w, orig_h), Image.NEAREST)
                        )
                    mode = "demo"

                # Environmental and physics diagnostics
                weather_diag = analyze_weather_lookalike(img_arr, bin_mask)
                is_lookalike = enable_weather_filter and (weather_diag["lookalike_prob"] >= weather_sensitivity)

                coverage = compute_coverage(bin_mask)
                detected = is_spill_detected(coverage, min_pct=cov_thresh)

                overlay = create_overlay(img_arr, bin_mask)
                era5 = compute_era5_wind_metrics(wind_u10, wind_v10)
                overlay_hud = draw_wind_drift_vector(overlay, wind_u10, wind_v10)

                st.session_state.results = {
                    "original": img_arr,
                    "mask": bin_mask,
                    "overlay": overlay,
                    "overlay_hud": overlay_hud,
                    "coverage": coverage,
                    "detected": detected,
                    "is_lookalike": is_lookalike,
                    "weather_diag": weather_diag,
                    "era5": era5,
                    "mode": mode,
                    "wind_u10": wind_u10,
                    "wind_v10": wind_v10,
                }

        # ── Results Display ──────────────────────────────────────────────────
        if st.session_state.results is not None:
            res = st.session_state.results
            detected = res["detected"]
            is_lookalike = res["is_lookalike"]
            era5 = res["era5"]
            weather = res["weather_diag"]
            bin_mask = res["mask"]
            coverage = res["coverage"]
            spill_px = int(np.sum(bin_mask > 127))
            total_px = bin_mask.size

            st.divider()

            # Status Banner
            if detected and not is_lookalike:
                st.error("⚠️ **POTENTIAL OIL SPILL DETECTED** — High confidence wave-dampened slick present.")
            elif detected and is_lookalike:
                st.warning("🌊 **LIKELY WEATHER LOOK-ALIKE** — Dark region exhibits diffuse borders typical of low-wind calm sea.")
            else:
                st.success("✅ **NO SIGNIFICANT OIL SPILL DETECTED** — Radar backscatter consistent with normal sea surface.")

            if res["mode"] == "demo":
                st.caption("ℹ️ *Running in demo mode with adaptive thresholding.*")

            # Core Metrics
            st.subheader("📊 Quantitative Assessment")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Estimated Coverage", f"{coverage:.2f}%")
            m2.metric("Spill Pixels", f"{spill_px:,} / {total_px:,}")
            m3.metric("Radar Damping Ratio", f"{weather.get('damping_ratio_db', 0.0):.1f} dB")
            m4.metric("Look-Alike Risk", f"{weather.get('lookalike_prob', 0.0):.0f}%")

            # ERA5 Environmental Weather Dashboard
            st.subheader("💨 ERA5 Reanalysis Weather Context (Dataset 2)")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("10m Wind Speed", f"{era5['speed_ms']:.1f} m/s", f"{era5['speed_knots']:.1f} knots")
            w2.metric("Wind Origin", f"{era5['met_compass']} ({era5['met_direction_deg']:.0f}°)")
            w3.metric("Spill Drift Vector", f"{era5['drift_compass']} ({era5['drift_direction_deg']:.0f}°)", f"~{era5['drift_speed_kmh']:.1f} km/h")
            w4.metric("SAR Feasibility", era5["feasibility_status"])

            st.info(f"**Observation Assessment:** {era5['feasibility_label']} — {era5['feasibility_desc']}")

            # 4-Panel Output Images
            st.subheader("🖼️ Visual Segmentation & Environmental HUD")
            p1, p2, p3, p4 = st.columns(4)
            p1.image(res["original"], caption="1. Original SAR Image", use_container_width=True)
            p2.image(res["mask"], caption="2. Predicted Binary Mask", use_container_width=True, clamp=True)
            p3.image(res["overlay"], caption="3. Highlighted Spill (Red)", use_container_width=True)
            p4.image(res["overlay_hud"], caption="4. Spill + ERA5 Drift Vector", use_container_width=True)

            # Export Section
            st.subheader("💾 Export Outputs")

            def to_png(arr: np.ndarray) -> bytes:
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                return buf.getvalue()

            status_text = (
                "POTENTIAL OIL SPILL DETECTED" if (detected and not is_lookalike)
                else ("LIKELY WEATHER LOOK-ALIKE" if (detected and is_lookalike) else "No significant spill")
            )

            report = (
                f"Oil Spill Detection & Environmental Assessment Report\n"
                f"======================================================\n"
                f"File              : {image_name}\n"
                f"Detection Status  : {status_text}\n"
                f"Estimated Coverage: {coverage:.2f}%\n"
                f"Spill Pixels      : {spill_px:,} of {total_px:,}\n"
                f"Radar Damping     : {weather.get('damping_ratio_db', 0.0):.2f} dB\n"
                f"Look-Alike Risk   : {weather.get('lookalike_prob', 0.0):.1f}%\n\n"
                f"--- ERA5 10m Wind & Drift Data ---\n"
                f"Wind Vector       : u10={res.get('wind_u10'):.2f} m/s, v10={res.get('wind_v10'):.2f} m/s\n"
                f"Wind Speed        : {era5['speed_ms']:.2f} m/s ({era5['speed_knots']:.1f} kts)\n"
                f"Wind Origin       : {era5['met_compass']} ({era5['met_direction_deg']:.1f}°)\n"
                f"Predicted Drift   : {era5['drift_compass']} ({era5['drift_direction_deg']:.1f}°) @ ~{era5['drift_speed_kmh']:.2f} km/h\n"
                f"SAR Feasibility   : {era5['feasibility_label']}\n\n"
                f"Note: Predictions are labelled as potential oil spills.\n"
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("⬇️ Mask PNG", to_png(res["mask"]), "spill_mask.png", "image/png", use_container_width=True)
            d2.download_button("⬇️ Overlay PNG", to_png(res["overlay"]), "spill_overlay.png", "image/png", use_container_width=True)
            d3.download_button("⬇️ Drift HUD PNG", to_png(res["overlay_hud"]), "spill_drift_hud.png", "image/png", use_container_width=True)
            d4.download_button("⬇️ Report TXT", report, "spill_report.txt", "text/plain", use_container_width=True)


# ─── Tab 2: Documentation ────────────────────────────────────────────────────

with tab_docs:
    st.subheader("📖 Project Architecture & Methodology")
    st.markdown("""
### 1. Deep Learning U-Net Architecture
- **Structure:** U-Net encoder-decoder network with skip connections.
- **Loss Function:** Combined Binary Cross-Entropy + Dice Loss for handling extreme class imbalance.
- **Evaluation Metrics:** Dice Coefficient (F1-score), Intersection over Union (IoU), Precision, Recall.

### 2. Core Datasets
- **Dataset 1 (SAR Imagery):** Synthetic Aperture Radar scenes (ALOS PALSAR & Sentinel-1) capturing capillary wave dampening.
  *Recommended benchmark:* [Refined Deep-SAR (Zenodo 15298010)](https://doi.org/10.5281/zenodo.15298010).
- **Dataset 2 (ERA5 Weather Reanalysis):** Copernicus Climate Data Store 10m eastward ($u_{10}$) and northward ($v_{10}$) wind vectors.

### 3. Environmental Physics & Look-Alikes
- **SAR Wind Window:**
  - **$< 3\text{ m/s}$ (Calm Sea):** Capillary waves absent $\\rightarrow$ dark look-alike false alarms.
  - **$3\\text{--}12\\text{ m/s}$ (Optimal):** Reliable contrast between oil and surrounding ocean waves.
  - **$> 12\\text{ m/s}$ (Rough Sea):** Wave breaking disperses oil and breaks slick cohesion.
- **Drift Velocity:** Surface oil slick trajectory is approximated at $\\approx 3.2\\%$ of the 10m wind vector.
""")
