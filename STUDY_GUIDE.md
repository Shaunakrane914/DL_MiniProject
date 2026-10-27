# Shaunak's Deep Learning Mini Project — Simple Code-by-Code Study Guide

Repository reviewed: https://github.com/Shaunakrane914/DL_MiniProject  
Reference notes reviewed: `SHAUNAK_DL_PROJECT_EXPLANATION.md` (uploaded by the user)  
Main purpose: help a student explain the actual implementation accurately in a viva or interview.

> **Important:** This guide distinguishes what the code actually does from what the README or notes claim. Do not describe a feature as implemented if the current code does not implement it.

---

## 1. The project in simple words

The project takes a satellite radar image of the ocean and uses a U-Net convolutional neural network to predict which pixels might belong to an oil slick. It then turns the prediction into a black-and-white mask, calculates the percentage of the image marked as spill, displays a red overlay, and calculates wind/drift figures from a selected predefined scenario.

The motivation is that oil slicks can appear as dark patches in Synthetic Aperture Radar (SAR) images because oil dampens small surface waves. But calm water and some natural slicks can also appear dark. So darkness alone is not proof of oil.

### A 30-second explanation you can say aloud

“My project is a deep-learning-based marine oil-spill candidate detector. It uses a U-Net model to perform semantic segmentation on SAR satellite images. The image is resized and normalized, and the model predicts a probability for every pixel. We apply a threshold to generate a binary spill mask and calculate how much of the image is marked. A Streamlit dashboard displays the original image, mask, and overlay. The application also uses preset wind-vector values to calculate wind speed, direction, and a simplified estimated drift speed, and warns when low wind could make a dark patch a false alarm. It is an academic prototype, not an operational satellite-monitoring service.”

### What it does and does not do

**Implemented:**
- Builds a U-Net CNN with an encoder, bottleneck, decoder, skip connections, and a one-channel sigmoid output.
- Defines BCE + Dice loss, Dice and IoU metrics, Adam optimization, Batch Normalization, Dropout, and He Normal initialization.
- Loads image/mask pairs in batches, normalizes them, and optionally flips both together.
- Defines a training script with validation, callbacks, and a training-curve plot.
- Loads a saved Keras model in a Streamlit application.
- Accepts sample images or uploaded PNG/JPG/JPEG images.
- Applies pixel and coverage thresholds, displays a mask and red overlay, and offers mask download.
- Calculates wind and approximate drift metrics from hard-coded scenario values.

**Not implemented by the visible repository code:**
- A live connection to the ERA5 service or automatic download of weather for the uploaded image's location and timestamp.
- Automatic satellite data acquisition or continuous real-time monitoring.
- A validated oil-chemistry classifier. A low-wind warning is a heuristic warning, not proof that a patch is or is not oil.
- A full drift forecast using ocean currents, waves, tides, coastline geometry, or a time-stepped trajectory model.
- The training dataset folder expected by the training script. The repository includes only seven sample image/mask pairs; the larger `dataset/` folder is ignored by Git and is not present in the repository tree.

---

## 2. Repository map: what every tracked file is for

| File/folder | What it is responsible for |
|---|---|
| `.gitignore` | Tells Git not to track cache files, virtual environments, and large local dataset folders. |
| `DL_Notes/` | Contains JPEG images of study notes; these are not imported by the application code. |
| `README.md` | Project overview, installation instructions, architecture summary, and dataset references. |
| `TRAINING_DETAILS.md` | Technical explanation, training metrics, and viva preparation notes. It is documentation, not executable code. |
| `backend/__init__.py` | Marks `backend` as a Python package; its contents are just a short module docstring. |
| `backend/utils.py` | The central module: constants, metrics/loss, U-Net architecture, data generator, image preprocessing/postprocessing, wind calculations, and model loading. |
| `backend/train_model.py` | Script for building and training a new U-Net, validating it, saving the best model checkpoint, and plotting curves. |
| `frontend/app.py` | Streamlit user interface, image selection/upload, prediction call, display, warning, and mask download. |
| `models/oil_spill_unet.keras` | Saved Keras model checkpoint. It stores the model configuration and learned state needed to load the trained network. It is not a Python source file. |
| `sample_data/images/` | Seven example SAR-style PNG images used by the demo. |
| `sample_data/masks/` | Seven matching mask PNGs. They are paired with the sample images by filename; the app does not currently display these ground-truth masks during inference. |
| `outputs/training_curves.png` | Saved training/validation loss and Dice plot. |
| `outputs/training_metrics.json` | Stored training summary with dataset counts, hyperparameters, and per-epoch metrics. |
| `requirements.txt` | Python package dependencies. |

### How the main files connect

1. `frontend/app.py` imports helper functions from `backend/utils.py`.
2. The app calls `load_saved_model()` to load `models/oil_spill_unet.keras`.
3. When the user presses Analyze, the app calls `preprocess_image()`, then `model.predict()`, then `postprocess_mask()`.
4. It calculates coverage and wind metrics, then renders the result.
5. Separately, `backend/train_model.py` imports the generator and model-building function from `backend/utils.py`, trains a new model, and saves a checkpoint and plot.

The dashboard does not train the model every time it is opened. Training and prediction are separate workflows.

---

## 3. `backend/__init__.py` — package marker

The file contains a docstring describing the backend package.

- The triple-quoted string is a module docstring: a human-readable description.
- There is no model logic here.
- It helps Python treat `backend` as a package so imports such as `from backend.utils import ...` are clear and reliable.

**Viva answer:** “This file identifies the backend folder as a Python package. It does not perform prediction or training itself.”

---

## 4. `backend/utils.py` — the core module

This is the most important file. Its sections are explained below in code order.

### 4.1 Imports

- `os`: file and directory paths, checking whether the saved model exists, and listing image files.
- `typing` types such as `Tuple`, `Optional`, and `Dict`: type hints that describe the expected kinds of values. They do not train the model.
- `numpy as np`: array operations, image conversion, random flips, sums, square roots, and trigonometry.
- `PIL.Image`: opens, converts, resizes, and saves images.
- `tensorflow as tf`: tensor operations and the deep-learning framework.
- `keras` and `layers`: model creation, layers, optimizers, callbacks, and data-sequence support.

### 4.2 Hyperparameters

```python
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
DETECTION_THRESHOLD = 0.5
COVERAGE_THRESHOLD = 0.5
```

- `IMG_SIZE = 256`: input images and training masks are resized to 256 by 256 pixels.
- `BATCH_SIZE = 8`: the generator returns up to eight image/mask pairs per batch in the intended training setup.
- `EPOCHS = 5`: the training script makes at most five passes through its generator.
- `LEARNING_RATE = 1e-4`: Adam's starting learning rate is 0.0001.
- `DETECTION_THRESHOLD = 0.5`: each output pixel is marked as spill only when its predicted score is greater than this threshold. The code uses `>` rather than `>=`.
- `COVERAGE_THRESHOLD = 0.5`: the app's default alert threshold is 0.5 percent of pixels marked as spill. This is a configurable engineering choice, not a scientifically universal boundary.

### 4.3 `dice_coefficient(y_true, y_pred, smooth=1.0)`

Purpose: measure how much the predicted mask overlaps the reference mask.

1. `tf.cast(..., tf.float32)` converts the inputs to floating-point tensors.
2. `tf.reshape(..., [-1])` flattens the tensors into one long vector.
3. `tf.reduce_sum(y_true_f * y_pred_f)` calculates the sum of elementwise products, which acts as the overlap term.
4. The function returns `(2 * intersection + smooth) / (sum_true + sum_pred + smooth)`.
5. `smooth=1.0` avoids a zero denominator when both masks are empty.

The score is usually interpreted as closer to 1 being better overlap and closer to 0 being poor overlap.

**Technical nuance:** this function uses the values supplied to it. During model training, `y_pred` is a probability map, not a thresholded binary mask. Also, flattening the complete tensor means the metric aggregates all pixels in the supplied batch together rather than calculating a separate Dice score for each image and then averaging.

### 4.4 `iou_score(y_true, y_pred, smooth=1.0)`

Purpose: calculate Intersection over Union (IoU), also called the Jaccard index.

1. Cast both inputs to floating point.
2. Flatten them.
3. Calculate the intersection by summing their elementwise product.
4. Calculate union as `sum_true + sum_pred - intersection`.
5. Return `(intersection + smooth) / (union + smooth)`.

Higher IoU means stronger overlap. Dice and IoU are related but are not the same score.

### 4.5 `bce_dice_loss(y_true, y_pred)`

```python
bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
dice_loss = 1.0 - dice_coefficient(y_true, y_pred)
return tf.reduce_mean(bce) + dice_loss
```

- Binary Cross-Entropy (BCE) penalizes incorrect pixel probabilities.
- Dice loss is `1 - Dice`, so better overlap produces a smaller Dice loss.
- The function averages BCE and adds Dice loss.

**Why combine them?** Most pixels may be ocean background, while spill pixels occupy a small part of the image. BCE provides pixel-wise probability feedback; Dice loss makes overlap with the spill region matter more directly. The combination is intended to reduce the tendency to favour the majority background class.

### 4.6 `double_conv_block(x, filters)`

This helper creates two convolution stages in a row. The loop `for _ in range(2)` repeats the same sequence twice:

1. `Conv2D(filters, kernel_size=3, padding="same", ...)`: applies learnable 3×3 filters to find local patterns. `padding="same"` keeps height and width unchanged when stride is one.
2. `kernel_initializer="he_normal"`: initializes weights using a distribution intended to work well with ReLU activations.
3. `use_bias=False`: omits a convolutional bias because Batch Normalization follows it and has its own learned shift parameter.
4. `BatchNormalization()`: normalizes intermediate activations using batch statistics during training and stored moving statistics during inference.
5. `Activation("relu")`: applies `max(0, x)`, adding non-linearity and zeroing negative activations.

The function returns the final feature tensor. `filters` controls how many channels the layer produces.

### 4.7 `encoder_block(x, filters)`

The encoder extracts features and reduces spatial resolution.

1. `skip = double_conv_block(x, filters)`: extract features at the current resolution and save them for a skip connection.
2. `MaxPooling2D(pool_size=2)`: halves width and height by taking local maxima.
3. `Dropout(0.15)`: randomly disables 15% of activations during training to discourage over-reliance on particular features.
4. `return skip, pool`: return both the detailed feature map and the downsampled tensor.

The skip tensor is important because the decoder will later reuse its spatial details.

### 4.8 `decoder_block(x, skip, filters)`

The decoder restores resolution to create a pixel-level mask.

1. `Conv2DTranspose(..., strides=2)`: learns an upsampling operation that doubles height and width.
2. `Concatenate()([x, skip])`: joins the upsampled tensor with the matching encoder feature map along the channel dimension.
3. `Dropout(0.15)`: regularizes the merged features during training.
4. `double_conv_block(...)`: combines the upsampled context and finer details.
5. Return the refined tensor.

### 4.9 `build_unet(input_shape=(256,256,3), base_filters=16)`

This function assembles the complete model.

- `Input(shape=input_shape, name="sar_image")`: expects one three-channel 256×256 image by default.
- Encoder 1 uses 16 filters; Encoder 2 uses 32; Encoder 3 uses 64; Encoder 4 uses 128.
- Each encoder stage saves a skip tensor and downsamples the other output.
- The bottleneck uses 256 filters at 16×16 resolution.
- Decoder stages use 128, 64, 32, and 16 filters while upsampling back to 256×256.
- `Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")`: maps the final features to one output channel. Sigmoid gives a score from 0 to 1 for every pixel.
- `keras.Model(...)`: packages the graph as a Keras model named `OilSpill_UNet`.
- `model.compile(...)`: selects Adam, the custom BCE+Dice loss, Dice metric, and IoU metric.
- `return model`: returns the compiled model. Compilation prepares it for training/evaluation; it does not train it by itself.

**Why U-Net?** The encoder learns context; the decoder returns to the original resolution; skip connections carry detail from early layers so thin or irregular boundaries are less likely to be lost.

### 4.10 `OilSpillDataGenerator`

This class inherits from `keras.utils.Sequence` and supplies batches rather than loading an entire dataset into memory.

**Constructor (`__init__`)**
- Stores image/mask paths, batch size, target size, augmentation and shuffle settings.
- Lists image filenames from `image_dir` and strips extensions to make IDs.
- Sorts IDs, which makes ordering deterministic before optional shuffling.
- Checks that at least one image exists.
- If `max_samples` is smaller than the available set, selects evenly spaced indices with `np.linspace`; it does not randomly sample the subset.
- Creates an array of indices and optionally shuffles it.

**Important implementation detail:** `EXTS` advertises `.png`, `.jpg`, `.jpeg`, and `.tif`, but `_load()` constructs both paths with a hard-coded `.png` extension. Therefore, non-PNG files can be discovered and then fail to load. In practice, this generator currently expects PNG files with matching names in the image and mask directories.

**`__len__`**
- Returns `len(indexes) // batch_size`, with a minimum return value of 1.
- Because it uses integer division, a leftover partial batch is discarded. If there are fewer samples than the batch size, returning 1 does not guarantee a full batch; the slice may be smaller or empty depending on the count.

**`on_epoch_end()`**
- Shuffles the index order after an epoch if `shuffle=True`.
- It does not change image content or labels.

**`__getitem__(idx)`**
- Slices the index list for the requested batch.
- Loads each matching image and mask.
- Collects them into NumPy arrays of `float32` and returns `(images, masks)` to Keras.

**`_load(img_id)`**
1. Builds image and mask paths from the ID and `.png` extension.
2. Opens the image, converts it to RGB, resizes to 256×256 by default, converts to an array, and divides pixel values by 255.
3. Opens the mask in grayscale (`L`), resizes with nearest-neighbour interpolation, divides by 255, and adds a final channel dimension.
4. If augmentation is enabled, independently decides whether to flip horizontally and vertically.
5. Each selected flip is applied to both image and its corresponding mask. That pairing is essential: a mask must continue to mark the same location as the spill in the transformed image.
6. Returns the image and mask arrays.

### 4.11 `preprocess_image(image, target_size=IMG_SIZE)`

Used for prediction in the dashboard, not for training the dataset generator.

1. Converts the uploaded/sample image to RGB.
2. Resizes it to 256×256.
3. Converts pixels to `float32` and divides by 255 to put them in the 0–1 range.
4. `np.expand_dims(arr, axis=0)` adds the batch dimension.

The resulting shape is `(1, 256, 256, 3)`: batch of one, height, width, channels.

### 4.12 `postprocess_mask(raw_output, threshold=0.5, original_size=None)`

1. `raw_output[0, :, :, 0]` selects the first image in the batch and its single output channel.
2. `prob > threshold` turns probabilities into Boolean values.
3. `.astype(np.uint8) * 255` converts the mask to 0 for background and 255 for detected pixels.
4. If `original_size` is provided, the mask is resized back to the original width and height with nearest-neighbour interpolation.

Nearest-neighbour is suitable for a categorical mask because it does not create blended intermediate class values.

### 4.13 `compute_coverage(mask)`

- Counts the total number of pixels using `mask.size`.
- Counts pixels above 127, which represents the white/detected class in a 0/255 mask.
- Returns `spill_pixels / total_pixels * 100` as a percentage, or 0 if the mask has no pixels.

It reports the proportion of the image marked by the model, not the physical area of oil in square kilometres. That would require georeferencing and image scale information.

### 4.14 `is_spill_detected(coverage, min_pct=0.5)`

Returns `True` if the calculated coverage is at least the configured minimum percentage. It is a simple rule on mask coverage, not a separate trained classifier.

### 4.15 `ERA5_INCIDENT_PRESETS`

This dictionary contains three named scenarios with hard-coded `u10`, `v10`, date, and description values:
- Mumbai High Offshore Spill scenario
- Gulf of Mexico Platform scenario
- Calm-sea false-alarm scenario

The names and descriptions help populate the dashboard. The code does **not** query ERA5 using the selected date, coordinates, or a live API. It simply reads the values already stored in the dictionary.

### 4.16 `compute_era5_wind_metrics(u10, v10)`

- `sqrt(u10**2 + v10**2)`: computes the magnitude of the two perpendicular wind components, in metres per second.
- `speed_ms * 1.94384`: converts metres per second to knots.
- `arctan2` plus degree conversion and modulo 360: calculates meteorological wind direction, using the convention for the direction the wind comes **from**.
- Adding 180 degrees and taking modulo 360: calculates the opposite compass direction, used as a simplified drift-toward direction.
- `speed_ms * 0.032 * 3.6`: estimates a drift speed using 3.2% of wind speed and converts m/s to km/h.
- `compass_sectors`: maps numeric angles to one of 16 compass labels.
- `speed_ms < 3.0`: flags a low-wind look-alike risk.
- The returned dictionary holds speed, direction, compass labels, drift speed, and the risk flag.

**Limitation:** this is a simplified estimate. Real oil drift can also depend on currents, waves, oil properties, and weather. The code produces a drift direction and speed, not a validated time-dependent trajectory or a map of where the spill will arrive.

### 4.17 `CUSTOM_OBJECTS` and `load_saved_model(model_path)`

- `CUSTOM_OBJECTS` maps the names of the custom loss and metrics to their Python functions. This lets Keras resolve them when loading the saved model.
- `os.path.exists(model_path)` checks whether the file exists; if not, the function returns `None`.
- `keras.models.load_model(..., custom_objects=CUSTOM_OBJECTS)` attempts to load the saved model.
- The `try/except` catches loading errors, prints an error, and returns `None` rather than crashing at this point.

**Important app behaviour:** if the model cannot be loaded, the Streamlit app falls back to a simple grayscale threshold. That fallback is not U-Net inference and should never be presented as deep-learning prediction.

---

## 5. `backend/train_model.py` — training flow

### Imports and path setup

- Imports `os` and `sys` for paths, Matplotlib for plotting, TensorFlow, and Keras.
- `SCRIPT_DIR` finds the directory containing this script.
- `PROJECT_ROOT` moves one directory up from `backend/`.
- Adding `PROJECT_ROOT` to `sys.path` helps Python resolve `from backend.utils import ...` when the script is run directly.
- It imports the U-Net builder, data generator, image size, batch size, and epoch count from `utils.py`.
- The print statements show the TensorFlow version and detected GPUs. They report the environment; they do not change the model.

### Dataset paths

The script expects:
- `dataset/images/train`
- `dataset/masks/train`
- `dataset/images/val`
- `dataset/masks/val`

It creates the `outputs` folder if needed. The repository's `.gitignore` excludes `dataset/`, and that folder is not in the current GitHub tree. So a clean clone does not contain the training data expected by this script. The seven demo pairs under `sample_data/` are not automatically substituted for those paths.

### Generators

- Training generator: batch size 8, augmentation enabled, shuffle enabled, and `max_samples=600`.
- Validation generator: batch size 8, augmentation disabled, shuffle disabled, and `max_samples=160`.
- Under the intended counts, this gives 75 training batches and 20 validation batches per epoch.
- The generator selects evenly spaced IDs when limiting the number of samples, rather than choosing a random subset.

### Build model and callbacks

- `build_unet(...)` creates the model and `model.summary()` prints its layers and parameter counts.
- The save path is `models/oil_spill_unet.keras`; its parent directory is created if needed.
- `ModelCheckpoint` watches `val_dice_coefficient`, maximizes it, and saves only when it improves.
- `EarlyStopping` watches validation loss and is configured to stop after five unimproved epochs, restoring best weights. Since the maximum run is only five epochs, this patience setting is unlikely to stop training early in a five-epoch run.
- `ReduceLROnPlateau` halves the learning rate after two unimproved validation-loss epochs, subject to a minimum learning rate of `1e-6`.

### `model.fit(...)`

This is the actual training call. Keras repeatedly:
1. requests batches from `train_gen`;
2. performs a forward pass;
3. computes the combined loss;
4. calculates gradients through backpropagation;
5. updates parameters with Adam;
6. checks performance on `val_gen` at the end of each epoch;
7. runs the callbacks.

### Training plot and evaluation

- The first Matplotlib plot shows training and validation loss over epochs. Lower is generally better.
- The second plot shows training and validation Dice. Higher is generally better.
- `plt.savefig(...)` writes `outputs/training_curves.png`.
- `model.evaluate(val_gen)` calculates final loss, Dice, and IoU and prints them.

**Documentation discrepancy:** the current `train_model.py` code does not write `outputs/training_metrics.json`. That JSON file is a committed report/artifact, but this script only saves the plot and model checkpoint.

---

## 6. `frontend/app.py` — Streamlit dashboard

### Imports and project paths

- `os`, `sys`: path operations and Python import path setup.
- `io`: in-memory byte buffer for a PNG download.
- `numpy`: array operations and grayscale fallback.
- `PIL.Image`: image reading/conversion.
- `streamlit as st`: creates the web UI.
- The project-root logic makes `backend.utils` importable when running the app from the project.
- It imports preprocessing, postprocessing, coverage, weather metrics, presets, and model-loading helpers from `utils.py`.

### `create_visual_overlay(image, mask, color=(220,30,30), alpha=0.45)`

- Converts the image to floating point.
- Defines detected pixels as mask values greater than 127.
- Copies the original image.
- For each RGB channel, blends detected pixels with a red colour at 45% opacity while leaving non-detected pixels unchanged.
- Clips values to 0–255 and returns `uint8` for display.

The red overlay is only a visualisation of the predicted mask; it is not additional evidence that the detection is correct.

### `mask_to_png_bytes(mask)`

- Creates an in-memory `BytesIO` buffer.
- Converts the array to a PIL image and encodes it as PNG.
- Returns the encoded bytes so Streamlit can serve a download without writing a temporary file.

### Page configuration and paths

- `st.set_page_config(...)` sets the browser tab title, icon, and wide layout.
- `MODEL_PATH` points to the saved `.keras` file.
- `SAMPLE_DIR` points to `sample_data/images`.

### `get_model()` and caching

- The `@st.cache_resource` decorator caches the loaded model resource across Streamlit reruns.
- `get_model()` calls `load_saved_model(MODEL_PATH)`.
- The sidebar shows whether a trained model loaded or whether the checkpoint is missing/unreadable.

### Sidebar controls

- `px_thresh` controls the per-pixel probability threshold (0.1 to 0.9, default 0.5).
- `cov_thresh` controls the minimum marked coverage required for an alert (0.1% to 5%, default 0.5%).
- `preset_name` selects one of the hard-coded scenarios.
- `wind_u10` and `wind_v10` are taken directly from that preset's dictionary entry.
- Date and description are displayed as text. They do not cause a weather API request.

### Image selection/upload

- The app lists supported files in `sample_data/images`.
- A radio button selects between a benchmark image and a custom upload.
- The benchmark path opens the selected file and converts it to RGB.
- The upload path accepts PNG, JPG, or JPEG and also converts the image to RGB.
- `orig_w, orig_h` save the original dimensions; `img_arr` stores the image as a NumPy array for overlay rendering.

The benchmark labels are selected by filename substring (`1005`, `1009`, `1014`, or `100`). This is a naming heuristic, not a model prediction or a ground-truth lookup.

### What happens when Analyze is clicked

1. If a model loaded, `preprocess_image()` prepares the input and `model.predict()` generates the per-pixel scores.
2. `postprocess_mask()` thresholds the scores and resizes the mask back to the original image size.
3. **If no model loaded**, the app instead converts the image to grayscale and marks pixels below intensity 70. That is a basic dark-pixel rule, not deep learning.
4. `compute_coverage()` calculates the percentage of marked pixels.
5. `is_spill_detected()` compares coverage with the UI's minimum threshold.
6. `create_visual_overlay()` creates the red-highlighted image.
7. `compute_era5_wind_metrics()` calculates wind/drift values from the selected preset.
8. If a spill-like mask is present and wind speed is below 3 m/s, the app displays a likely-look-alike warning. Otherwise it displays a potential-spill message or a no-significant-spill message.
9. Four metric cards show coverage, wind speed, wind origin, and projected drift.
10. Three panels show original image, binary mask, and red overlay.
11. `mask_to_png_bytes()` and `st.download_button()` let the user download the predicted mask as a PNG.

**Crucial wording:** the UI itself says “POTENTIAL OIL SPILL DETECTED”, not scientifically confirmed oil. Low wind is only a warning signal; the current code does not inspect chemical composition or verify a real spill with independent evidence.

---

## 7. Saved model, sample data, outputs, and dependencies

### `models/oil_spill_unet.keras`

This is a saved Keras model artifact, not source code to read line by line. The accompanying JSON reports 1,945,521 total parameters, 1,941,681 trainable parameters, and 3,840 non-trainable parameters. The `.keras` file is approximately 23.6 million bytes (reported in the notes as about 22.54 MB). It contains the trained network state needed for inference.

### `sample_data/`

There are seven image files and seven same-named masks. The masks are useful as reference labels, but the current dashboard does not compare predictions against them or calculate per-sample accuracy when you click Analyze. The seven sample pairs are demonstration assets, not the 600 training pairs or 160 validation pairs reported in the training summary.

### `outputs/training_metrics.json`

The stored summary reports:
- 6,455 training pairs and 1,615 validation pairs available in the dataset split;
- 600 training samples and 160 validation samples used in the recorded run;
- 5 epochs;
- final validation loss 1.3488;
- final validation Dice 0.3210;
- final validation IoU 0.2232.

The dataset counts describe the reported training setup, but the corresponding large `dataset/` directory is not committed in the repository. The metrics are a saved report, not proof that retraining can run from a clean clone without obtaining the dataset.

### Interpreting validation performance honestly

A validation Dice of 0.321 and IoU of 0.223 indicate that predicted and reference spill regions have limited overlap on the reported validation run. The score improved across the five recorded epochs, but it is not strong evidence of deployment-ready detection. A useful next step would be evaluation on a larger, clearly documented held-out test set, per-scene metrics, threshold tuning, and inspection of false positives and false negatives.

### `requirements.txt`

It lists TensorFlow, Streamlit, NumPy, Pillow, Matplotlib, scikit-learn, OpenCV, Seaborn, pandas, and tqdm with minimum versions. Some listed packages are not directly imported in the three main Python files, but may be intended for the broader course environment. The TensorFlow minimum is broad (`>=2.13.0`), so reproducibility would improve with a tested, pinned environment or lock file.

### `.gitignore`

The key line is `dataset/`, which excludes the local dataset from Git. This helps avoid committing large data files, but it means anyone cloning the repository must separately obtain and arrange the dataset before running `backend/train_model.py`.

---

## 8. Deep-learning concepts to explain simply

### What is a tensor?
A tensor is a multidimensional array. One RGB image is shaped `(256, 256, 3)`. A batch of eight is `(8, 256, 256, 3)`. The prediction for a batch of one is `(1, 256, 256, 1)`.

### What is convolution?
A small learned filter slides across the image and responds to local patterns such as edges and texture. Many filters create many feature maps.

### What is an encoder?
The contracting part of U-Net. It extracts increasingly abstract features while reducing spatial size from 256 to 128, 64, 32, and 16 pixels.

### What is the bottleneck?
The deepest representation, here 16×16 with 256 channels. It holds compressed contextual features before the decoder reconstructs the mask.

### What is a decoder?
The expanding part of U-Net. It upsamples feature maps and combines them with encoder features to produce a full-resolution prediction.

### What is a skip connection?
It concatenates an encoder feature map with the matching decoder feature map. It helps preserve fine spatial details that repeated pooling could lose.

### Why ReLU in hidden layers?
ReLU returns `max(0, x)`. It is simple, adds non-linearity, and has a gradient of 1 for positive inputs. It does not guarantee that vanishing gradients can never occur.

### Why Sigmoid at the output?
This is binary segmentation with one output channel. Sigmoid maps each pixel's score to 0–1, which the app then thresholds. The values should be called model scores or probabilities as model outputs; do not claim they are perfectly calibrated probabilities without calibration testing.

### Why BCE + Dice loss?
BCE penalizes wrong pixel probabilities. Dice loss encourages overlap with the foreground mask. Combining them is useful when the oil region is much smaller than the ocean background.

### Why Adam?
Adam adapts parameter updates using moving averages of gradients and squared gradients. It is a common practical optimizer and often works well without as much manual tuning as basic SGD.

### Why Batch Normalization?
It normalizes activations using batch statistics during training and moving statistics at inference. This can stabilize optimization.

### Why Dropout?
It randomly disables a fraction of activations during training to reduce reliance on specific features. Dropout is disabled during normal inference.

### Why He Normal initialization?
It initializes weights with ReLU layers in mind, helping maintain activation scale as signals pass through the network.

### What is an epoch?
One pass through all batches that the generator exposes for training. With 600 samples and batch size 8, the reported setup has 75 batches per epoch.

### What is a learning rate?
The scale of the optimizer's parameter update. Here Adam starts at 0.0001; `ReduceLROnPlateau` can lower it if validation loss stops improving.

### What is backpropagation?
The method used to calculate how each trainable parameter contributed to the loss. TensorFlow handles the gradients during `model.fit()`.

---

## 9. SAR and wind physics in plain English

### How SAR images can show oil
Synthetic Aperture Radar sends microwave energy and measures the energy returning to the sensor. Wind-generated small surface waves scatter radar energy back. An oil film can dampen those small waves, reducing backscatter and making a slick appear dark.

### What is a look-alike?
A dark patch that resembles oil but has another cause. Calm seas, wind shadows, and natural biological slicks can all create low-backscatter regions. A model trained on limited examples may confuse them.

### What do `u10` and `v10` mean?
They are the eastward and northward components of wind at 10 metres above the surface. Their vector magnitude is `sqrt(u10² + v10²)`.

### What does the code's 3.2% rule mean?
It estimates a simplified drift speed as 0.032 times wind speed, then converts it to km/h. This is an empirical approximation, not a full oceanographic forecast. A more realistic model would consider currents, waves, oil properties, and uncertainty.

### What does the low-wind warning mean?
If calculated wind speed is below 3 m/s and the model marks a region, the UI warns that calm water might explain the dark patch. It does not prove the prediction is false; it flags a condition that deserves extra checking.

---

## 10. Important corrections to the uploaded explanation notes

These are worth knowing because a professor may ask whether the written claims match the code.

1. **No live ERA5 retrieval:** `ERA5_INCIDENT_PRESETS` contains fixed values. The app does not download or query ERA5 weather for an image's location/time.
2. **No guaranteed confirmation:** the low-wind rule is a heuristic. It cannot eliminate all false alarms or confirm oil chemically.
3. **Training data missing from GitHub:** `train_model.py` expects a `dataset/` folder that is ignored and absent from the repository tree. The seven `sample_data/` pairs do not automatically satisfy those training paths.
4. **Training metrics JSON is not written by the training script:** `train_model.py` saves the curve plot and model checkpoint, and prints evaluation values, but the shown code does not export `training_metrics.json`.
5. **Fallback is not deep learning:** if loading the saved model fails, the app uses `gray < 70` to mark dark pixels. The resulting mask is a simple intensity threshold, not U-Net output.
6. **Validation score is modest:** final reported validation Dice is 0.3210 and IoU is 0.2232. Say the model is a prototype and acknowledge limited overlap; do not claim high accuracy or operational reliability.
7. **“Real-time” is too strong:** prediction runs on demand for an image selected by the user. There is no live satellite feed or automatic weather API in the current code.
8. **Generator extension mismatch:** it lists JPEG/TIFF extensions but constructs `.png` paths in `_load()`. Non-PNG inputs in the training directories may fail.
9. **Partial batches are dropped:** `__len__` uses floor division. The remainder is not included in a normal epoch.
10. **Metric nuance:** Dice and IoU are calculated on flattened probability tensors during training, not necessarily on binary masks thresholded at 0.5. Avoid saying the reported training metric is identical to a thresholded, per-image segmentation score.
11. **Training duration is small:** only five epochs and 600/160 samples are used in the recorded configuration. It is a compact academic run, not exhaustive model training.
12. **Dataset selection:** when `max_samples` is applied, the generator selects evenly spaced IDs from sorted filenames; it does not randomly choose a subset. This can affect how representative the subset is.

---

## 11. Viva questions and short, defensible answers

**Q1. What is the project?**  
It is a prototype that segments potential oil-spill regions in SAR images using U-Net and adds simple wind-context calculations in a Streamlit dashboard.

**Q2. Why use deep learning?**  
A CNN can learn combinations of local texture, shape, and context from labelled examples instead of relying only on a hand-written dark-pixel rule.

**Q3. Why U-Net?**  
It is designed for pixel-level segmentation. Its encoder captures context, its decoder restores resolution, and skip connections preserve spatial detail.

**Q4. What is the input and output shape?**  
Input `(256, 256, 3)`; output `(256, 256, 1)` for one image, with a batch dimension present during model execution.

**Q5. Why three input channels if SAR is often grayscale?**  
The code converts images to RGB, so the model expects three channels. This may simply replicate grayscale intensity into three channels; it does not automatically mean the input contains three distinct radar bands.

**Q6. Why use ReLU?**  
It adds non-linearity, is computationally simple, and has a non-saturating gradient for positive inputs.

**Q7. Why use Sigmoid at the output?**  
Each pixel is classified as foreground or background, so one output score per pixel can be thresholded into a binary mask.

**Q8. What is the loss?**  
Binary Cross-Entropy plus Dice loss. BCE gives pixel-wise feedback, while Dice emphasizes overlap with the small foreground region.

**Q9. What is Dice?**  
A mask-overlap measure: `2 × intersection / (predicted positives + true positives)` with smoothing in this implementation. Higher is better.

**Q10. What is IoU?**  
Intersection divided by union, with smoothing. It measures how much the predicted and true regions overlap.

**Q11. Why Adam?**  
It adapts updates using moving averages of gradients and squared gradients, making it a practical starting optimizer.

**Q12. Why use augmentation?**  
Flipping can expose the model to different orientations. Image and mask must receive exactly the same flip so labels remain aligned.

**Q13. What is a skip connection?**  
It concatenates encoder features with decoder features at the same resolution, helping recover fine boundary information.

**Q14. What is the difference between training and inference?**  
Training updates weights using labelled images and loss; inference uses an already trained model to predict a mask for a new image.

**Q15. What does the threshold slider do?**  
It changes the minimum model score required to mark a pixel as spill. A lower threshold usually marks more pixels; a higher threshold usually marks fewer.

**Q16. What does coverage mean?**  
The percentage of image pixels classified as spill. It is not the real-world area of oil unless the image's geographic scale is known.

**Q17. What is SAR?**  
Synthetic Aperture Radar is an active radar imaging technique that measures microwave backscatter and can image in darkness and through many cloud conditions.

**Q18. Why can oil appear dark?**  
Oil can dampen small surface waves that would normally scatter radar energy back to the sensor, reducing backscatter.

**Q19. What is a look-alike?**  
A non-oil phenomenon, such as calm water, that produces a dark radar signature resembling an oil slick.

**Q20. What are `u10` and `v10`?**  
Eastward and northward 10-metre wind components. The code combines them to calculate wind speed and direction.

**Q21. Does the project get live ERA5 data?**  
No. The current app uses fixed values stored in three preset scenarios. Live ERA5 retrieval would be a future extension.

**Q22. Does it predict the exact future path of the spill?**  
No. It estimates a drift direction and speed from a simplified wind percentage. It does not run a time-dependent trajectory simulation or include ocean currents.

**Q23. What is the reported model performance?**  
The saved metrics report validation Dice 0.3210 and IoU 0.2232 after five epochs. That indicates limited overlap and needs improvement before any operational use.

**Q24. Can I retrain it immediately after cloning?**  
Not from the repository alone. The training script expects a separate `dataset/` directory with train/validation images and masks, and that folder is not included in the repository.

**Q25. What happens if the model file cannot be loaded?**  
The app falls back to a grayscale threshold that marks pixels below intensity 70. That is a simple image-processing fallback, not U-Net inference.

**Q26. What would you improve next?**  
First, provide reproducible dataset setup and test-set evaluation; then improve Dice/IoU, inspect false positives and false negatives, connect to real geolocated weather data, and use currents and uncertainty for drift estimation.

---

## 12. A safe answer if asked about limitations

“The main limitation is validation quality and data reproducibility. The saved report gives a validation Dice of 0.321 and IoU of 0.223, so I would not claim production-level accuracy. The current repository also does not include the large training dataset required by the training script, and the weather values are predefined presets rather than live ERA5 retrieval. The next step would be to make training reproducible, evaluate on a separate test set, improve segmentation performance, and integrate verified weather and ocean-current data.”

This answer is stronger than overstating the project because it shows that you understand both the implementation and its limitations.

---

## 13. Best way to study this project

1. Memorize the 30-second explanation, but understand each sentence.
2. Learn the data flow: image → preprocessing → U-Net prediction → threshold mask → coverage → weather metrics → dashboard.
3. Study `utils.py` first, because it contains the model and most helper functions.
4. Study `train_model.py` second to understand how weights are learned and saved.
5. Study `app.py` last to understand how the trained model is used in an interface.
6. Practise explaining the U-Net shapes: 256 → 128 → 64 → 32 → 16, then back to 256.
7. Be ready to state the measured Dice/IoU values and the limits of the preset weather calculations.
8. Never claim you personally implemented or trained something unless that is true for your contribution.
