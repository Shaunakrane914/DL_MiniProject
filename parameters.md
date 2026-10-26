
Got it — you want all the important parameters and settings used throughout the project, not just the total number of learnable parameters.

Here is the overall parameter sheet for your oil spill detection project.

## 1. Model architecture parameters

| Parameter               | Value in your project                            | Purpose                                                |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| Model architecture      | U-Net                                            | Performs pixel-level image segmentation.               |
| Input image size        | `256 × 256`                                   | Standardizes image dimensions.                         |
| Input channels          | `3`                                            | RGB input representation.                              |
| Base filters            | `16`                                           | Sets the starting feature-map count.                   |
| Encoder filters         | `16, 32, 64, 128`                              | Extract features at progressively smaller resolutions. |
| Bottleneck filters      | `256`                                          | Learns deeper, higher-level features.                  |
| Convolution kernel size | `3 × 3`                                       | Detects local image patterns.                          |
| Pooling size            | `2 × 2`                                       | Reduces spatial dimensions.                            |
| Upsampling              | Transposed convolution, `2 × 2`, stride `2` | Increases spatial dimensions in the decoder.           |
| Output channels         | `1`                                            | Predicts one spill score per pixel.                    |
| Hidden activation       | ReLU                                             | Adds non-linearity.                                    |
| Output activation       | Sigmoid                                          | Produces pixel scores between 0 and 1.                 |
| Weight initialization   | He Normal                                        | Initializes weights for ReLU-based layers.             |
| Convolution bias        | Disabled                                         | `use_bias=False` in convolution layers.              |
| Batch Normalization     | Enabled                                          | Normalizes activations.                                |
| Dropout                 | `0.15`                                         | Regularization to help reduce overfitting.             |

## 2. Training hyperparameters

| Parameter                  | Value                                | Purpose                                             |
| -------------------------- | ------------------------------------ | --------------------------------------------------- |
| Optimizer                  | Adam                                 | Updates model weights.                              |
| Learning rate              | `0.0001` (`1e-4`)                | Controls update step size.                          |
| Batch size                 | `8`                                | Image-mask pairs processed per training batch.      |
| Maximum epochs             | `5`                                | Maximum training passes through the selected data.  |
| Image size                 | `256 × 256`                       | Training image resolution.                          |
| Maximum training samples   | `600`                              | Limits training data used by the training script.   |
| Maximum validation samples | `160`                              | Limits validation data used by the training script. |
| Training shuffle           | Enabled                              | Changes sample order between epochs.                |
| Training augmentation      | Random horizontal and vertical flips | Provides variation in training images.              |
| Validation augmentation    | Disabled                             | Keeps validation examples unaugmented.              |

## 3. Loss function and evaluation parameters

| Parameter                        | Value or method | Purpose                                                |
| -------------------------------- | --------------- | ------------------------------------------------------ |
| Training loss                    | BCE + Dice Loss | Combines pixel-level and region-overlap errors.        |
| Dice smoothing                   | `1.0`         | Stabilizes the Dice calculation when masks are empty.  |
| Dice coefficient                 | Custom metric   | Measures predicted versus actual mask overlap.         |
| IoU score                        | Custom metric   | Measures intersection relative to union.               |
| Model checkpoint metric          | Validation Dice | Saves the model when validation Dice improves.         |
| Early stopping monitor           | Validation loss | Stops training if validation loss does not improve.    |
| Early stopping patience          | `5` epochs    | Sets how long to wait for improvement.                 |
| Learning-rate reduction monitor  | Validation loss | Reduces learning rate when progress plateaus.          |
| Learning-rate reduction factor   | `0.5`         | Multiplies the learning rate by one-half.              |
| Learning-rate reduction patience | `2` epochs    | Number of epochs without improvement before reduction. |
| Minimum learning rate            | `1e-6`        | Lower bound for learning-rate reduction.               |

## 4. Image preprocessing and prediction parameters

| Parameter                 | Value                          | Purpose                                                 |
| ------------------------- | ------------------------------ | ------------------------------------------------------- |
| Image color mode          | RGB                            | Ensures three input channels.                           |
| Input pixel range         | `0–255`                     | Typical image intensity range before normalization.     |
| Normalization             | Divide by `255.0`            | Converts values to the range `0–1`.                  |
| Model input tensor        | `(1, 256, 256, 3)`           | Includes the batch dimension.                           |
| Raw model output          | `(1, 256, 256, 1)`           | One predicted score per output pixel.                   |
| Pixel detection threshold | `0.5` by default             | Converts scores to binary classifications.              |
| Positive mask value       | `255`                        | Represents detected spill pixels in the displayed mask. |
| Background mask value     | `0`                          | Represents pixels not classified as spill.              |
| Mask resizing             | Nearest-neighbor interpolation | Preserves discrete mask labels when resizing.           |

## 5. Application and oil-spill alert parameters

These settings belong to the Streamlit application and its weather calculations, rather than the U-Net architecture itself.

| Parameter                     | Value                  | Purpose                                                      |
| ----------------------------- | ---------------------- | ------------------------------------------------------------ |
| Default coverage threshold    | `0.5%`               | Minimum predicted spill area required for an alert.          |
| Adjustable pixel threshold    | `0.1–0.9`           | Lets the user change pixel-level sensitivity.                |
| Adjustable coverage threshold | `0.1–5.0%`          | Lets the user change the minimum area required for an alert. |
| Low-wind look-alike threshold | `3 m/s`              | Flags calm-sea conditions that can resemble oil spills.      |
| Wind drift factor             | `3.2%` of wind speed | Estimates surface drift speed from wind.                     |
| Overlay color                 | RGB `(220, 30, 30)`  | Red highlighting for predicted spill regions.                |
| Overlay opacity               | `0.45`               | Controls the strength of the red overlay.                    |
| Model caching                 | `@st.cache_resource` | Avoids repeatedly loading the trained model.                 |

The weather calculations use predefined wind scenarios in the application; they are not live ERA5 data retrieval.

## 6. Main categories to remember

Learnable parameters

Convolution weights and Batch Normalization's trainable gamma and beta.

Hyperparameters

Learning rate, batch size, epochs, dropout rate, base filters and image size.

Architecture parameters

Kernel sizes, filter counts, pooling, upsampling, input channels and output channels.

Inference and application parameters

Pixel threshold, coverage threshold, wind threshold and overlay opacity.

Important: Some parameters overlap categories. For example, image size is both a preprocessing setting and a model configuration setting; the pixel threshold is an inference setting, not a learnable parameter. This list covers the major settings documented for your project, rather than every variable or function argument in the source code.
