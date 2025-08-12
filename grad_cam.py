
# DUMMY GRAD_CAM

import cv2
import numpy as np

def grad_cam_no_model_rice_leaf(img_bgr):
    """
    Highlight diseased regions in a rice leaf image based on color heuristics (no model used).

    Args:
        img_bgr: Input image in BGR format (as loaded by OpenCV).

    Returns:
        heatmap: Grayscale mask representing importance of diseased area (normalized).
        superimposed_img: Original image with heatmap overlay highlighting disease.
    """

    # Convert BGR to HSV for better color segmentation
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define color ranges for healthy green (rice leaf) and diseased areas (yellow to gray)
    # Healthy green range (approximate)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Diseased areas: yellow and gray shades (approximate HSV ranges)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    # Gray can be low saturation and mid to high value in HSV
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])

    # Create masks for healthy, yellow and gray areas
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)

    # Combine yellow and gray masks as potential disease areas
    disease_mask = cv2.bitwise_or(mask_yellow, mask_gray)

    # Optional: Remove small noise via morphological operations
    kernel = np.ones((5,5), np.uint8)
    disease_mask_clean = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    disease_mask_clean = cv2.morphologyEx(disease_mask_clean, cv2.MORPH_DILATE, kernel)

    # Normalize mask to range 0-1 as heatmap
    heatmap = disease_mask_clean.astype(float) / 255.0

    # Convert heatmap to color heatmap using colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image (weighted)
    superimposed_img = cv2.addWeighted(img_bgr, 0.7, heatmap_color, 0.4, 0)

    return heatmap, superimposed_img

import matplotlib.pyplot as plt

def plot_grad_cam_output(heatmap, superimposed_img, title_prefix="Grad-CAM"):
    """
    Plots the heatmap and the superimposed image side by side.

    Args:
        heatmap: The heatmap array (2D or 3D).
        superimposed_img: The original image with heatmap overlay (BGR or RGB).
        title_prefix: Prefix for plot titles.
    """
    plt.figure(figsize=(10, 4))

    # Plot heatmap
    plt.subplot(1, 2, 1)
    if heatmap.ndim == 2:  # Grayscale heatmap
        plt.imshow(heatmap, cmap='jet')
    else:  # Color heatmap in BGR or RGB
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_prefix} Heatmap")
    plt.axis('off')

    # Plot superimposed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_prefix} Superimposed")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
img=cv2.imread("Dataset/DB1/archive/Bacterialblight/BACTERAILBLIGHT3_007.jpg")
a,b=grad_cam_no_model_rice_leaf(img)
plot_grad_cam_output(a,b)


# ORIGINAL GRAD_CAM


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def grad_cam_with_model(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for a classification model.

    Args:
        model: A Keras model.
        img_array: Preprocessed input image array of shape (1, H, W, C).
        last_conv_layer_name: (optional) Name of the last conv layer. If None, function finds it.
        pred_index: (optional) Index of the predicted class. If None, uses highest prediction.

    Returns:
        heatmap: Grad-CAM heatmap (H, W) scaled to [0, 1].
        superimposed_img: Original image with heatmap overlay.
    """

    # Find the last convolutional layer automatically if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name or 'Conv' in layer.name:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in the model.")

    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute the gradient of the class output value with respect to feature map activations
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute the guided gradients (mean intensity of gradients for each feature map channel)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in feature map array by its corresponding importance weight
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Relu on heatmap for positive importance only
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap to range [0, 1]
    max_heat = tf.reduce_max(heatmap)
    if max_heat == 0:
        max_heat = tf.constant(1e-10)  # Avoid division by zero
    heatmap /= max_heat

    heatmap = heatmap.numpy()

    # Rescale heatmap to original image size
    img = img_array[0]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert original image to uint8 RGB if needed (assuming input is normalized 0-1)
    if img.max() <= 1.0:
        img_uint8 = np.uint8(255 * img)
    else:
        img_uint8 = np.uint8(img)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img_uint8, 0.6, heatmap_rgb, 0.4, 0)

    return heatmap, superimposed_img
