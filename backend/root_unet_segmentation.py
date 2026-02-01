import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# U-NET ARCHITECTURE
# --------------------------------------------------
def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# --------------------------------------------------
# INFERENCE & ANALYTICS
# --------------------------------------------------
def segment_root_unet(image_path):
    """
    Attempts to segment root using U-Net.
    Returns: dictionary of metrics or None if failed.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Preprocess
        original_h, original_w = img.shape
        img_resized = cv2.resize(img, (256, 256))
        img_input = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)

        # Load weights (Mock path for now, safe fallback)
        weights_path = os.path.join(os.path.dirname(__file__), "..", "model", "root_unet.h5")
        
        if os.path.exists(weights_path):
            model = build_unet()
            model.load_weights(weights_path)
            pred_mask = model.predict(img_input, verbose=0)[0, :, :, 0]
        else:
            # Safe Fallback: Otsu Thresholding "simulating" deep learning mask
            # The prompt says "Trigger ONLY IF...". If model missing, we fallback.
            # But here we return a fallback result that is "better" than nothing? 
            # Or we return None to signal "use default RootNav".
            # Requirement: "If any new model or feature fails -> fallback to existing behavior"
            # So returning None here is safer to let caller handle fallback.
            # echo print for debugging
            # print("U-Net weights not found. Using fallback.")
            return None 

        # Post-process mask
        pred_mask = cv2.resize(pred_mask, (original_w, original_h))
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Extract Metrics from U-Net Mask
        total_pixels = binary_mask.size
        root_pixels = np.count_nonzero(binary_mask)
        density = (root_pixels / total_pixels) * 100

        # Skeletonization for length/branching
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_mask > 0)
        skeleton_pixels = np.count_nonzero(skeleton)
        
        # Branch points (junctions)
        # Simple convolution to find junctions in skeleton
        # (Assuming skeleton is 1-pixel wide)
        # Kernels for cross/T-junctions...
        # For simplicity/speed, we approximate branch count with skeleton size / factor
        # or implement a kernel check.
        
        # Custom kernel approach for branch points
        skel_uint8 = skeleton.astype(np.uint8)
        # kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        # filtered = cv2.filter2D(skel_uint8, -1, kernel)
        # branch_points = np.sum(filtered >= 13) # Center (10) + at least 3 neighbors (3) -> 13
        
        # Let's use existing logic from root_traits_extractor if possible or keep it simple.
        # We will return the mask-derived metrics.
        
        branch_count = 0 # Placeholder if we don't do complex convolution here
        
        # Total Length Estimate (pixels)
        total_length_px = skeleton_pixels 
        
        return {
            "unet_density": density,
            "unet_total_length_px": total_length_px,
            "unet_mask_available": True
        }

    except Exception as e:
        print(f"⚠️ U-Net Segmentation failed: {e}")
        return None
