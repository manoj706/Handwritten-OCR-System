import cv2
import numpy as np
from deskew import determine_skew
from typing import Tuple, Optional

def grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def denoise(image: np.ndarray) -> np.ndarray:
    """Applies Gaussian Blur for noise removal."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """Applies thresholding to binarize the image."""
    if method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

def correct_skew(image: np.ndarray) -> np.ndarray:
    """Corrects skew in the image."""
    grayscale_image = grayscale(image)
    angle = determine_skew(grayscale_image)
    if angle is None:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def remove_borders(image: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Crops the percentage of borders to remove scanning artifacts/bindings."""
    h, w = image.shape[:2]
    # Crop Left and Right more aggressively for spiral bindings
    x_start = int(w * margin)
    x_end = int(w * (1 - margin))
    y_start = int(h * (margin / 2)) # Less crop on top/bottom
    y_end = int(h * (1 - (margin / 2)))
    
    return image[y_start:y_end, x_start:x_end]

def preprocess_image(image: np.ndarray, denoise_strength: int = 1, method: str = 'adaptive') -> np.ndarray:
    """Full preprocessing pipeline: Grayscale -> Denoise -> Threshold."""
    # 1. Grayscale
    gray = grayscale(image)
    
    # 2. Aggressive Denoise
    # Median blur is highly effective against salt-and-pepper noise dots
    gray = cv2.medianBlur(gray, 3)
    for _ in range(denoise_strength):
        gray = denoise(gray)
        
    # 3. Threshold (binarize)
    if method == 'clean':
        # Combined approach: blur + otsu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Better adaptive parameters: block size 15, constant 10 to suppress more noise
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
    
    # 4. Post-threshold noise removal (Morphological opening)
    # Increase kernel size for more aggression
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.bitwise_not(binary)
    opened = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
    binary = cv2.bitwise_not(opened)
    
    # 5. Marginal Crop (remove top/bottom 3% to kill edge noise)
    h, w = binary.shape
    top_crop = int(h * 0.03)
    bot_crop = int(h * 0.03)
    # Make sure we don't crop everything
    if top_crop + bot_crop < h * 0.5:
        binary[0:top_crop, :] = 255 # Fill with white (background)
        binary[h-bot_crop:h, :] = 255
    
    return binary
