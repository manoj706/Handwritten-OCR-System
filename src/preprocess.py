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

def preprocess_image(image: np.ndarray, denoise_strength: int = 1) -> np.ndarray:
    """Full preprocessing pipeline: Grayscale -> Denoise -> Threshold -> Skew Correction."""
    # 0. Remove Borders (Scan artifacts)
    # image = remove_borders(image, margin=0.08) # Remove 8% from left/right
    # Let's do this on grayscale to save compute
    
    # 1. Grayscale
    gray = grayscale(image)
    
    # Border removal disabled — preserves full image
    # gray = remove_borders(gray, margin=0.08)
    
    # 2. Denoise
    # Multiple passes for stronger denoising if needed
    for _ in range(denoise_strength):
        gray = denoise(gray)
        
    # 3. Threshold (binarize) — use adaptive for complex images with backgrounds
    binary = threshold(gray, method='adaptive')
    
    # 4. Skew Correction (optional, can be expensive or error prone on small text)
    # corrected = correct_skew(binary) 
    
    return binary
