import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .preprocess import threshold, grayscale

def project_horizontal(image: np.ndarray) -> np.ndarray:
    """Computes the horizontal projection profile of the image."""
    # Sum along rows (axis 1). Text is black (0) usually, but we work with binary where text is white (255)
    # If text is black, we invert first.
    # Assuming input is binary (0=bg, 255=text)
    return np.sum(image, axis=1)

def find_peaks(projection: np.ndarray, threshold_val: int = 10) -> List[tuple]:
    """Finds peak regions (start, end) in the projection profile."""
    peaks = []
    start = None
    for i, val in enumerate(projection):
        if val > threshold_val and start is None:
            start = i
        elif val <= threshold_val and start is not None:
            peaks.append((start, i))
            start = None
    if start is not None:
        peaks.append((start, len(projection)))
    return peaks

def segment_lines(image: np.ndarray) -> List[np.ndarray]:
    """Segments a page image into lines using horizontal projection."""
    gray = grayscale(image)
    binary = threshold(gray) # Text is 0, BG is 255 usually
    
    # Invert for projection analysis (Text should be signal=255, BG=0)
    binary_inv = cv2.bitwise_not(binary)
    
    # Horizontal projection
    proj = project_horizontal(binary_inv)
    
    # Find separate lines
    # Normalize projection to be safe
    MAX_VAL = np.max(proj)
    if MAX_VAL == 0: return []
    
    # Simple peak finding logic
    # Using mean + std? Or just a lower static %
    # If text is sparse, mean might be low. 
    # Let's try to just find non-zero regions robustly.
    peaks = find_peaks(proj, threshold_val=MAX_VAL * 0.01) # Lower threshold
    
    lines = []
    for start, end in peaks:
        height = end - start
        
        # 1. Stricter Minimum Height Filter: Skip segments < 20 pixels high
        # Most handwriting lines are at least 30-50px. 20px safely catches noise.
        if height < 20:
            continue
            
        # Extract line with some padding
        pad = 2
        y1 = max(0, start - pad)
        y2 = min(image.shape[0], end + pad)
        line_img = image[y1:y2, :]
        
        # 2. Stricter Text Density Filter: Skip segments with very little "text" content
        line_binary = binary_inv[y1:y2, :]
        text_pixel_count = np.count_nonzero(line_binary)
        total_pixels = line_binary.shape[0] * line_binary.shape[1]
        density = text_pixel_count / total_pixels
        
        # If density < 0.5%, it's likely just scattered noise dots
        if density < 0.005:
            continue
            
        lines.append(line_img)
        
    return lines

def segment_words(line_image: np.ndarray) -> List[np.ndarray]:
    """Segments a line image into words using vertical projection."""
    gray = grayscale(line_image)
    binary = threshold(gray)
    
    # Invert for projection
    binary_inv = cv2.bitwise_not(binary)
    
    # Vertical projection (sum along columns, axis 0)
    proj = np.sum(binary_inv, axis=0)
    
    MAX_VAL = np.max(proj)
    if MAX_VAL == 0: return []
    
    # Find words (gaps between words)
    peaks = find_peaks(proj, threshold_val=MAX_VAL * 0.05)
    
    words = []
    for start, end in peaks:
         # Extract word with padding
        pad = 2
        x1 = max(0, start - pad)
        x2 = min(line_image.shape[1], end + pad)
        words.append(line_image[:, x1:x2])
        
    return words
