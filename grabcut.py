import cv2
import numpy as np


def apply_grabcut(img, rect, iter_count=1):
    """
    Apply GrabCut algorithm to segment foreground from background.
    
    Args:
        img (np.ndarray): Input color image for segmentation
        rect (tuple): Rectangle coordinates (x, y, width, height) defining initial foreground region
        iter_count (int): Number of iterations for GrabCut algorithm (default: 1)
        
    Returns:
        np.ndarray: Binary mask where 255 represents foreground and 0 represents background
    """
    # Initialize mask and model arrays
    mask = np.zeros(img.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut algorithm with rectangle initialization
    cv2.grabCut(img, mask, rect, bg_model, fg_model, iter_count, cv2.GC_INIT_WITH_RECT)
    
    # Convert GrabCut mask values to binary mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                mask[i, j] = 255  # Foreground
            elif mask[i, j] == 2:
                mask[i, j] = 0    # Probable background
            elif mask[i, j] == 3:
                mask[i, j] = 255  # Probable foreground
            # mask[i, j] == 0 remains 0 (definite background)
    
    return mask
