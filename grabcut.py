import cv2
import numpy as np

def apply_grabcut(img, rect, iter_count=1):
    mask = np.zeros(img.shape[:2], np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bg, fg, iter_count, cv2.GC_INIT_WITH_RECT)
    
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                mask[i, j] = 255 # Foreground
            elif mask[i, j] == 2:
                mask[i, j] = 0 # Probable background
            elif mask[i, j] == 3:
                mask[i, j] = 255 # Probable foreground
    return mask
