import cv2
import numpy as np

def extract_depth(mask, depth_image):
    """
    Extracts the depth values from the depth image using the provided mask.
    
    Parameters:
    - mask: A binary mask where the foreground is marked with 255 and background with 0.
    - depth_image: The depth image from which to extract values.
    
    Returns:
    - A new image containing only the depth values where the mask is applied.
    """
    # Ensure the mask is a binary image
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Apply the mask to the depth image
    extracted_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
    return extracted_depth