import cv2
import numpy as np

def denoise(depth_image):
    """
    Denoises the depth image using a Gaussian filter.
    
    Parameters:
    - depth_image: The input depth image to be denoised.
    
    Returns:
    - Denoised depth image.
    """
    #gaussian filter to smooth the depth image
    #depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
    kernel = np.ones((3,3),np.uint8)
    # Apply morphological opening to remove noise
    denoised_image = cv2.morphologyEx(depth_image,cv2.MORPH_OPEN,kernel, iterations = 3)
    # Apply morphological closing to fill small holes
    denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    
    
    return denoised_image