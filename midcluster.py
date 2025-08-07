import cv2
import numpy as np


def distance_transform(thresh):
    """
    Apply distance transform to a thresholded image.
    
    Args:
        thresh (np.ndarray): Input thresholded image
        
    Returns:
        np.ndarray: Distance transformed image as uint8
    """
    # Convert the thresh image to a binary image
    midcluster = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX)
    midcluster = np.uint8(midcluster)
    _, midcluster = cv2.threshold(midcluster, 0, 255, cv2.THRESH_BINARY)
    
    # Apply a distance transform to the thresholded image
    midcluster = cv2.distanceTransform(midcluster, cv2.DIST_L2, 5)
    
    # Convert to uint8 format
    # Note: normalize step is commented out in original
    # midcluster = cv2.normalize(midcluster, None, 0, 255, cv2.NORM_MINMAX)
    midcluster = np.uint8(midcluster)
    
    return midcluster

    
def erode_until_area(image, kernel, min_area):
    """
    Keeps eroding the image until the area of the biggest segment in foreground is equal to min_area.
    
    Args:
        image (np.ndarray): Input binary image to erode
        kernel (np.ndarray): Morphological kernel for erosion
        min_area (int): Minimum area threshold to stop erosion
        
    Returns:
        np.ndarray: Eroded image where largest contour area <= min_area
    """
    while True:
        # Apply erosion
        eroded = cv2.erode(image, kernel, iterations=1)
        
        # Find contours in eroded image
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Break if no contours found
        if not contours:
            break
            
        # Calculate areas of all contours
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        
        # Stop erosion if max area is below threshold
        if max_area <= min_area:
            break
            
        # Continue with eroded image
        image = eroded
        
    return image
