import cv2
import numpy as np

def distance_transform(thresh):
    #Now need to convert the thresh image to a binary image
    midcluster = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX)
    midcluster = np.uint8(midcluster)
    _, midcluster = cv2.threshold(midcluster, 0, 255, cv2.THRESH_BINARY)
    
    #apply a distance transform to the thresholded image
    midcluster = cv2.distanceTransform(midcluster, cv2.DIST_L2, 5)
    #normalize the distance transformed image
    #midcluster = cv2.normalize(midcluster, None, 0, 255, cv2.NORM_MINMAX)
    midcluster = np.uint8(midcluster)
    return midcluster

    
def erode_until_area(image, kernel, min_area):
    """
    Keeps eroding the image until the area of the biggest segment in foreground is equal to min_area.
    """
    while True:
        eroded = cv2.erode(image, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            break
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        if max_area <= min_area:
            break
        image = eroded
    return image
    