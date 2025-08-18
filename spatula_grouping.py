import cv2
import numpy as np

def process_crystal_clustering(image_data):
    """
    Process crystal clustering using K-means on detected crystal regions.
    
    Args:
        image_data (ImageData): ImageData object containing contoured_image and color_image
    
    Returns:
        ImageData: Updated ImageData object with clustering results
    """
    
    if image_data.contoured_image is None:
        print("No contoured image found. Cannot proceed.")
        return image_data
    
    if image_data.color_image is None:
        print("No color image found. Cannot proceed.")
        return image_data
    
    # Step 1: Apply contoured_image as mask to color_image
    # Create mask from contoured_image (look for green crystal regions)
    green_lower = np.array([0, 200, 0])  # Lower bound for green (BGR)
    green_upper = np.array([100, 255, 100])  # Upper bound for green (BGR)
    crystal_mask = cv2.inRange(image_data.contoured_image, green_lower, green_upper)
    
    # Apply mask to extract only crystal regions from color image
    detected_color_crystals = image_data.color_image.copy()
    detected_color_crystals[crystal_mask == 0] = [0, 0, 0]  # Set non-crystal regions to black
    
    
    
    # Step 2: Get crystal pixel coordinates and RGB values
    crystal_coords = np.where(crystal_mask > 0)
    
    if len(crystal_coords[0]) == 0:
        print("No crystal pixels found for clustering.")
        return image_data
    
    # Extract BGR values and convert to RGB for K-means
    crystal_pixels_bgr = image_data.color_image[crystal_coords]
    crystal_pixels_rgb = cv2.cvtColor(crystal_pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    crystal_pixels_rgb = np.float32(crystal_pixels_rgb)

    
    # Step 3: Apply K-means clustering to find 3 predominant colors
    k = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(crystal_pixels_rgb, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8 RGB
    centers = np.uint8(centers)
    
    
    # Step 4: Assign each pixel to closest cluster using Euclidean distance
    rgb_image = cv2.cvtColor(image_data.color_image, cv2.COLOR_BGR2RGB)
    
    # Create new image to draw pixels with their assigned cluster colors
    clustered_image = np.zeros_like(image_data.color_image)
    
    # Process each crystal pixel
    for y, x in zip(crystal_coords[0], crystal_coords[1]):
        pixel_rgb = rgb_image[y, x]
        
        # Calculate Euclidean distance to each cluster center
        distances = []
        for center in centers:
            r_diff = int(center[0]) - int(pixel_rgb[0])
            g_diff = int(center[1]) - int(pixel_rgb[1])
            b_diff = int(center[2]) - int(pixel_rgb[2])
            distance = np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
            distances.append(distance)
        
        # Find closest cluster
        closest_cluster = np.argmin(distances)
        
        # Assign pixel the color of its closest cluster center (convert RGB to BGR for OpenCV)
        cluster_color_bgr = [centers[closest_cluster][2], centers[closest_cluster][1], centers[closest_cluster][0]]
        clustered_image[y, x] = cluster_color_bgr

    #convert the RGB centers into HSV
    hsv_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    #find the highest saturated cluster
    
    # Fix the HSV comparison - OpenCV uses H: 0-179, S: 0-255, V: 0-255
    green_hue = 60  # Green hue in OpenCV HSV (0-179 range)
    green_clusters = []

    for center in hsv_centers:
        # Ensure we're working with the correct data types and ranges
        hue = int(center[0]) if center[0] < 180 else 179  # Clamp hue to valid range
        saturation = int(center[1])
        
        if abs(hue - green_hue) < 20 and saturation > 100:  # High saturation threshold
            green_clusters.append(center)
            #print(f"Green cluster found: H={hue}, S={saturation}, V={int(center[2])}")

    #set all pixels in the green clusters to black
    for green_cluster in green_clusters:
        # Convert HSV back to RGB for comparison
        green_rgb = cv2.cvtColor(green_cluster.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        green_bgr = [green_rgb[2], green_rgb[1], green_rgb[0]]  # Convert RGB to BGR
        
        for y, x in zip(crystal_coords[0], crystal_coords[1]):
            if np.allclose(clustered_image[y, x], green_bgr, atol=1):  # Allow small tolerance
                clustered_image[y, x] = [0, 0, 0]

    # create a new mask image based on the clustered image, where all pixels that are not black are set to white
    mask_image = np.zeros_like(clustered_image)
    non_black_mask = np.any(clustered_image != [0, 0, 0], axis=2)
    mask_image[non_black_mask] = [255, 255, 255]
    # Update image_data with the clustered image and mask
    image_data.spatula_results = mask_image
    return image_data
