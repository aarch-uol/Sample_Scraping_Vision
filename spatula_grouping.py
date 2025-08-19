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
        
        if abs(hue - green_hue) < 20 and saturation > 60:  # High saturation threshold
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

def find_crystal_midpoints(image_data):
    """
    Find crystal midpoints using K-means clustering on white pixels in spatula_results_smoothed_cropped.
    
    Args:
        image_data (ImageData): ImageData object containing spatula_results_smoothed_cropped
    
    Returns:
        ImageData: Updated ImageData object with final_midpoints and final_midpoints_image
    """
    
    if image_data.spatula_results_smoothed_cropped is None:
        print("No spatula_results_smoothed_cropped found. Cannot find midpoints.")
        return image_data
    
    # Step 1: Find white pixels in spatula_results_smoothed_cropped
    
    # Color image - check if all channels are 255 (white)
    white_mask = np.all(image_data.spatula_results_smoothed_cropped == 255, axis=2)
    
    # Get coordinates of white pixels
    white_coords = np.where(white_mask)
    
    if len(white_coords[0]) == 0:
        print("No white pixels found for clustering.")
        image_data.final_midpoints = np.array([])
        image_data.final_midpoints_image = image_data.spatula_results_smoothed_cropped.copy()
        return image_data
    
    # Step 2: Determine optimal k value based on white pixel distribution
    white_pixels = np.column_stack((white_coords[1], white_coords[0]))  # (x, y) format
    num_white_pixels = len(white_pixels)

    # Heuristic for k: estimate number of clusters based on cropped box area and pixel density
    
        # Use the actual cropped rectangle area
    x, y, width, height = image_data.cropped_rect
    image_area = width * height
    
    image_data.percentage_coverage = (num_white_pixels / (width * height)) * 100  # Percentage coverage
    
    # Estimate k based on density and typical crystal sizes
    k = max(1, num_white_pixels // 500)  # Default to 1 cluster or fewer based on pixel count

    # Ensure k doesn't exceed number of white pixels
    k = min(k, num_white_pixels)
    
    # Step 3: Apply K-means clustering to white pixel coordinates
    if k == 1:
        # Only one cluster - calculate centroid directly
        centroid_x = np.mean(white_pixels[:, 0])
        centroid_y = np.mean(white_pixels[:, 1])
        cluster_centers = np.array([[centroid_x, centroid_y]])
    else:
        # Convert to float32 for k-means
        white_pixels_float = np.float32(white_pixels)
        
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        try:
            _, labels, cluster_centers = cv2.kmeans(white_pixels_float, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        except cv2.error as e:
            print(f"K-means failed: {e}. Using single centroid.")
            centroid_x = np.mean(white_pixels[:, 0])
            centroid_y = np.mean(white_pixels[:, 1])
            cluster_centers = np.array([[centroid_x, centroid_y]])
    
    
    # Step 5: Create final_midpoints_image and plot red dots
    image_data.final_midpoints_image = image_data.spatula_results_smoothed_cropped.copy()

    #Find out how many pixels correspond to each midpoint
    midpoint_pixel_counts = np.zeros(len(cluster_centers), dtype=int)
    #loop through each pixel in the white mask
    for y, x in zip(white_coords[0], white_coords[1]):
        # Calculate distances to each midpoint
        distances = np.linalg.norm(cluster_centers - np.array([x, y]), axis=1)
        closest_midpoint = np.argmin(distances)
        midpoint_pixel_counts[closest_midpoint] += 1

    # work out the midpoint_pixel_counts as a percentage of total area given by the cropped rectangle
    total_area = image_data.cropped_rect[2] * image_data.cropped_rect[3]  # width * height
    midpoint_pixel_counts = (midpoint_pixel_counts / total_area) * 100  # Convert to percentage

    # Store midpoints and their pixel counts as tuples
    image_data.final_midpoints = [(int(center[0]), int(center[1]), count) for center, count in zip(cluster_centers, midpoint_pixel_counts)]

    # Ensure image is in color format for drawing red dots
    if len(image_data.final_midpoints_image.shape) == 2:
        # Convert grayscale to color
        image_data.final_midpoints_image = cv2.cvtColor(image_data.final_midpoints_image, cv2.COLOR_GRAY2BGR)
    index = 0
    # Draw red dots at each midpoint
    for midpoint in cluster_centers:
        x, y = int(midpoint[0]), int(midpoint[1])
        index += 1
        # Ensure coordinates are within image bounds
        if 0 <= x < image_data.final_midpoints_image.shape[1] and 0 <= y < image_data.final_midpoints_image.shape[0]:
            # Draw red filled circle (BGR format: red is (0, 0, 255))
            cv2.circle(image_data.final_midpoints_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            # Put the index number at the center of the circle
            cv2.putText(image_data.final_midpoints_image, str(index), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

    #print the midpoints, and the percentage coverage
    print(f"Percentage coverage of detected crystals: {image_data.percentage_coverage:.2f}%")
    print(f"Found {len(image_data.final_midpoints)} crystal midpoints")
    print("Crystal midpoints (x, y):")
    for midpoint in image_data.final_midpoints:
        print(f"    ({midpoint[0]}, {midpoint[1]})")
    # Display the final midpoints image
    cv2.waitKey(500)  # Update display
    
    return image_data


