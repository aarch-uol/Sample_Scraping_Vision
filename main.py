import cv2
import numpy as np

# Import custom modules
import YOLO_boundingbox
import evaluations  # Import the evaluations module
import saving  # Import the saving module
import live_camera  # Import the live camera module
import all_image_processor  # Import the all image processor module
import spatula_grouping  # Import the spatula grouping module
import video_pipeline  # Import the video pipeline module




def midpoint_depth(depth_image, threshold_ratio=0.3):
    """
    Calculate threshold for depth image based on closest and furthest points.
    
    Args:
        depth_image (np.ndarray): Input depth image
        threshold_ratio (float): Ratio for threshold calculation (0.0-1.0)
    
    Returns:
        float: Calculated threshold value, None if no valid depth values found
    """
    depth_image_with_thresh = np.copy(depth_image)
    
    # Find closest and furthest depth points
    closest = np.inf
    furthest = -np.inf
    
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] > 0:
                if depth_image[i, j] < closest:
                    closest = depth_image[i, j]
                if depth_image[i, j] > furthest:
                    furthest = depth_image[i, j]
    
    # Check if valid depth values were found
    if closest == np.inf or furthest == -np.inf:
        print("No valid depth values found in the image.")
        return None
    
    # Calculate threshold and apply
    threshold = closest + threshold_ratio * (furthest - closest)
    #threshold = closest + 70
    depth_image_with_thresh[depth_image > threshold] = 0
    
    return threshold


def find_contours(image_data):
    """
    Find and draw contour areas and mark contour midpoints.
    
    Args:
        image_data (ImageData): ImageData object containing contourmap and color_image
        
    Returns:
        ImageData: Updated ImageData object with individual_contour_images, contoured_image, and midpoint_coords
    """
    # Validate input
    assert len(image_data.contourmap.shape) == 2, "Input must be a single-channel binary image"
    
    # Initialize variables
    output = np.copy(image_data.color_image)
    image_data.contoured_image = np.zeros_like(image_data.color_image)
    #image_data.contoured_image = image_data.color_image.copy()
    image_data.individual_contour_images = []
    image_data.midpoint_coords = []
    
    # Find contours
    contours, hierarchy = cv2.findContours(image_data.contourmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for i in range(len(contours)):
        # Skip inner contours
        if hierarchy[0][i][3] != -1:
            continue

        
        # Reset for each contour
        output = image_data.color_image.copy()
        
        # Draw filled contour in green
        cv2.drawContours(output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(image_data.contoured_image, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        #find the midpoint of the contour
        M = cv2.moments(contours[i])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            image_data.midpoint_coords.append((cX, cY))
        else:
            cX, cY = 0, 0
        image_data.midpoint_coords.append((cX, cY))

        image_data.individual_contour_images.append(output)

    return image_data


def process_depth_data(image_data):
    """
    Extract, denoise and threshold depth data.
    
    Args:
        image_data (ImageData): ImageData object containing grabcut_mask and depth_image
        
    Returns:
        ImageData: Updated ImageData object with extracted_depth, depth_denoised, and thresh
    """
    # Extract depth values using the mask
    image_data.extracted_depth = cv2.bitwise_and(image_data.depth_image, image_data.depth_image, mask=image_data.grabcut_mask)


    
    # Denoise the depth image
    image_data.depth_denoised = denoise(image_data.extracted_depth)
    #image_data.depth_denoised = image_data.extracted_depth.copy()



    # Apply threshold
    image_data.thresh = np.copy(image_data.depth_denoised)
    threshold = midpoint_depth(image_data.thresh)
    image_data.thresh[image_data.depth_denoised > threshold] = 0
    
    return image_data


def create_binary_map(image_data):
    """
    Convert thresholded depth to binary map.
    
    Args:
        image_data (ImageData): ImageData object containing thresh
        
    Returns:
        ImageData: Updated ImageData object with contourmap
    """
    image_data.contourmap = image_data.thresh.copy()
    image_data.contourmap[image_data.contourmap > 0] = 255
    image_data.contourmap[image_data.contourmap <= 0] = 0
    image_data.contourmap = image_data.contourmap.astype(np.uint8)
    
    return image_data

def detect_and_segment_vial(image_data):
    """
    Detect vial using YOLO and segment it using GrabCut.
    
    Args:
        image_data (ImageData): ImageData object containing color image
        
    Returns:
        ImageData: Updated ImageData object with detection results
    """
    # Get bounding box from YOLO
    rect = YOLO_boundingbox.find_bounding_box(image_data.color_image)
    if rect is None:
        print("No bounding box found. Skipping frame...")
        image_data.grabcut_mask = None
        image_data.bounding_image = None
        image_data.yolo_rect = None
        return image_data
    
    # Store original rect for evaluation cropping (x, y, width, height)
    image_data.yolo_rect = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
    
    # Draw bounding box for visualization
    image_data.bounding_image = image_data.color_image.copy()
    cv2.rectangle(image_data.bounding_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 3)
    
    # Convert rect format for GrabCut (x, y, width, height)
    rect = list(rect)
    rect[2] = rect[2] - rect[0]  # width = x2 - x1
    rect[3] = rect[3] - rect[1]  # height = y2 - y1
    rect = tuple(rect)
    
    # Apply GrabCut algorithm
    image_data.grabcut_mask = apply_grabcut(image_data.color_image, rect, iter_count=5)
    
    return image_data

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

def denoise(depth_image):
    """
    Denoises the depth image using morphological operations and removes depth outliers.
    
    Parameters:
    - depth_image: The input depth image to be denoised.
    
    Returns:
    - Denoised depth image with outliers removed.
    """
    # Apply morphological operations first
    kernel = np.ones((3,3),np.uint8)
    # Apply morphological opening to remove noise
    #denoised_image = cv2.morphologyEx(depth_image,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Apply morphological closing to fill small holes
    #denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    denoised_image = depth_image.copy()

    # Remove depth outliers
    # Get non-zero depth values for statistics
    non_zero_depths = denoised_image[denoised_image > 0]
    
    if len(non_zero_depths) > 0:
        # Calculate mean and standard deviation of non-zero depths
        mean_depth = np.mean(non_zero_depths)
        std_depth = np.std(non_zero_depths)

        # Define outlier threshold (pixels beyond 1.5 standard deviations)
        outlier_threshold = 1.5
        lower_bound = mean_depth - outlier_threshold * std_depth
        upper_bound = mean_depth + outlier_threshold * std_depth
        
        # Remove outliers by setting them to zero
        outlier_mask = (denoised_image < lower_bound) | (denoised_image > upper_bound)
        denoised_image[outlier_mask] = 0
        
        print(f"Depth stats: mean={mean_depth:.1f}, std={std_depth:.1f}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
    
    return denoised_image

def extract_3d_coordinates(image_data):
    """
    Extract 3D coordinates from white pixels in spatula_results_smoothed.
    
    Args:
        image_data (ImageData): ImageData object containing spatula_results_smoothed and depth_image
        
    Returns:
        ImageData: Updated ImageData object with coordinates array
    """
    # Find white pixels in spatula_results_smoothed
    # For grayscale image, white pixels have value 255
    # For color image, white pixels have all channels at 255
    if len(image_data.spatula_results_smoothed.shape) == 3:
        # Color image - check if all channels are 255
        white_mask = np.all(image_data.spatula_results_smoothed == 255, axis=2)
    else:
        # Grayscale image - check if pixel value is 255
        white_mask = image_data.spatula_results_smoothed == 255
    
    # Get coordinates of white pixels
    white_coords = np.where(white_mask)
    y_coords = white_coords[0]  # Row indices (y positions)
    x_coords = white_coords[1]  # Column indices (x positions)
    
    # Extract corresponding depth values
    depth_values = image_data.depth_image[y_coords, x_coords]
    
    # Create 3D coordinate array: [x, y, depth]
    num_points = len(x_coords)
    coordinates_3d = np.zeros((num_points, 3))
    coordinates_3d[:, 0] = x_coords  # X positions
    coordinates_3d[:, 1] = y_coords  # Y positions  
    coordinates_3d[:, 2] = depth_values  # Depth values
    
    # Store in image_data
    image_data.coordinates = coordinates_3d
    return image_data


def save_coordinates_to_csv(image_data, filename=None):
    """
    Save 3D coordinates to a CSV file.
    
    Args:
        image_data (ImageData): ImageData object containing coordinates array
        filename (str): Output CSV filename. If None, uses image name
    """
    import csv
    import os
    
    # Generate filename if not provided
    if filename is None:
        base_name = os.path.splitext(image_data.image_name)[0]
        filename = f"{base_name}_3d_coordinates.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "coordinate_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path for the CSV file
    csv_path = os.path.join(output_dir, filename)
    
    # Write coordinates to CSV
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['X', 'Y', 'Depth'])
            
            # Write coordinate data
            for coord in image_data.cropped_coords:
                writer.writerow([int(coord[0]), int(coord[1]), int(coord[2])])


    except Exception as e:
        print(f"Error saving coordinates to CSV: {e}")


def crop_and_mask_results(image_data, crop_percent, offset=0):
    """
    Crop the YOLO rectangle horizontally and apply as mask to coordinates and spatula results.
    
    Args:
        image_data (ImageData): ImageData object containing yolo_rect, coordinates, and spatula_results_smoothed
        crop_percent (float): Percentage to crop from each side horizontally (0.0 to 0.5)
                             e.g., 0.1 crops 10% from left and 10% from right (20% total)
        offset (int): Horizontal offset in pixels. Positive values shift right, negative values shift left (default: 0)
    
    Returns:
        ImageData: Updated ImageData object with cropped_coords and spatula_results_smoothed_cropped
    """
    # Check if required data exists
    if image_data.yolo_rect is None:
        print("No YOLO rectangle found. Cannot crop.")
        return image_data
    
    if image_data.coordinates is None:
        print("No coordinates found. Cannot crop coordinates.")
        return image_data
    
    if image_data.spatula_results_smoothed is None:
        print("No spatula results found. Cannot crop spatula results.")
        return image_data
    
    # Get original YOLO rectangle (x, y, width, height)
    x, y, width, height = image_data.yolo_rect
    
    # Calculate crop amount (pixels to remove from each side)
    crop_pixels = int(width * crop_percent)
    
    # Calculate vertical crop from top (20% of height)
    vertical_crop_pixels = int(height * 0.25)
    
    # Create cropped rectangle coordinates with offset
    cropped_x_start = x + crop_pixels + offset
    cropped_x_end = x + width - crop_pixels + offset
    cropped_y_start = y + vertical_crop_pixels  # Crop 20% from top
    cropped_y_end = y + height
    
    # Create mask for coordinates
    coord_mask = (
        (image_data.coordinates[:, 0] >= cropped_x_start) &  # X >= left bound
        (image_data.coordinates[:, 0] <= cropped_x_end) &    # X <= right bound
        (image_data.coordinates[:, 1] >= cropped_y_start) &  # Y >= top bound
        (image_data.coordinates[:, 1] <= cropped_y_end)      # Y <= bottom bound
    )
    
    # Apply mask to coordinates
    image_data.cropped_coords = image_data.coordinates[coord_mask]
    
    # Create spatial mask for spatula results image
    image_height, image_width = image_data.spatula_results_smoothed.shape[:2]
    
    # Ensure bounds are within image dimensions
    cropped_x_start = max(0, min(cropped_x_start, image_width))
    cropped_x_end = max(0, min(cropped_x_end, image_width))
    cropped_y_start = max(0, min(cropped_y_start, image_height))
    cropped_y_end = max(0, min(cropped_y_end, image_height))
    
    # Create a copy of spatula results and apply mask
    image_data.spatula_results_smoothed_cropped = image_data.spatula_results_smoothed.copy()
    
    # Set pixels outside the cropped rectangle to black
    # Set everything to black first
    image_data.spatula_results_smoothed_cropped[:, :] = 0
    
    # Copy only the cropped region from original
    image_data.spatula_results_smoothed_cropped[cropped_y_start:cropped_y_end, 
                                                cropped_x_start:cropped_x_end] = \
        image_data.spatula_results_smoothed[cropped_y_start:cropped_y_end, 
                                           cropped_x_start:cropped_x_end]
    
    # Draw red rectangle showing the crop mask boundaries
    
    cv2.rectangle(image_data.spatula_results_smoothed_cropped, 
                  (cropped_x_start, cropped_y_start), 
                  (cropped_x_end, cropped_y_end), 
                  (0, 0, 255),  # Red color in BGR format
                  thickness=2)
    
    # Store cropped rectangle coordinates
    image_data.cropped_rect = (cropped_x_start, cropped_y_start, cropped_x_end - cropped_x_start, cropped_y_end - cropped_y_start)
    
    return image_data

def process_single_image(image_data):
    """Process a single image through the entire pipeline."""
    
    image_data = detect_and_segment_vial(image_data)
    if image_data.grabcut_mask is None or image_data.bounding_image is None:
        print(f"Vial detection failed for {image_data.image_name}. Skipping...")
        return False, image_data
    
    image_data = process_depth_data(image_data)
    image_data = create_binary_map(image_data)
    image_data = find_contours(image_data)
    image_data = saving.create_visualizations(image_data)
    image_data = spatula_grouping.process_crystal_clustering(image_data)
    image_data.spatula_results_smoothed = denoise(image_data.spatula_results)

    # Create a variable to hold the 3d coordinates of the detected crystals
    image_data = extract_3d_coordinates(image_data)
    
    # Crop the results by x% on each side horizontally
    crop_percent = 0.3
    image_data = crop_and_mask_results(image_data, crop_percent, offset=-10)
    # Find crystal midpoints using K-means clustering
    image_data = spatula_grouping.find_crystal_midpoints(image_data)
    '''
    cv2.imshow('spatula_results_smoothed', image_data.spatula_results_smoothed)
    cv2.imshow('contoured_image', image_data.contoured_image)
    cv2.imshow('spatula_results_smoothed_cropped', image_data.spatula_results_smoothed_cropped)
    cv2.imshow('original_image', image_data.color_image)
    cv2.waitKey(0)  # Wait for a key press to continue
    cv2.destroyAllWindows()
    '''
    # Save coordinates to CSV (now using cropped coordinates)
    #save_coordinates_to_csv(image_data)

    
    

    # Save results (optional)
    #output_path = saving.save_combined_results(image_data, results_path='results')
    return True, image_data

def main():
    """Main pipeline orchestrator - choose between live camera or static images"""
    print("=== Crystal Detection Pipeline ===")
    print("Choose processing mode:")
    print("1. Live camera processing")
    print("2. Process all images sequentially")
    print("3. Process all images with manual label evaluation")
    dataset_path = 'Test_dataset4'  # Path to the dataset containing images
    
    while True:
        choice = input("Enter your choice (1, 2, 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, 3")
    
    if choice == '1':
        live_camera.run_live_pipeline(process_single_image)

    elif choice == '2':
        # Process all images sequentially
        all_image_processor.process_all_images(dataset_path, process_single_image)

    elif choice == '3':
        # Process all images with manual label evaluation
        video_pipeline.process_all_bag_videos(
            bag_folder_path='Bag_files',
            process_single_image_func=process_single_image,
            frame_skip=1,
            max_frames_per_bag=None
        )
    else:
        # Process all images with manual label evaluation
        evaluations.process_all_images_with_evaluation(dataset_path, process_single_image)
    


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
