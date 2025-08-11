import cv2
import numpy as np

# Import custom modules
import YOLO_boundingbox
import spatula_extract
import evaluations  # Import the evaluations module
import saving  # Import the saving module
import live_camera  # Import the live camera module
import all_image_processor  # Import the all image processor module





def midpoint_depth(depth_image, threshold_ratio=0.5):
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
    depth_image_with_thresh[depth_image > threshold] = 0
    print(f"Depth range: {closest} - {furthest}")
    
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
    image_data.contoured_image = image_data.color_image.copy()
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
    
    # Apply threshold
    image_data.thresh = np.copy(image_data.depth_denoised)
    threshold = midpoint_depth(image_data.thresh)
    print(f"Threshold value: {threshold}")
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
    print(f'Rectangle coordinates: {rect}')
    
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


def process_single_image(image_data):
    """Process a single image through the entire pipeline."""
    print(f"\n=== Processing {image_data.image_name} ===")
    
    image_data = detect_and_segment_vial(image_data)
    if image_data.grabcut_mask is None or image_data.bounding_image is None:
        print(f"Vial detection failed for {image_data.image_name}. Skipping...")
        return False, image_data
    
    image_data = process_depth_data(image_data)
    image_data = create_binary_map(image_data)
    image_data = find_contours(image_data)
    image_data = saving.create_visualizations(image_data)
    #image_data.spatula_results = spatula_extract.process_spatula_separation(image_data.contoured_image, image_data.color_image)


    # Save results (optional)
    #output_path = saving.save_combined_results(image_data, results_path='results')

    print(f"Successfully processed {image_data.image_name}")
    return True, image_data

def main():
    """Main pipeline orchestrator - choose between live camera or static images"""
    print("=== Crystal Detection Pipeline ===")
    print("Choose processing mode:")
    print("1. Live camera processing")
    print("2. Process all images sequentially")
    print("3. Process all images with manual label evaluation")
    dataset_path = 'Test_dataset3'  # Path to the dataset containing images
    
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

    else:
        # Process all images with manual label evaluation
        evaluations.process_all_images_with_evaluation(dataset_path, process_single_image)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
