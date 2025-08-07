import cv2
import numpy as np
import pyrealsense2 as rs
import os
from datetime import datetime

# Import custom modules
import grabcut
import depth_extract
import denoise
import getdata
import midcluster
import YOLO_boundingbox


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


def find_contours(binary_img, original_image):
    """
    Draw contour areas and mark contour midpoints without small scanning boxes.
    
    Args:
        binary_img (np.ndarray): Binary image (white on black)
        original_image (np.ndarray): Original image to draw boxes on
        box_w (int): Width of scanning box (unused, kept for compatibility)
        box_h (int): Height of scanning box (unused, kept for compatibility)
        threshold_perc (int): Percentage threshold (unused, kept for compatibility)
        
    Returns:
        tuple: (individual_contour_images, final_output, midpoint_coords)
    """
    # Validate input
    assert len(binary_img.shape) == 2, "Input must be a single-channel binary image"
    
    # Initialize variables
    output = np.copy(original_image)
    final_output = original_image.copy()
    output_images = []
    midpoint_coords = []
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for i in range(len(contours)):
        # Skip inner contours
        if hierarchy[0][i][3] != -1:
            continue

        
        # Reset for each contour
        output = original_image.copy()
        
        # Draw filled contour in green
        cv2.drawContours(output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(final_output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        #find the midpoint of the contour
        M = cv2.moments(contours[i])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            midpoint_coords.append((cX, cY))
        else:
            cX, cY = 0, 0
        midpoint_coords.append((cX, cY))

        output_images.append(output)
    
    return output_images, final_output, midpoint_coords


def process_depth_data(mask, depth_image):
    """
    Extract, denoise and threshold depth data.
    
    Args:
        mask (np.ndarray): Binary mask from segmentation
        depth_image (np.ndarray): Original depth image
        
    Returns:
        tuple: (extracted_depth, depth_denoised, thresh)
    """
    # Extract depth values using the mask
    extracted_depth = depth_extract.extract_depth(mask, depth_image)
    
    # Denoise the depth image
    depth_denoised = denoise.denoise(extracted_depth)
    
    # Apply threshold
    thresh = np.copy(depth_denoised)
    threshold = midpoint_depth(thresh)
    print(f"Threshold value: {threshold}")
    thresh[depth_denoised > threshold] = 0
    
    return extracted_depth, depth_denoised, thresh


def create_binary_map(thresh):
    """
    Convert thresholded depth to binary map.
    
    Args:
        thresh (np.ndarray): Thresholded depth image
        
    Returns:
        np.ndarray: Binary image (0 or 255)
    """
    contourmap = thresh.copy()
    contourmap[contourmap > 0] = 255
    contourmap[contourmap <= 0] = 0
    contourmap = contourmap.astype(np.uint8)
    
    return contourmap


def analyze_crystals(contourmap, color_image):
    """
    Detect and analyze crystal formations.
    
    Args:
        contourmap (np.ndarray): Binary map of crystal areas
        color_image (np.ndarray): Original color image
        
    Returns:
        tuple: (individual_contour_images, contoured_image, midpoint_coords)
    """
    individual_contour_images, contoured_image, midpoint_coords = find_contours(contourmap, color_image)
    #print(f"Crystal midpoint coordinates: {midpoint_coords}")

    return individual_contour_images, contoured_image, midpoint_coords


def detect_and_segment_vial(color_image):
    """
    Detect vial using YOLO and segment it using GrabCut.
    
    Args:
        color_image (np.ndarray): Input color image
        
    Returns:
        tuple: (mask, bounding_image, yolo_rect) or (None, None, None) if detection fails
    """
    # Get bounding box from YOLO
    rect = YOLO_boundingbox.find_bounding_box(color_image)
    if rect is None:
        print("No bounding box found. Skipping frame...")
        return None, None, None
    
    # Store original rect for evaluation cropping (x, y, width, height)
    yolo_rect = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
    
    # Draw bounding box for visualization
    bounding_image = color_image.copy()
    cv2.rectangle(bounding_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 3)
    print(f'Rectangle coordinates: {rect}')
    
    # Convert rect format for GrabCut (x, y, width, height)
    rect = list(rect)
    rect[2] = rect[2] - rect[0]  # width = x2 - x1
    rect[3] = rect[3] - rect[1]  # height = y2 - y1
    rect = tuple(rect)
    
    # Apply GrabCut algorithm
    mask = grabcut.apply_grabcut(color_image, rect, iter_count=5)
    
    return mask, bounding_image, yolo_rect


def create_visualizations(extracted_depth, depth_denoised, thresh):
    """
    Apply colormaps to depth images for visualization.
    
    Args:
        extracted_depth (np.ndarray): Original extracted depth
        depth_denoised (np.ndarray): Denoised depth image
        thresh (np.ndarray): Thresholded depth image
        
    Returns:
        tuple: (extracted_depth_colored, depth_denoised_colored, thresh_colored)
    """
    # Apply JET colormap to each depth image
    extracted_depth_colored = cv2.normalize(extracted_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    extracted_depth_colored = cv2.applyColorMap(extracted_depth_colored, cv2.COLORMAP_JET)
    
    depth_denoised_colored = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_denoised_colored = cv2.applyColorMap(depth_denoised_colored, cv2.COLORMAP_JET)
    
    thresh_colored = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_colored = cv2.applyColorMap(thresh_colored, cv2.COLORMAP_JET);
    
    return extracted_depth_colored, depth_denoised_colored, thresh_colored


def display_live_results(color_image, final_boxed_image=None):
    """
    Display original and final result in separate windows for live processing.
    
    Args:
        color_image (np.ndarray): Original color image
        final_boxed_image (np.ndarray): Final result with crystal analysis (optional)
    """
    # Show original image
    cv2.imshow('Live Camera Feed', color_image)
    
    # Show final result if available, otherwise show original again
    if final_boxed_image is not None:
        cv2.imshow('Crystal Detection Result', final_boxed_image)
    else:
        cv2.imshow('Crystal Detection Result', color_image)


def process_live_frame(color_image, depth_image, frame_count):
    """
    Process a single live frame through the pipeline.
    
    Args:
        color_image (np.ndarray): Input color image
        depth_image (np.ndarray): Input depth image
        frame_count (int): Current frame number
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        print(f"\n=== Processing Frame {frame_count} ===")
        
        # Step 1: Detect and segment vial
        mask, bounding_image = detect_and_segment_vial(color_image)
        if mask is None or bounding_image is None:
            print(f"Vial detection failed for frame {frame_count}. Showing original image...")
            # Still display the camera feed even if detection fails
            display_live_results(color_image)
            return False
        
        # Step 2: Process depth data
        extracted_depth, depth_denoised, thresh = process_depth_data(mask, depth_image)
        
        # Step 3: Create binary map
        contourmap = create_binary_map(thresh)
        
        # Step 4: Analyze crystals
        individual_contour_images, contoured_image, midpoint_coords = analyze_crystals(contourmap, color_image)

        # Step 5: Display results (original and final)
        display_live_results(color_image, contoured_image)
        
        print(f"Successfully processed frame {frame_count}")
        return True
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        # Still display the camera feed even if processing fails
        display_live_results(color_image)
        return False


def run_live_pipeline():
    """Run the crystal detection pipeline on live camera data."""
    print("=== Starting Live Crystal Detection Pipeline ===")
    print("Press 'q' to quit")
    
    # Initialize camera
    pipeline, profile = getdata.initialize_camera()
    if pipeline is None:
        print("Failed to initialize camera")
        return
    
    frame_count = 0
    
    try:
        while True:
            # Get frames from camera
            color_image, depth_image = getdata.get_live_frame(pipeline)
            if color_image is None or depth_image is None:
                print("Failed to get frames")
                continue
            
            frame_count += 1
            # Process the live frame
            process_live_frame(color_image, depth_image, frame_count)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting live pipeline...")
                break
    
    finally:
        # Clean up
        getdata.cleanup_camera(pipeline)
        cv2.destroyAllWindows()
        print(f"\nLive pipeline ended. Processed {frame_count} frames.")


def save_combined_results(color_image, bounding_image, mask, extracted_depth_colored, 
                         depth_denoised_colored, thresh_colored, final_boxed_image, image_name):
    """
    Combine all results into a single image and save to results folder.
    
    Args:
        color_image (np.ndarray): Original color image
        bounding_image (np.ndarray): Image with YOLO bounding box
        mask (np.ndarray): GrabCut segmentation mask
        extracted_depth_colored (np.ndarray): Colored depth visualization
        depth_denoised_colored (np.ndarray): Colored denoised depth
        thresh_colored (np.ndarray): Colored thresholded depth
        final_boxed_image (np.ndarray): Final result with crystal analysis
        image_name (str): Name of the original image
        
    Returns:
        str: Path to saved combined image
    """
    # Create results folder
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Convert mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions for dynamic scaling
    h, w = color_image.shape[:2]
    
    # Calculate dynamic text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))
    text_y = max(30, int(h / 20))
    
    # Create labeled copies
    images_to_label = [
        (color_image.copy(), 'Original'),
        (bounding_image.copy(), 'YOLO Detection'),
        (mask_colored.copy(), 'GrabCut Mask'),
        (extracted_depth_colored.copy(), 'Extracted Depth'),
        (depth_denoised_colored.copy(), 'Denoised Depth'),
        (thresh_colored.copy(), 'Thresholded'),
        (final_boxed_image.copy(), 'Crystal Analysis')
    ]
    
    labeled_images = []
    for img, label in images_to_label:
        cv2.putText(img, label, (10, text_y), font, font_scale, color_white, thickness)
        labeled_images.append(img)
    
    # Combine images in 2x4 grid
    top_row = np.hstack(labeled_images[:4])
    bottom_row = np.hstack(labeled_images[4:] + [np.zeros_like(labeled_images[0])])  # Add empty space
    combined_image = np.vstack([top_row, bottom_row])
    
    # Save combined image
    base_name = os.path.splitext(image_name)[0]
    output_path = os.path.join(results_folder, f"{base_name}_results.jpg")
    cv2.imwrite(output_path, combined_image)
    
    print(f"Combined results saved to: {output_path}")
    
    return output_path


def save_combined_results_with_evaluation(results_folder, color_image, bounding_image, mask, extracted_depth_colored, 
                                        depth_denoised_colored, thresh_colored, final_boxed_image, 
                                        evaluation_image, image_name, correctly_guessed, missed_crystal, incorrect_guess, dataset_path):
    """
    Combine all results including evaluation comparison into a single image and save to eval_results folder.
    
    Args:
        color_image (np.ndarray): Original color image
        bounding_image (np.ndarray): Image with YOLO bounding box
        mask (np.ndarray): GrabCut segmentation mask
        extracted_depth_colored (np.ndarray): Colored depth visualization
        depth_denoised_colored (np.ndarray): Colored denoised depth
        thresh_colored (np.ndarray): Colored thresholded depth
        final_boxed_image (np.ndarray): Final result with crystal analysis
        evaluation_image (np.ndarray): Evaluation comparison image
        image_name (str): Name of the original image
        correctly_guessed (int): Number of correctly guessed pixels
        missed_crystal (int): Number of missed crystal pixels
        incorrect_guess (int): Number of incorrect guess pixels
        
    Returns:
        str: Path to saved combined image
    """
    # Create eval_results folder
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Convert mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Load and create manual label superimposed image
    manual_label_path = os.path.join(dataset_path, 'manual_label', f"{image_name}.jpg")
    manual_superimposed = color_image.copy()
    
    if os.path.exists(manual_label_path):
        manual_label = cv2.imread(manual_label_path)
        if manual_label is not None:
            # Ensure same dimensions
            if manual_label.shape != color_image.shape:
                manual_label = cv2.resize(manual_label, (color_image.shape[1], color_image.shape[0]))
            
            # Define green color threshold for manual labels
            green_lower = np.array([0, 200, 0])
            green_upper = np.array([100, 255, 100])
            
            # Create mask for green areas in manual label
            green_mask = cv2.inRange(manual_label, green_lower, green_upper)
            
            # Superimpose green areas from manual label onto original image
            manual_superimposed[green_mask > 0] = [0, 255, 0]  # Green color
    
    # Get image dimensions for dynamic scaling
    h, w = color_image.shape[:2]
    
    # Calculate dynamic text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))
    text_y = max(30, int(h / 20))
    
    # Create labeled copies - now with 9 images in 3x3 grid
    images_to_label = [
        (color_image.copy(), 'Original'),
        (bounding_image.copy(), 'YOLO Detection'),
        (mask_colored.copy(), 'GrabCut Mask'),
        (extracted_depth_colored.copy(), 'Extracted Depth'),
        (depth_denoised_colored.copy(), 'Denoised Depth'),
        (thresh_colored.copy(), 'Thresholded'),
        (final_boxed_image.copy(), 'Crystal Analysis'),
        (manual_superimposed.copy(), 'Manual Label Overlay'),
        (evaluation_image.copy(), 'Evaluation vs Manual')
    ]
    
    labeled_images = []
    for img, label in images_to_label:
        cv2.putText(img, label, (10, text_y), font, font_scale, color_white, thickness)
        labeled_images.append(img)
    
    # Add evaluation metrics to the evaluation image
    eval_text_y = text_y + 30
    cv2.putText(labeled_images[-1], f"TP: {correctly_guessed}", (10, eval_text_y), 
                font, font_scale * 0.7, (0, 255, 0), thickness)  # Green text for TP
    cv2.putText(labeled_images[-1], f"FN: {missed_crystal}", (10, eval_text_y + 25), 
                font, font_scale * 0.7, (255, 0, 0), thickness)  # Blue text for FN
    cv2.putText(labeled_images[-1], f"FP: {incorrect_guess}", (10, eval_text_y + 50), 
                font, font_scale * 0.7, (0, 0, 255), thickness)  # Red text for FP
    
    # Add legend to evaluation image
    legend_y = eval_text_y + 80
    cv2.putText(labeled_images[-1], "Green=TP, Blue=FN, Red=FP", (10, legend_y), 
                font, font_scale * 0.5, color_white, thickness)
    
    # Combine images in 3x3 grid
    top_row = np.hstack(labeled_images[:3])
    middle_row = np.hstack(labeled_images[3:6])
    bottom_row = np.hstack(labeled_images[6:9])
    combined_image = np.vstack([top_row, middle_row, bottom_row])
    
    # Save combined image
    base_name = os.path.splitext(image_name)[0]
    output_path = os.path.join(results_folder, f"{base_name}_evaluation_results.jpg")
    cv2.imwrite(output_path, combined_image)
    
    print(f"Combined evaluation results saved to: {output_path}")
    
    return output_path


def compare_with_manual_labels(final_boxed_image, image_name, mask, yolo_rect=None, crop_percent=1, dataset_path='Test_dataset2'):
    """
    Compare generated contours with manual labels and create evaluation metrics.
    
    Args:
        final_boxed_image (np.ndarray): Image with generated green contours
        image_name (str): Name of the image to find corresponding manual label
        mask (np.ndarray): GrabCut mask to define the vial area
        yolo_rect (tuple): YOLO bounding box coordinates (x, y, width, height)
        
    Returns:
        tuple: (correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn)
    """
    # Load manual label image
    manual_label_path = os.path.join(dataset_path, 'manual_label', f"{image_name}.jpg")

    if not os.path.exists(manual_label_path):
        print(f"Manual label not found: {manual_label_path}")
        return 0, 0, 0, final_boxed_image, 0, 0, 0, 0
    
    manual_label = cv2.imread(manual_label_path)
    if manual_label is None:
        print(f"Could not load manual label: {manual_label_path}")
        return 0, 0, 0, final_boxed_image, 0, 0, 0, 0
    
    print(f"Comparing with manual label: {manual_label_path}")
    
    # Ensure both images have the same dimensions
    if final_boxed_image.shape != manual_label.shape:
        manual_label = cv2.resize(manual_label, (final_boxed_image.shape[1], final_boxed_image.shape[0]))
    
    # Ensure mask has same dimensions
    if mask.shape[:2] != final_boxed_image.shape[:2]:
        mask = cv2.resize(mask, (final_boxed_image.shape[1], final_boxed_image.shape[0]))
    
    # Calculate crop region (middle crop_percent of YOLO bounding box width)
    crop_x_start = 0
    crop_x_end = final_boxed_image.shape[1]
    crop_y_start = 0
    crop_y_end = final_boxed_image.shape[0]
    
    if yolo_rect is not None:
        x, y, width, height = yolo_rect
        # Calculate middle crop_percent of width
        crop_width = int(width * crop_percent)
        crop_x_offset = int(width * (1 - crop_percent) / 2)  # Center the crop region horizontally

        # Calculate crop region with 25% removed from top
        crop_height_offset = int(height * 0.25)  # Remove 25% from top

        crop_x_start = max(0, x + crop_x_offset)
        crop_x_end = min(final_boxed_image.shape[1], x + crop_x_offset + crop_width)
        crop_y_start = max(0, y + crop_height_offset)  # Start 25% down from top to remove the threads.
        crop_y_end = min(final_boxed_image.shape[0], y + height)
        
        print(f"YOLO bounding box: {yolo_rect}")
        print(f"Evaluation crop region: x={crop_x_start}-{crop_x_end}, y={crop_y_start}-{crop_y_end}")
    else:
        print("No YOLO bounding box provided, evaluating entire image")
    
    # Initialize counters
    correctly_guessed = 0  # True Positives (TP)
    missed_crystal = 0     # False Negatives (FN)
    incorrect_guess = 0    # False Positives (FP)
    true_negatives = 0     # True Negatives (TN)
    
    # Create evaluation image (start with original image without contours)
    try:
        original_color_path = os.path.join(dataset_path, 'color_images', f"{image_name}.jpg")
        evaluation_image = cv2.imread(original_color_path)
        if evaluation_image is None:
            evaluation_image = final_boxed_image.copy()
    except:
        evaluation_image = final_boxed_image.copy()
    
    # Define color thresholds for green detection (BGR format)
    green_lower = np.array([0, 200, 0])    # Lower bound for green
    green_upper = np.array([100, 255, 100]) # Upper bound for green
    
    # Process each pixel within the cropped region and mask
    for y in range(crop_y_start, crop_y_end):
        for x in range(crop_x_start, crop_x_end):
            # Only process pixels within the vial mask
            if mask[y, x] > 0:  # Inside vial area
                # Get pixel values
                generated_pixel = final_boxed_image[y, x]
                manual_pixel = manual_label[y, x]
                
                # Check if pixels are green (using color range)
                generated_is_green = cv2.inRange(generated_pixel.reshape(1, 1, 3), green_lower, green_upper)[0, 0] > 0
                manual_is_green = cv2.inRange(manual_pixel.reshape(1, 1, 3), green_lower, green_upper)[0, 0] > 0
                
                if generated_is_green and manual_is_green:
                    # Both are green - True Positive (TP)
                    correctly_guessed += 1
                    evaluation_image[y, x] = [0, 255, 0]  # Green
                    
                elif manual_is_green and not generated_is_green:
                    # Manual is green but generated isn't - False Negative (FN)
                    missed_crystal += 1
                    evaluation_image[y, x] = [255, 0, 0]  # Blue
                    
                elif generated_is_green and not manual_is_green:
                    # Generated is green but manual isn't - False Positive (FP)
                    incorrect_guess += 1
                    evaluation_image[y, x] = [0, 0, 255]  # Red
                    
                else:
                    # Neither is green - True Negative (TN)
                    true_negatives += 1
                    # Keep original pixel color for true negatives
    
    # Draw crop region rectangle on evaluation image for visualization
    if yolo_rect is not None:
        cv2.rectangle(evaluation_image, (crop_x_start, crop_y_start), (crop_x_end, crop_y_end), (255, 255, 0), 2)
    
    # Calculate additional metrics
    tp = correctly_guessed
    fp = incorrect_guess
    fn = missed_crystal
    tn = true_negatives
    
    # Print results
    total_pixels = tp + fp + fn + tn
    if total_pixels > 0:
        accuracy = ((tp + tn) / total_pixels) * 100
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

        print(f"Evaluation Results for {image_name} (cropped to middle {crop_percent * 100:.0f}%):")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall (Sensitivity): {recall:.2f}%")
        print(f"  Specificity: {specificity:.2f}%")
    
    return correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn
    

def process_all_images_with_evaluation(dataset_path):
    """Process all images sequentially with manual label evaluation."""
    print("\n=== Processing All Images with Manual Label Evaluation (Middle Crop) ===")

    results_folder = 'eval_cropped_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    successful_count = 0
    failed_count = 0
    failed_images = []
    
    # Overall statistics
    total_correctly_guessed = 0
    total_missed_crystal = 0
    total_incorrect_guess = 0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    try:
        for color_image, depth_image, image_name in getdata.get_all_images(dataset_path):
            try:
                # Process image - now returns yolo_rect
                mask, bounding_image, yolo_rect = detect_and_segment_vial(color_image)
                if mask is None or bounding_image is None:
                    print(f"Vial detection failed for {image_name}. Skipping...")
                    failed_count += 1
                    failed_images.append(f"{image_name} - Vial detection failed")
                    continue
                
                extracted_depth, depth_denoised, thresh = process_depth_data(mask, depth_image)
                contourmap = create_binary_map(thresh)
                individual_contour_images, contoured_image, midpoint_coords = analyze_crystals(contourmap, color_image)

                # Compare with manual labels (now includes yolo_rect parameter)
                correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn = compare_with_manual_labels(
                    contoured_image, image_name, mask, yolo_rect, crop_percent=0.4, dataset_path=dataset_path
                )
                
                # Add to totals
                total_correctly_guessed += correctly_guessed
                total_missed_crystal += missed_crystal
                total_incorrect_guess += incorrect_guess
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                
                # Save results
                extracted_depth_colored, depth_denoised_colored, thresh_colored = create_visualizations(
                    extracted_depth, depth_denoised, thresh
                )

                save_combined_results_with_evaluation(results_folder, color_image, bounding_image, mask, extracted_depth_colored,
                                                    depth_denoised_colored, thresh_colored, contoured_image,
                                                    evaluation_image, image_name, correctly_guessed, missed_crystal, incorrect_guess, dataset_path)
                
                successful_count += 1
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                failed_count += 1
                failed_images.append(f"{image_name} - Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        return
    
    create_failed_images_log(failed_images)
    create_evaluation_summary_with_confusion_matrix(results_folder, total_tp, total_fp, total_tn, total_fn, successful_count, failed_count)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total attempted: {successful_count + failed_count} images")
    
    print(f"\n=== Overall Evaluation Results (Middle Crop) ===")
    print_confusion_matrix_table(total_tp, total_fp, total_tn, total_fn)
    
    if failed_images:
        print(f"Failed images logged to: results/failed_images.txt")


def print_confusion_matrix_table(tp, fp, tn, fn):
    """Print a formatted confusion matrix table."""
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"{'':>20} | {'Predicted Positive':>15} | {'Predicted Negative':>15}")
    print("-"*60)
    print(f"{'Actual Positive':>20} | {tp:>15} | {fn:>15}")
    print(f"{'Actual Negative':>20} | {fp:>15} | {tn:>15}")
    print("-"*60)
    
    total_pixels = tp + fp + tn + fn
    if total_pixels > 0:
        accuracy = ((tp + tn) / total_pixels) * 100
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print(f"\nCLASSIFICATION METRICS:")
        print(f"True Positives (TP):    {tp:>10,}")
        print(f"False Positives (FP):   {fp:>10,}")
        print(f"True Negatives (TN):    {tn:>10,}")
        print(f"False Negatives (FN):   {fn:>10,}")
        print(f"Total Pixels:           {total_pixels:>10,}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"Accuracy:               {accuracy:>10.2f}%")
        print(f"Precision:              {precision:>10.2f}%")
        print(f"Recall (Sensitivity):   {recall:>10.2f}%")
        print(f"Specificity:            {specificity:>10.2f}%")
        print(f"F1-Score:               {f1_score:>10.2f}%")
    print("="*60)


def create_evaluation_summary_with_confusion_matrix(results_folder, total_tp, total_fp, total_tn, total_fn, successful_count, failed_count):
    """Create a summary file of the evaluation results including confusion matrix."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    summary_path = os.path.join(results_folder, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CRYSTAL DETECTION EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PROCESSING RESULTS:\n")
        f.write(f"Successfully processed: {successful_count} images\n")
        f.write(f"Failed to process: {failed_count} images\n")
        f.write(f"Total attempted: {successful_count + failed_count} images\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'':>20} | {'Predicted Positive':>15} | {'Predicted Negative':>15}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Actual Positive':>20} | {total_tp:>15,} | {total_fn:>15,}\n")
        f.write(f"{'Actual Negative':>20} | {total_fp:>15,} | {total_tn:>15,}\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("CLASSIFICATION COUNTS:\n")
        f.write(f"True Positives (TP):    {total_tp:>10,}\n")
        f.write(f"False Positives (FP):   {total_fp:>10,}\n")
        f.write(f"True Negatives (TN):    {total_tn:>10,}\n")
        f.write(f"False Negatives (FN):   {total_fn:>10,}\n")
        f.write(f"Total Pixels:           {total_tp + total_fp + total_tn + total_fn:>10,}\n\n")
        
        total_relevant = total_tp + total_fp + total_tn + total_fn
        if total_relevant > 0:
            accuracy = ((total_tp + total_tn) / total_relevant) * 100
            precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0
            recall = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0
            specificity = (total_tn / (total_tn + total_fp)) * 100 if (total_tn + total_fp) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            f.write(f"\nPERFORMANCE METRICS:\n")
            f.write(f"Accuracy:               {accuracy:.2f}%             (tp+tn)/(tp+tn+fp+fn)      - Measure of all predictions, how many were true\n")
            f.write(f"Precision:              {precision:.2f}%            (tp)/(tp+fp)               - Measure of positive predictions, how many were true\n")
            f.write(f"Recall:                 {recall:.2f}%               (tp)/(tp+fn)               - Measure of actual positives, how many were found\n")
            f.write(f"Specificity:            {specificity:.2f}%          (tn)/(tn+fp)               - Measure of actual negatives, how many were found\n\n")
            f.write(f"F1-Score:               {f1_score:.2f}%             (2*precision*recall)/(precision+recall)\n")

    print(f"Evaluation summary with confusion matrix created: {summary_path}")


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
        run_live_pipeline()
    
    elif choice == '2':
        # Process all images sequentially
        process_all_images(dataset_path)
    
    else:
        # Process all images with manual label evaluation
        process_all_images_with_evaluation(dataset_path)


# Keep existing functions for static image processing
def display_results(color_image, bounding_image, mask, extracted_depth_colored, 
                   depth_denoised_colored, thresh_colored, final_boxed_image):
    """Display all pipeline results in separate windows."""
    cv2.imshow('Color Image', color_image)
    cv2.imshow('Bounding Box', bounding_image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Depth Image', extracted_depth_colored)
    cv2.imshow('Denoised Depth', depth_denoised_colored)
    cv2.imshow('Thresholded Depth', thresh_colored)
    cv2.imshow('Boxed Map', final_boxed_image)
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def process_single_image(color_image, depth_image, image_name):
    """Process a single image through the entire pipeline."""
    print(f"\n=== Processing {image_name} ===")
    
    mask, bounding_image = detect_and_segment_vial(color_image)
    if mask is None or bounding_image is None:
        print(f"Vial detection failed for {image_name}. Skipping...")
        return False
    
    extracted_depth, depth_denoised, thresh = process_depth_data(mask, depth_image)
    contourmap = create_binary_map(thresh)
    individual_contour_images, contoured_image, midpoint_coords = analyze_crystals(contourmap, color_image)
    extracted_depth_colored, depth_denoised_colored, thresh_colored = create_visualizations(
        extracted_depth, depth_denoised, thresh
    )
    
    save_combined_results(color_image, bounding_image, mask, extracted_depth_colored,
                         depth_denoised_colored, thresh_colored, contoured_image, image_name)

    print(f"Successfully processed {image_name}")
    return True


def process_all_images(dataset_path):
    """Process all images sequentially without interactive display."""
    print("\n=== Processing All Images Sequentially ===")
    
    successful_count = 0
    failed_count = 0
    failed_images = []
    
    
    try:
        for color_image, depth_image, image_name in getdata.get_all_images(dataset_path):
            try:
                success = process_single_image(color_image, depth_image, image_name)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
                    failed_images.append(f"{image_name} - Vial detection failed")
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                failed_count += 1
                failed_images.append(f"{image_name} - Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        return
    
    create_failed_images_log(failed_images)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total attempted: {successful_count + failed_count} images")
    
    if failed_images:
        print(f"Failed images logged to: results/failed_images.txt")


def create_failed_images_log(failed_images):
    """Create a text file listing all failed images with reasons."""
    if not failed_images:
        return
    
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    log_file_path = os.path.join(results_folder, 'failed_images.txt')
    with open(log_file_path, 'w') as f:
        f.write("FAILED IMAGES LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total failed images: {len(failed_images)}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, failed_image in enumerate(failed_images, 1):
            f.write(f"{i:3d}. {failed_image}\n")
    
    print(f"Failed images log created: {log_file_path}")


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
