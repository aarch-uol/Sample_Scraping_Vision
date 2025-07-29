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


def draw_boxes_on_binary_image(binary_img, original_image, box_w=10, box_h=10, threshold_perc=25):
    """
    Draw boxes over significant areas in binary image and mark contour midpoints.
    
    Args:
        binary_img (np.ndarray): Binary image (white on black)
        original_image (np.ndarray): Original image to draw boxes on
        box_w (int): Width of scanning box
        box_h (int): Height of scanning box
        threshold_perc (int): Percentage of white pixels needed to trigger a rectangle
        
    Returns:
        tuple: (individual_boxed_images, final_output, midpoint_coords)
    """
    # Validate input
    assert len(binary_img.shape) == 2, "Input must be a single-channel binary image"
    
    # Initialize variables
    output = np.copy(original_image)
    final_output = original_image.copy()
    output_images = []
    midpoint_coords = []
    contour_midpoints = []
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dst = np.zeros_like(binary_img)
    
    # Calculate pixel threshold for box scanning
    pixel_threshold = (box_w * box_h * threshold_perc) // 100
    
    # Process each contour
    for i in range(len(contours)):
        # Skip inner contours
        if hierarchy[0][i][3] != -1:
            continue
        
        # Reset for each contour
        output = original_image.copy()
        dst = np.zeros_like(dst)
        
        # Draw filled contour
        cv2.drawContours(dst, contours, i, 255, thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(final_output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        
        # Calculate and mark contour midpoint
        M = cv2.moments(contours[i])
        if M["m00"] != 0:
            contour_cx = int(M["m10"] / M["m00"])
            contour_cy = int(M["m01"] / M["m00"])
            contour_midpoints.append((contour_cx, contour_cy))
            
            # Draw large X at contour midpoint
            x_size = 20
            thickness = 3
            color_red = (0, 0, 255)
            
            # Draw X lines on both output and final_output
            for target_img in [output, final_output]:
                cv2.line(target_img, (contour_cx - x_size, contour_cy - x_size), 
                        (contour_cx + x_size, contour_cy + x_size), color_red, thickness)
                cv2.line(target_img, (contour_cx + x_size, contour_cy - x_size), 
                        (contour_cx - x_size, contour_cy + x_size), color_red, thickness)
        
        # Get bounding box for scanning
        x, y, w, h = cv2.boundingRect(contours[i])
        
        # Scan with boxes inside the bounding rectangle
        for j in range(x, x + w, box_w):
            for k in range(y, y + h, box_h):
                # Check if box fits within image bounds
                if j + box_w <= dst.shape[1] and k + box_h <= dst.shape[0]:
                    roi = dst[k:k + box_h, j:j + box_w]
                    count = cv2.countNonZero(roi)
                    
                    # Draw rectangle if threshold is met
                    if count > pixel_threshold:
                        cv2.rectangle(output, (j, k), (j + box_w, k + box_h), (255, 0, 0), 1, lineType=8)
                        cv2.rectangle(final_output, (j, k), (j + box_w, k + box_h), (255, 0, 0), 1, lineType=8)
                        
                        # Mark midpoint of the box
                        x_mid = (j + box_w / 2)
                        y_mid = (k + box_h / 2)
                        cv2.circle(final_output, (int(x_mid), int(y_mid)), 1, (0, 0, 255), 1)
                        midpoint_coords.append((x_mid, y_mid))
        
        output_images.append(output)
    
    # Debug output
    print(f"Found {len(contour_midpoints)} contours with midpoints: {contour_midpoints}")
    
    return output_images, final_output, midpoint_coords


def detect_and_segment_vial(color_image):
    """
    Detect vial using YOLO and segment it using GrabCut.
    
    Args:
        color_image (np.ndarray): Input color image
        
    Returns:
        tuple: (mask, bounding_image) or (None, None) if detection fails
    """
    # Get bounding box from YOLO
    rect = YOLO_boundingbox.find_bounding_box(color_image)
    if rect is None:
        print("No bounding box found. Exiting...")
        return None, None
    
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
    
    return mask, bounding_image


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
        tuple: (individual_boxed_images, final_boxed_image, midpoint_coords)
    """
    individual_boxed_images, final_boxed_image, midpoint_coords = draw_boxes_on_binary_image(
        contourmap, color_image, box_w=10, box_h=10, threshold_perc=10
    )
    print(f"Crystal midpoint coordinates: {midpoint_coords}")
    
    return individual_boxed_images, final_boxed_image, midpoint_coords


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
    thresh_colored = cv2.applyColorMap(thresh_colored, cv2.COLORMAP_JET)
    
    return extracted_depth_colored, depth_denoised_colored, thresh_colored


def display_results(color_image, bounding_image, mask, extracted_depth_colored, 
                   depth_denoised_colored, thresh_colored, final_boxed_image):
    """
    Display all pipeline results in separate windows.
    
    Args:
        color_image (np.ndarray): Original color image
        bounding_image (np.ndarray): Image with YOLO bounding box
        mask (np.ndarray): GrabCut segmentation mask
        extracted_depth_colored (np.ndarray): Colored depth visualization
        depth_denoised_colored (np.ndarray): Colored denoised depth
        thresh_colored (np.ndarray): Colored thresholded depth
        final_boxed_image (np.ndarray): Final result with crystal analysis
    """
    # Display all images
    cv2.imshow('Color Image', color_image)
    cv2.imshow('Bounding Box', bounding_image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Depth Image', extracted_depth_colored)
    cv2.imshow('Denoised Depth', depth_denoised_colored)
    cv2.imshow('Thresholded Depth', thresh_colored)
    cv2.imshow('Boxed Map', final_boxed_image)
    
    # Wait for user to press 'q'
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


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
    print(f"Combined image size: {combined_image.shape}")
    
    return output_path


def create_failed_images_log(failed_images):
    """
    Create a text file listing all failed images with reasons.
    
    Args:
        failed_images (list): List of failed image descriptions
    """
    if not failed_images:
        return
    
    # Ensure results folder exists
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Write log file
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


def process_single_image(color_image, depth_image, image_name):
    """
    Process a single image through the entire pipeline.
    
    Args:
        color_image (np.ndarray): Input color image
        depth_image (np.ndarray): Input depth image
        image_name (str): Name of the image being processed
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print(f"\n=== Processing {image_name} ===")
    
    # Step 1: Detect and segment vial
    mask, bounding_image = detect_and_segment_vial(color_image)
    if mask is None or bounding_image is None:
        print(f"Vial detection failed for {image_name}. Skipping...")
        return False
    
    # Step 2: Process depth data
    extracted_depth, depth_denoised, thresh = process_depth_data(mask, depth_image)
    
    # Step 3: Create binary map
    contourmap = create_binary_map(thresh)
    
    # Step 4: Analyze crystals
    individual_boxed_images, final_boxed_image, midpoint_coords = analyze_crystals(contourmap, color_image)
    
    # Step 5: Create visualizations
    extracted_depth_colored, depth_denoised_colored, thresh_colored = create_visualizations(
        extracted_depth, depth_denoised, thresh
    )
    
    # Step 6: Save combined results
    save_combined_results(color_image, bounding_image, mask, extracted_depth_colored,
                         depth_denoised_colored, thresh_colored, final_boxed_image, image_name)
    
    print(f"Successfully processed {image_name}")
    return True, bounding_image, mask, extracted_depth_colored, depth_denoised_colored, thresh_colored, final_boxed_image


def process_single_random_image():
    """Process a single random image with interactive display."""
    try:
        # Load random image
        color_image, depth_image, image_name = getdata.get_random_image()
        
        # Process through pipeline
        success, bounding_image, mask, extracted_depth_colored, depth_denoised_colored, thresh_colored, final_boxed_image = process_single_image(color_image, depth_image, image_name)
        if not success:
            print(f"Processing failed for {image_name}.")
            return
        # Display results
        display_results(color_image, bounding_image, mask, extracted_depth_colored,
                       depth_denoised_colored, thresh_colored, final_boxed_image)
        
    except Exception as e:
        print(f"Error processing random image: {e}")


def process_all_images():
    """Process all images sequentially without interactive display."""
    print("\n=== Processing All Images Sequentially ===")
    
    # Initialize counters and tracking
    successful_count = 0
    failed_count = 0
    failed_images = []
    
    try:
        # Process each image in the dataset
        for color_image, depth_image, image_name in getdata.get_all_images():
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
    
    # Create failed images log
    create_failed_images_log(failed_images)
    
    # Print summary
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total attempted: {successful_count + failed_count} images")
    
    if failed_images:
        print(f"Failed images logged to: results/failed_images.txt")


def main():
    """Main pipeline orchestrator."""
    # Toggle between single random image and all images
    # Uncomment the desired processing mode:
    
    process_single_random_image()  # For single image with interactive display
    #process_all_images()             # For batch processing all images


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
