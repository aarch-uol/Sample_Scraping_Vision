import cv2
import numpy as np
import pyrealsense2 as rs
import grabcut
import depth_extract
import denoise
import getdata
import midcluster
import YOLO_boundingbox
import os



def midpoint_depth(depth_image, threshold_ratio=0.5):
    depth_image_with_thresh = np.copy(depth_image)
    # Iterate through the depth image to find the closest and furthest points
    closest = np.inf
    furthest = -np.inf
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] > 0:
                if depth_image[i, j] < closest:
                    closest = depth_image[i, j]
                if depth_image[i, j] > furthest:
                    furthest = depth_image[i, j]
    if closest == np.inf or furthest == -np.inf:
        print("No valid depth values found in the image.")
        return None
    threshold = closest + threshold_ratio * (furthest - closest)
    depth_image_with_thresh[depth_image > threshold] = 0
    print(closest, furthest)

    return threshold



def draw_boxes_on_binary_image(binary_img, original_image, box_w=10, box_h=10, threshold_perc=25):
    """
    Draws boxes over significant areas in a binary image (white regions) and marks contour midpoints with X.
    
    Parameters:
        binary_img (np.ndarray): Binary image (white on black).
        original_img (np.ndarray): Optional original image to draw boxes on. 
                                   If None, a color version of binary image is used.
        box_w (int): Width of scanning box.
        box_h (int): Height of scanning box.
        threshold_perc (int): % of white pixels in box needed to trigger a drawn rectangle.
        
    Returns:
        np.ndarray: Image with blue rectangles drawn and X marks at contour midpoints.
    """
    output = np.copy(original_image) 
    # Ensure the binary image is 8-bit single channel
    assert len(binary_img.shape) == 2, "Input must be a single-channel binary image"
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create blank canvas to draw filled contours
    dst = np.zeros_like(binary_img)

    # Calculate pixel threshold for box
    pixel_threshold = (box_w * box_h * threshold_perc) // 100
    
    output_images = []
    final_output = original_image.copy()
    midpoint_coords = []
    contour_midpoints = []
    
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            continue

        output = original_image.copy()
        dst = np.zeros_like(dst)

        # Draw filled contour
        cv2.drawContours(dst, contours, i, 255, thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(final_output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)

        # Calculate contour midpoint using moments
        M = cv2.moments(contours[i])
        if M["m00"] != 0:
            contour_cx = int(M["m10"] / M["m00"])
            contour_cy = int(M["m01"] / M["m00"])
            contour_midpoints.append((contour_cx, contour_cy))
            
            # Draw large X at contour midpoint
            x_size = 20  # Size of the X
            thickness = 3
            color_red = (0, 0, 255)  # Red color for X
            
            # Draw X lines
            cv2.line(output, (contour_cx - x_size, contour_cy - x_size), 
                    (contour_cx + x_size, contour_cy + x_size), color_red, thickness)
            cv2.line(output, (contour_cx + x_size, contour_cy - x_size), 
                    (contour_cx - x_size, contour_cy + x_size), color_red, thickness)
            
            cv2.line(final_output, (contour_cx - x_size, contour_cy - x_size), 
                    (contour_cx + x_size, contour_cy + x_size), color_red, thickness)
            cv2.line(final_output, (contour_cx + x_size, contour_cy - x_size), 
                    (contour_cx - x_size, contour_cy + x_size), color_red, thickness)

        # Get bounding box for box scanning
        x, y, w, h = cv2.boundingRect(contours[i])

        # Scan with box inside the bounding rectangle
        for j in range(x, x + w, box_w):
            for k in range(y, y + h, box_h):
                # Check if box fits within image bounds
                if j + box_w <= dst.shape[1] and k + box_h <= dst.shape[0]:
                    roi = dst[k:k + box_h, j:j + box_w]
                    count = cv2.countNonZero(roi)

                    if count > pixel_threshold:
                        cv2.rectangle(output, (j, k), (j + box_w, k + box_h), (255, 0, 0), 1, lineType=8)
                        cv2.rectangle(final_output, (j, k), (j + box_w, k + box_h), (255, 0, 0), 1, lineType=8)
                        x_mid = (j + box_w/2)
                        y_mid = (k + box_h/2)
                        final_output = cv2.circle(final_output, (int(x_mid), int(y_mid)), 1, (0,0,255), 1)
                        midpoint_coords.append((x_mid, y_mid))
        output_images.append(output)
    
    # Print contour midpoints for debugging
    print(f"Found {len(contour_midpoints)} contours with midpoints: {contour_midpoints}")
    
    return output_images, final_output, midpoint_coords

def detect_and_segment_vial(color_image):
    """Detect vial using YOLO and segment it using GrabCut"""
    rect = YOLO_boundingbox.find_bounding_box(color_image)
    if rect is None:
        print("No bounding box found. Exiting...")
        return None, None
    bounding_image = color_image.copy()
    cv2.rectangle(bounding_image, (rect[0],rect[1]), (rect[2], rect[3]), (255,0,0), 3)
    print('rectangle:', rect)
    
    # Convert rect format for GrabCut
    rect = list(rect)
    rect[2] = rect[2] - rect[0]
    rect[3] = rect[3] - rect[1]
    rect = tuple(rect)
    
    # Apply GrabCut algorithm
    mask = grabcut.apply_grabcut(color_image, rect, iter_count=5)
    return mask, bounding_image

def process_depth_data(mask, depth_image):
    """Extract, denoise and threshold depth data"""
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
    """Convert thresholded depth to binary map"""
    contourmap = thresh.copy()
    contourmap[contourmap > 0] = 255
    contourmap[contourmap <= 0] = 0
    contourmap = contourmap.astype(np.uint8)
    return contourmap

def analyze_crystals(contourmap, color_image):
    """Detect and analyze crystal formations"""
    individual_boxed_images, final_boxed_image, midpoint_coords = draw_boxes_on_binary_image(
        contourmap, color_image, box_w=10, box_h=10, threshold_perc=10
    )
    print(midpoint_coords)
    return individual_boxed_images, final_boxed_image, midpoint_coords

def create_visualizations(extracted_depth, depth_denoised, thresh):
    """Apply colormaps to depth images for visualization"""
    # Original Depth Image
    extracted_depth_colored = cv2.normalize(extracted_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    extracted_depth_colored = cv2.applyColorMap(extracted_depth_colored, cv2.COLORMAP_JET)
    
    # Denoised Depth Image
    depth_denoised_colored = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_denoised_colored = cv2.applyColorMap(depth_denoised_colored, cv2.COLORMAP_JET)
    
    # Thresholded Depth Image
    thresh_colored = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_colored = cv2.applyColorMap(thresh_colored, cv2.COLORMAP_JET)
    
    return extracted_depth_colored, depth_denoised_colored, thresh_colored

def display_results(color_image, bounding_image, mask, extracted_depth_colored, 
                   depth_denoised_colored, thresh_colored, final_boxed_image):
    """Display all pipeline results"""
    cv2.imshow('Color Image', color_image)
    cv2.imshow('Bounding Box', bounding_image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Depth Image', extracted_depth_colored)
    cv2.imshow('Denoised Depth', depth_denoised_colored)
    cv2.imshow('Thresholded Depth', thresh_colored)
    cv2.imshow('Boxed Map', final_boxed_image)
    
    # Wait for a key press
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

def save_combined_results(color_image, bounding_image, mask, extracted_depth_colored, 
                         depth_denoised_colored, thresh_colored, final_boxed_image, image_name):
    """Combine all results into a single image and save to results folder"""
    # Create results folder if it doesn't exist
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Convert single channel mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Use original image dimensions (no resizing)
    h, w = color_image.shape[:2]
    
    # Add text labels to each image (adjust font scale based on image size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))  # Scale font based on image width
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))  # Scale thickness based on image width
    text_y = max(30, int(h / 20))  # Scale text position based on image height
    
    # Create copies for labeling
    color_labeled = color_image.copy()
    bounding_labeled = bounding_image.copy()
    mask_labeled = mask_colored.copy()
    depth_labeled = extracted_depth_colored.copy()
    denoised_labeled = depth_denoised_colored.copy()
    thresh_labeled = thresh_colored.copy()
    boxed_labeled = final_boxed_image.copy()
    
    cv2.putText(color_labeled, 'Original', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(bounding_labeled, 'YOLO Detection', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(mask_labeled, 'GrabCut Mask', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(depth_labeled, 'Extracted Depth', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(denoised_labeled, 'Denoised Depth', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(thresh_labeled, 'Thresholded', (10, text_y), font, font_scale, color_white, thickness)
    cv2.putText(boxed_labeled, 'Crystal Analysis', (10, text_y), font, font_scale, color_white, thickness)
    
    # Combine images in a grid (2x4 layout) at full resolution
    # Top row
    top_row = np.hstack([color_labeled, bounding_labeled, mask_labeled, depth_labeled])
    # Bottom row
    bottom_row = np.hstack([denoised_labeled, thresh_labeled, boxed_labeled, 
                           np.zeros_like(color_labeled)])  # Empty space for alignment
    
    # Combine rows
    combined_image = np.vstack([top_row, bottom_row])
    
    # Extract base filename without extension
    base_name = os.path.splitext(image_name)[0]
    output_path = os.path.join(results_folder, f"{base_name}_results.jpg")
    
    # Save the combined image at full resolution
    cv2.imwrite(output_path, combined_image)
    print(f"Combined results saved to: {output_path}")
    print(f"Combined image size: {combined_image.shape}")
    
    return output_path

def main():
    """Main pipeline orchestrator - allows choosing between single random image or all images"""
    
    process_single_random_image()
    
    #process_all_images()

def process_single_random_image():
    """Process a single random image with interactive display"""
    print("\n=== Processing Single Random Image ===")
    
    try:
        color_image, depth_image, image_name = getdata.get_random_image()
        print(f"Processing: {image_name}")
        
        # Step 1: Detect and segment vial
        mask, bounding_image = detect_and_segment_vial(color_image)
        
        # Check if detection failed
        if mask is None or bounding_image is None:
            print(f"Vial detection failed for {image_name}.")
            return
        
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
        
        # Step 7: Display results interactively
        print(f"Displaying results for {image_name}. Press 'q' to close windows.")
        display_results(color_image, bounding_image, mask, extracted_depth_colored,
                       depth_denoised_colored, thresh_colored, final_boxed_image)
        
        print(f"Successfully processed {image_name}")
        
    except Exception as e:
        print(f"Error processing random image: {e}")

def process_all_images():
    """Process all images sequentially without interactive display"""
    print("\n=== Processing All Images Sequentially ===")
    successful_count = 0
    failed_count = 0
    
    try:
        for color_image, depth_image, image_name in getdata.get_all_images():
            try:
                success = process_single_image(color_image, depth_image, image_name)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                failed_count += 1
                continue
                
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        return
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total attempted: {successful_count + failed_count} images")

def process_single_image(color_image, depth_image, image_name):
    """Process a single image through the pipeline (for batch processing)"""
    print(f"\n=== Processing {image_name} ===")
    
    # Step 1: Detect and segment vial
    mask, bounding_image = detect_and_segment_vial(color_image)
    
    # Check if detection failed
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
    return True

# ...existing code...

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
