'''This file is to save the results of the pipeline to a file, as well as create human visualizations
for the results.'''

import os
import cv2
import numpy as np
from datetime import datetime
from image_data import ImageData

def create_failed_images_log(failed_images, results_path='results'):
    """Create a text file listing all failed images with reasons."""
    if not failed_images:
        return

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log_file_path = os.path.join(results_path, 'failed_images.txt')
    with open(log_file_path, 'w') as f:
        f.write("FAILED IMAGES LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total failed images: {len(failed_images)}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log_file_path = os.path.join(results_path, 'failed_images.txt')
    with open(log_file_path, 'w') as f:
        f.write("FAILED IMAGES LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total failed images: {len(failed_images)}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, failed_image in enumerate(failed_images, 1):
            f.write(f"{i:3d}. {failed_image}\n")
    
    print(f"Failed images log created: {log_file_path}")


def save_combined_results(image_data, results_path = 'results'):
    """
    Combine all results into a single image and save to results folder.
    
    Args:
        image_data (ImageData): ImageData object containing all processed image data
        
    Returns:
        str: Path to saved combined image
    """
    # Create results folder
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Convert mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(image_data.grabcut_mask, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions for dynamic scaling
    h, w = image_data.color_image.shape[:2]
    
    # Calculate dynamic text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))
    text_y = max(30, int(h / 20))
    
    # Create labeled copies
    images_to_label = [
        (image_data.color_image.copy(), 'Original'),
        (image_data.bounding_image.copy(), 'YOLO Detection'),
        (mask_colored.copy(), 'GrabCut Mask'),
        (image_data.extracted_depth_colored.copy(), 'Extracted Depth'),
        (image_data.depth_denoised_colored.copy(), 'Denoised Depth'),
        (image_data.thresh_colored.copy(), 'Thresholded'),
        (image_data.contoured_image.copy(), 'Crystal Analysis'),
        (image_data.spatula_results_smoothed_cropped.copy(), 'Final Crystals')
    ]
    
    labeled_images = []
    for img, label in images_to_label:
        cv2.putText(img, label, (10, text_y), font, font_scale, color_white, thickness)
        labeled_images.append(img)
    
    # Combine images in 2x4 grid
    top_row = np.hstack(labeled_images[:4])
    bottom_row = np.hstack(labeled_images[4:])
    combined_image = np.vstack([top_row, bottom_row])
    
    # Save combined image
    base_name = os.path.splitext(image_data.image_name)[0]
    output_path = os.path.join(results_path, f"{base_name}_results.jpg")
    cv2.imwrite(output_path, combined_image)
    
    print(f"Combined results saved to: {output_path}")
    
    return output_path


def create_visualizations(image_data):
    """
    Apply colormaps to depth images for visualization.
    
    Args:
        image_data (ImageData): ImageData object containing extracted_depth, depth_denoised, and thresh
        
    Returns:
        ImageData: Updated ImageData object with colored visualizations
    """
    # Apply JET colormap to each depth image
    image_data.extracted_depth_colored = cv2.normalize(image_data.extracted_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_data.extracted_depth_colored = cv2.applyColorMap(image_data.extracted_depth_colored, cv2.COLORMAP_JET)
    
    image_data.depth_denoised_colored = cv2.normalize(image_data.depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_data.depth_denoised_colored = cv2.applyColorMap(image_data.depth_denoised_colored, cv2.COLORMAP_JET)
    
    image_data.thresh_colored = cv2.normalize(image_data.thresh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_data.thresh_colored = cv2.applyColorMap(image_data.thresh_colored, cv2.COLORMAP_JET);
    
    return image_data


def save_simplified_results(image_data, results_path='results'):
    """
    Combine original image and final crystal results into a single image and save to results folder.
    
    Args:
        image_data (ImageData): ImageData object containing color_image and spatula_results_smoothed_cropped
        results_path (str): Path to save results
        
    Returns:
        str: Path to saved combined image
    """
    # Create results folder
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Get image dimensions for dynamic scaling
    h, w = image_data.color_image.shape[:2]
    
    # Calculate dynamic text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))
    text_y = max(30, int(h / 20))

    # Draw rectangle around detected crystals
    if image_data.cropped_rect:
        cropped_x_start, cropped_y_start, width, height = image_data.cropped_rect
        cv2.rectangle(image_data.color_image, 
                      (cropped_x_start, cropped_y_start), 
                      (cropped_x_start + width, cropped_y_start + height), 
                      (0, 0, 255),  # Red color in BGR format
                      thickness=2)

    # Create labeled copies
    images_to_label = [
        (image_data.color_image.copy(), 'Original Image'),
        (image_data.final_midpoints_image.copy(), 'Detected Crystals')
    ]
    
    labeled_images = []
    for img, label in images_to_label:
        cv2.putText(img, label, (10, text_y), font, font_scale, color_white, thickness)
        labeled_images.append(img)
    
    # Combine images side by side (1x2 grid)
    combined_image = np.hstack(labeled_images)
    
    # Add percentage coverage text to bottom right
    if hasattr(image_data, 'percentage_coverage') and image_data.percentage_coverage is not None:
        coverage_text = f"Coverage: {image_data.percentage_coverage:.1f}%"
        text_size = cv2.getTextSize(coverage_text, font, font_scale, thickness)[0]
        text_x = combined_image.shape[1] - text_size[0] - 10
        text_y_bottom = combined_image.shape[0] - 10
        cv2.putText(combined_image, coverage_text, (text_x, text_y_bottom), font, font_scale, color_white, thickness)

    # Add percentage coverage of each midpoint (top right)
    if hasattr(image_data, 'final_midpoints') and image_data.final_midpoints is not None:
        for i, (x, y, count) in enumerate(image_data.final_midpoints):
            coverage_text = f"Midpoint {i+1}: {count:.1f}%"
            text_size = cv2.getTextSize(coverage_text, font, font_scale, thickness)[0]
            text_x = combined_image.shape[1] - text_size[0] - 10
            text_y_top = text_y + i * (text_size[1] + 10)
            cv2.putText(combined_image, coverage_text, (text_x, text_y_top), font, font_scale, color_white, thickness)
    # Save combined image
    base_name = os.path.splitext(image_data.image_name)[0]
    output_path = os.path.join(results_path, f"{base_name}_simplified.jpg")
    cv2.imwrite(output_path, combined_image)
    
    #print(f"Simplified results saved to: {output_path}")
    
    return output_path
