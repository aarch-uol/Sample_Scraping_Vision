import cv2
import numpy as np
import os
from datetime import datetime
import saving
import getdata
from image_data import ImageData

def compare_with_manual_labels(image_data, crop_percent=1, dataset_path='Test_dataset2'):
    """
    Compare generated contours with manual labels and create evaluation metrics.
    
    Args:
        image_data (ImageData): ImageData object containing all processed image data
        crop_percent (float): Percentage of YOLO bounding box width to use for evaluation
        dataset_path (str): Path to the dataset containing manual labels
        
    Returns:
        tuple: (correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn)
    """
    # Load manual label image
    manual_label_path = os.path.join(dataset_path, 'manual_label', f"{image_data.image_name}.jpg")

    if not os.path.exists(manual_label_path):
        print(f"Manual label not found: {manual_label_path}")
        return 0, 0, 0, image_data.contoured_image, 0, 0, 0, 0
    
    manual_label = cv2.imread(manual_label_path)
    if manual_label is None:
        print(f"Could not load manual label: {manual_label_path}")
        return 0, 0, 0, image_data.contoured_image, 0, 0, 0, 0
    
    print(f"Comparing with manual label: {manual_label_path}")
    
    # Ensure both images have the same dimensions
    if image_data.contoured_image.shape != manual_label.shape:
        manual_label = cv2.resize(manual_label, (image_data.contoured_image.shape[1], image_data.contoured_image.shape[0]))
    
    # Ensure mask has same dimensions
    if image_data.grabcut_mask.shape[:2] != image_data.contoured_image.shape[:2]:
        mask = cv2.resize(image_data.grabcut_mask, (image_data.contoured_image.shape[1], image_data.contoured_image.shape[0]))
    else:
        mask = image_data.grabcut_mask
    
    # Calculate crop region (middle crop_percent of YOLO bounding box width)
    crop_x_start = 0
    crop_x_end = image_data.contoured_image.shape[1]
    crop_y_start = 0
    crop_y_end = image_data.contoured_image.shape[0]
    
    if image_data.yolo_rect is not None:
        x, y, width, height = image_data.yolo_rect
        # Calculate middle crop_percent of width
        crop_width = int(width * crop_percent)
        crop_x_offset = int(width * (1 - crop_percent) / 2)  # Center the crop region horizontally

        # Calculate crop region with 25% removed from top
        crop_height_offset = int(height * 0.25)  # Remove 25% from top

        crop_x_start = max(0, x + crop_x_offset)
        crop_x_end = min(image_data.contoured_image.shape[1], x + crop_x_offset + crop_width)
        crop_y_start = max(0, y + crop_height_offset)  # Start 25% down from top to remove the threads.
        crop_y_end = min(image_data.contoured_image.shape[0], y + height)
        
        print(f"YOLO bounding box: {image_data.yolo_rect}")
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
        original_color_path = os.path.join(dataset_path, 'color_images', f"{image_data.image_name}.jpg")
        evaluation_image = cv2.imread(original_color_path)
        if evaluation_image is None:
            evaluation_image = image_data.contoured_image.copy()
    except:
        evaluation_image = image_data.contoured_image.copy()
    
    # Define color thresholds for green detection (BGR format)
    green_lower = np.array([0, 200, 0])    # Lower bound for green
    green_upper = np.array([100, 255, 100]) # Upper bound for green
    
    # Process each pixel within the cropped region and mask
    for y in range(crop_y_start, crop_y_end):
        for x in range(crop_x_start, crop_x_end):
            # Only process pixels within the vial mask
            if mask[y, x] > 0:  # Inside vial area
                # Get pixel values
                generated_pixel = image_data.contoured_image[y, x]
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
    if image_data.yolo_rect is not None:
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

        print(f"Evaluation Results for {image_data.image_name} (cropped to middle {crop_percent * 100:.0f}%):")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall (Sensitivity): {recall:.2f}%")
        print(f"  Specificity: {specificity:.2f}%")
    
    return correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn

def save_combined_results_with_evaluation(results_folder, image_data, evaluation_image, correctly_guessed, missed_crystal, incorrect_guess, dataset_path):
    """
    Combine all results including evaluation comparison into a single image and save to eval_results folder.
    
    Args:
        results_folder (str): Folder to save evaluation results
        image_data (ImageData): ImageData object containing all processed image data
        evaluation_image (np.ndarray): Evaluation comparison image
        correctly_guessed (int): Number of correctly guessed pixels
        missed_crystal (int): Number of missed crystal pixels
        incorrect_guess (int): Number of incorrect guess pixels
        dataset_path (str): Path to dataset for manual labels
        
    Returns:
        str: Path to saved combined image
    """
    # Create eval_results folder
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Convert mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(image_data.grabcut_mask, cv2.COLOR_GRAY2BGR)
    
    # Load and create manual label superimposed image
    manual_label_path = os.path.join(dataset_path, 'manual_label', f"{image_data.image_name}.jpg")
    manual_superimposed = image_data.color_image.copy()
    
    if os.path.exists(manual_label_path):
        manual_label = cv2.imread(manual_label_path)
        if manual_label is not None:
            # Ensure same dimensions
            if manual_label.shape != image_data.color_image.shape:
                manual_label = cv2.resize(manual_label, (image_data.color_image.shape[1], image_data.color_image.shape[0]))
            
            # Define green color threshold for manual labels
            green_lower = np.array([0, 200, 0])
            green_upper = np.array([100, 255, 100])
            
            # Create mask for green areas in manual label
            green_mask = cv2.inRange(manual_label, green_lower, green_upper)
            
            # Superimpose green areas from manual label onto original image
            manual_superimposed[green_mask > 0] = [0, 255, 0]  # Green color
    
    # Get image dimensions for dynamic scaling
    h, w = image_data.color_image.shape[:2]
    
    # Calculate dynamic text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, w / 800))
    color_white = (255, 255, 255)
    thickness = max(1, int(w / 400))
    text_y = max(30, int(h / 20))
    
    # Create labeled copies - now with 9 images in 3x3 grid
    images_to_label = [
        (image_data.color_image.copy(), 'Original'),
        (image_data.bounding_image.copy(), 'YOLO Detection'),
        (mask_colored.copy(), 'GrabCut Mask'),
        (image_data.extracted_depth_colored.copy(), 'Extracted Depth'),
        (image_data.depth_denoised_colored.copy(), 'Denoised Depth'),
        (image_data.thresh_colored.copy(), 'Thresholded'),
        (image_data.contoured_image.copy(), 'Crystal Analysis'),
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
    base_name = os.path.splitext(image_data.image_name)[0]
    output_path = os.path.join(results_folder, f"{base_name}_evaluation_results.jpg")
    cv2.imwrite(output_path, combined_image)
    
    print(f"Combined evaluation results saved to: {output_path}")
    
    return output_path


def process_all_images_with_evaluation(dataset_path, process_single_image_func):
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
                #create an image data object
                image_data = ImageData(color_image=color_image, depth_image=depth_image, image_name=image_name)
                # Process the image using the provided function
                success, image_data = process_single_image_func(image_data)
                if not success:
                    failed_count += 1
                    failed_images.append(f"{image_name} - Vial detection failed")
                    continue
                # Compare with manual labels (now includes yolo_rect parameter)
                correctly_guessed, missed_crystal, incorrect_guess, evaluation_image, tp, fp, tn, fn = compare_with_manual_labels(
                   image_data, crop_percent=0.4, dataset_path=dataset_path
                )
                
                # Add to totals
                total_correctly_guessed += correctly_guessed
                total_missed_crystal += missed_crystal
                total_incorrect_guess += incorrect_guess
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                

                save_combined_results_with_evaluation(results_folder, image_data, evaluation_image, correctly_guessed, missed_crystal, incorrect_guess, dataset_path)
                
                successful_count += 1
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                failed_count += 1
                failed_images.append(f"{image_name} - Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        return
    
    saving.create_failed_images_log(failed_images)
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

