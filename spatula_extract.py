import cv2
import numpy as np
COLOR_SIMILARITY_THRESHOLD = 80  # Adjust this threshold as needed for color similarity


def subtract_spatula_mask(final_image, spatula_mask_path='spatula_mask.jpg'):
    """
    Step 1: Subtract spatula mask from final crystal detection image.
    
    Args:
        final_image (np.ndarray): Final crystal detection image with green regions
        spatula_mask_path (str): Path to the spatula mask image
        
    Returns:
        np.ndarray: Crystal regions without spatula overlap
    """
    # Load spatula mask
    spatula_mask = cv2.imread(spatula_mask_path)
    if spatula_mask is None:
        print(f"Could not load spatula mask from {spatula_mask_path}")
        return final_image
    
    # Ensure same dimensions
    if spatula_mask.shape != final_image.shape:
        spatula_mask = cv2.resize(spatula_mask, (final_image.shape[1], final_image.shape[0]))
    
    # Create binary mask for red pixels (spatula regions)
    red_lower = np.array([0, 0, 200])  # Lower bound for red (BGR)
    red_upper = np.array([50, 50, 255])  # Upper bound for red (BGR)
    spatula_binary = cv2.inRange(spatula_mask, red_lower, red_upper)
    
    # Create crystals_no_spat by removing spatula regions
    crystals_no_spat = final_image.copy()
    crystals_no_spat[spatula_binary > 0] = [0, 0, 0]  # Set spatula regions to black
    
    # Display result
    cv2.imshow('Crystals without Spatula', crystals_no_spat)
    print("Step 1: Spatula regions removed from crystal detection")
    
    return crystals_no_spat


def apply_crystal_mask_to_color(crystals_no_spat, color_image):
    """
    Step 2: Apply crystal mask (without spatula) to original color image.
    
    Args:
        crystals_no_spat (np.ndarray): Crystal regions without spatula
        color_image (np.ndarray): Original color image
        
    Returns:
        np.ndarray: Color image showing only crystal regions (no spatula)
    """
    # Create binary mask for green crystal regions
    green_lower = np.array([0, 200, 0])  # Lower bound for green (BGR)
    green_upper = np.array([100, 255, 100])  # Upper bound for green (BGR)
    crystal_binary = cv2.inRange(crystals_no_spat, green_lower, green_upper)
    
    # Apply mask to color image
    crystals_no_spat_color = color_image.copy()
    crystals_no_spat_color[crystal_binary == 0] = [0, 0, 0]  # Set non-crystal regions to black
    
    # Display result
    cv2.imshow('Crystal Colors (No Spatula)', crystals_no_spat_color)
    print("Step 2: Crystal regions isolated in color image")
    
    return crystals_no_spat_color

def calculate_average_crystal_color(crystals_no_spat_color):
    """
    Step 3: Calculate average color of crystal regions across all 3 BGR channels.
    
    Args:
        crystals_no_spat_color (np.ndarray): Color image with only crystal regions
        
    Returns:
        np.ndarray: Average BGR color of crystal regions
    """
    # Create mask for non-black pixels (crystal regions) - check all channels
    non_black_mask = np.any(crystals_no_spat_color > 0, axis=2)
    
    # Calculate average color of crystal regions across all 3 channels
    if np.any(non_black_mask):
        crystal_pixels = crystals_no_spat_color[non_black_mask]
        average_crystal_color = np.mean(crystal_pixels, axis=0)
        
        print(f"Step 3: Average crystal color (BGR): [{average_crystal_color[0]:.1f}, {average_crystal_color[1]:.1f}, {average_crystal_color[2]:.1f}]")
        print(f"Average crystal color (RGB): [{average_crystal_color[2]:.1f}, {average_crystal_color[1]:.1f}, {average_crystal_color[0]:.1f}]")
        
        return average_crystal_color
    else:
        print("Step 3: No crystal regions found for color averaging")
        return np.array([0.0, 0.0, 0.0])


def extract_colors_under_spatula(color_image, spatula_mask_path='spatula_mask.jpg'):
    """
    Step 4: Extract color regions under spatula mask.
    
    Args:
        color_image (np.ndarray): Original color image
        spatula_mask_path (str): Path to the spatula mask image
        
    Returns:
        np.ndarray: Color image showing only regions under spatula
    """
    # Load spatula mask
    spatula_mask = cv2.imread(spatula_mask_path)
    if spatula_mask is None:
        print(f"Could not load spatula mask from {spatula_mask_path}")
        return color_image
    
    # Ensure same dimensions
    if spatula_mask.shape != color_image.shape:
        spatula_mask = cv2.resize(spatula_mask, (color_image.shape[1], color_image.shape[0]))
    
    # Create binary mask for red pixels (spatula regions)
    red_lower = np.array([0, 0, 200])  # Lower bound for red (BGR)
    red_upper = np.array([50, 50, 255])  # Upper bound for red (BGR)
    spatula_binary = cv2.inRange(spatula_mask, red_lower, red_upper)
    
    # Apply spatula mask to color image
    colors_under_spat = color_image.copy()
    colors_under_spat[spatula_binary == 0] = [0, 0, 0]  # Set non-spatula regions to black
    
    # Display result
    cv2.imshow('Colors Under Spatula', colors_under_spat)
    print("Step 4: Colors under spatula extracted")
    
    return colors_under_spat


def find_crystals_under_spatula(colors_under_spat, average_crystal_color, threshold=COLOR_SIMILARITY_THRESHOLD):
    """
    Step 5: Find pixels under spatula that match crystal color across all 3 channels.
    
    Args:
        colors_under_spat (np.ndarray): Color image showing regions under spatula
        average_crystal_color (np.ndarray): Average BGR color of crystals
        threshold (float): Color similarity threshold (lower = more strict matching)
        
    Returns:
        np.ndarray: Binary mask of crystal-colored regions under spatula
    """
    # Create mask for non-black pixels (spatula regions) - check all channels
    spatula_mask = np.any(colors_under_spat > 0, axis=2)
    
    # Initialize crystals_under_spat as black image
    crystals_under_spat = np.zeros_like(colors_under_spat)
    
    # Find pixels similar to average crystal color
    if np.any(spatula_mask):
        # Calculate color differences for all spatula pixels across all 3 channels
        spatula_pixels = colors_under_spat[spatula_mask]
        color_differences = np.linalg.norm(spatula_pixels - average_crystal_color, axis=1)
        
        # Create boolean mask for pixels within threshold
        similar_pixels = color_differences <= threshold
        
        # Apply similar pixels to output image
        crystal_candidates = np.zeros_like(spatula_mask)
        crystal_candidates[spatula_mask] = similar_pixels
        
        # Set matching pixels to green in crystals_under_spat
        crystals_under_spat[crystal_candidates] = [0, 255, 0]  # Green color
        
        similar_count = np.sum(similar_pixels)
        total_spatula_pixels = np.sum(spatula_mask)
        
        print(f"Step 5: Found {similar_count} crystal-like pixels under spatula")
        print(f"Similarity threshold: {threshold}")
        print(f"Percentage of spatula area with crystals: {(similar_count/total_spatula_pixels)*100:.2f}%")
        
        # Show color comparison info
        if similar_count > 0:
            matching_colors = spatula_pixels[similar_pixels]
            avg_matching_color = np.mean(matching_colors, axis=0)
            print(f"Average matching color (BGR): [{avg_matching_color[0]:.1f}, {avg_matching_color[1]:.1f}, {avg_matching_color[2]:.1f}]")
            print(f"Target crystal color (BGR): [{average_crystal_color[0]:.1f}, {average_crystal_color[1]:.1f}, {average_crystal_color[2]:.1f}]")
    else:
        print("Step 5: No spatula regions found")
    
    # Display result
    cv2.imshow('Crystals Under Spatula', crystals_under_spat)
    
    return crystals_under_spat


def process_spatula_separation(final_image, color_image, spatula_mask_path='spatula_mask.jpg'):
    """
    Complete spatula separation pipeline.
    
    Args:
        final_image (np.ndarray): Final crystal detection image with green regions
        color_image (np.ndarray): Original color image
        spatula_mask_path (str): Path to the spatula mask image
        
    Returns:
        dict: Dictionary containing all intermediate results
    """
    print("\n=== Starting Spatula Separation Pipeline ===")
    
    # Step 1: Subtract spatula mask from crystal detection
    crystals_no_spat = subtract_spatula_mask(final_image, spatula_mask_path)
    
    # Step 2: Apply crystal mask to color image
    crystals_no_spat_color = apply_crystal_mask_to_color(crystals_no_spat, color_image)
    
    # Step 3: Calculate average crystal color
    average_crystal_color = calculate_average_crystal_color(crystals_no_spat_color)
    
    # Step 4: Extract colors under spatula
    colors_under_spat = extract_colors_under_spatula(color_image, spatula_mask_path)
    
    # Step 5: Find crystals under spatula
    crystals_under_spat = find_crystals_under_spatula(colors_under_spat, average_crystal_color)
    
    print("=== Spatula Separation Pipeline Complete ===")
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return all results
    return {
        'crystals_no_spat': crystals_no_spat,
        'crystals_no_spat_color': crystals_no_spat_color,
        'average_crystal_color': average_crystal_color,
        'colors_under_spat': colors_under_spat,
        'crystals_under_spat': crystals_under_spat
    }