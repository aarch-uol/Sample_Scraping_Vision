import cv2
import numpy as np
import pyrealsense2 as rs
import grabcut
import depth_extract
import denoise
import getdata
import midcluster
import YOLO_boundingbox



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
    Draws boxes over significant areas in a binary image (white regions).
    
    Parameters:
        binary_img (np.ndarray): Binary image (white on black).
        original_img (np.ndarray): Optional original image to draw boxes on. 
                                   If None, a color version of binary image is used.
        box_w (int): Width of scanning box.
        box_h (int): Height of scanning box.
        threshold_perc (int): % of white pixels in box needed to trigger a drawn rectangle.
        
    Returns:
        np.ndarray: Image with blue rectangles drawn.
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
    midpoint_coords =  []
    
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            continue

        output = original_image.copy()
        dst = np.zeros_like(dst)

        # Draw filled contour
        cv2.drawContours(dst, contours, i, 255, thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)
        cv2.drawContours(final_output, contours, i, (0, 255, 0), thickness=cv2.FILLED, lineType=8, hierarchy=hierarchy)

        # Get bounding box
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
                        final_output = cv2.circle(final_output, (int(x_mid), int(y_mid)),1, (0,0,255), 1)
                        midpoint_coords.append((x_mid, y_mid))
        output_images.append(output)
    return output_images, final_output, midpoint_coords
    
def main():
    #rect, color_image, depth_image = getdata.main()
    #cv2.imwrite('color_image4.jpg', color_image)
    #np.save('depth_image4.npy', depth_image)
    
    
    # Load the color image and depth image from files
    color_image = cv2.imread('color_image4.jpg')
    depth_image = np.load('depth_image4.npy')
    rect = cv2.selectROI(color_image)
    #rect = (287, 212, 397, 456)
    #rect = YOLO_boundingbox.find_bounding_box(color_image)
    print('rectange:', rect)
    color_image = cv2.rectangle(color_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 5 )

    # Apply GrabCut algorithm
    mask = grabcut.apply_grabcut(color_image, rect, iter_count=5)
    # Extract depth values using the mask
    extracted_depth = depth_extract.extract_depth(mask, depth_image)
    
    # Normalize the extracted depth image for better visualization
    #extracted_depth = cv2.normalize(extracted_depth, None, 0, 255, cv2.NORM_MINMAX)

    
    #denoise the depth image
    depth_denoised = denoise.denoise(extracted_depth)

    
    #initalise a picture called thresh to hold the picture after getting rid of all values above 950mm
    thresh = np.copy(depth_denoised)
    threshold = midpoint_depth(thresh)
    print(f"Threshold value: {threshold}")
    #set all values above 950(mm) to 0 (this can be changed when finetuning for when on the robot)
    thresh[depth_denoised > threshold] = 0
    


    contourmap = thresh.copy()
    #convert the image to a binary image
    contourmap[contourmap > 0] = 255
    contourmap[contourmap <= 0] = 0
    contourmap = contourmap.astype(np.uint8)

    individual_boxed_images, final_boxed_image, midpoint_coords  = draw_boxes_on_binary_image(contourmap, color_image, box_w=10, box_h=10, threshold_perc=30)
    print(midpoint_coords)
    
    

    #applying colormaps to all values to visualize
    #Original Depth Image
    extracted_depth_colored = cv2.normalize(extracted_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    extracted_depth_colored = cv2.applyColorMap(extracted_depth_colored, cv2.COLORMAP_JET)
    #Denoised Depth Image
    depth_denoised_colored = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_denoised_colored = cv2.applyColorMap(depth_denoised_colored, cv2.COLORMAP_JET)
    #Thresholded Depth Image
    thresh_colored = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_colored = cv2.applyColorMap(thresh_colored, cv2.COLORMAP_JET)
    


    

    # Display the images
    cv2.imshow('Color Image', color_image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Depth Image', extracted_depth_colored)
    cv2.imshow('Denoised Depth', depth_denoised_colored)
    cv2.imshow('Thresholded Depth', thresh_colored)
    cv2.imshow('Boxed Map', final_boxed_image)
    
    #wait for a key press
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
            

if __name__ == "__main__":
    main()



