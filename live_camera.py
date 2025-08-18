'''this file is to run the entire pipeline on the live camera feed.'''
import cv2
import getdata
from image_data import ImageData

def run_live_pipeline(process_single_image_func):
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
            #create ImageData object to hold results
            image_data = ImageData(color_image, depth_image, f"frame_{frame_count}")
            # Process the live frame
            success, image_data = process_single_image_func(image_data)
            # Display results
            display_live_results(color_image, image_data.spatula_results_smoothed_cropped if success else None)
            print(f"Processed frame {frame_count}")
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



def display_live_results(color_image, final_image=None):
    """
    Display original and final result in separate windows for live processing.
    
    Args:
        color_image (np.ndarray): Original color image
        final_image (np.ndarray): Final result with crystal analysis (optional)
    """
    # Show original image
    cv2.imshow('Live Camera Feed', color_image)
    
    # Show final result if available, otherwise show original again
    if final_image is not None:
        cv2.imshow('Crystal Detection Result', final_image)
    else:
        cv2.imshow('Crystal Detection Result', color_image)
