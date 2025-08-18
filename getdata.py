import pyrealsense2 as rs
import cv2
import numpy as np
import os
import random


def initialize_camera():
    """
    Initialize the RealSense camera for live streaming.
    
    Returns:
        tuple: (pipeline, profile) or (None, None) if initialization fails
    """
    try:
        # Configure pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Configure depth sensor
        depth_sensor = profile.get_device().first_depth_sensor()
        try:
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
            print("Visual preset set to High Density")
        except Exception as e:
            print(f"Failed to set visual preset: {e}")
        
        print("Camera initialized successfully")
        return pipeline, profile
        
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None, None


def get_live_frame(pipeline):
    """
    Get a single frame from the live camera stream.
    
    Args:
        pipeline: RealSense pipeline object
        
    Returns:
        tuple: (color_image, depth_image) or (None, None) if failed
    """
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
        
    except Exception as e:
        print(f"Failed to get frame: {e}")
        return None, None


def cleanup_camera(pipeline):
    """
    Clean up camera resources.
    
    Args:
        pipeline: RealSense pipeline object
    """
    try:
        pipeline.stop()
        print("Camera cleaned up successfully")
    except Exception as e:
        print(f"Error during camera cleanup: {e}")


def main():
    """
    Main function for live camera capture with ROI selection.
    
    Returns:
        tuple: (color_image, depth_image) or None if failed
    """
    # Initialize camera
    pipeline, profile = initialize_camera()
    if pipeline is None:
        return None
    
    try:
        # Get a single frame
        color_image, depth_image = get_live_frame(pipeline)
        if color_image is None or depth_image is None:
            print("No frames received. Exiting...")
            return None
            
        return color_image, depth_image
        
    finally:
        cleanup_camera(pipeline)
        cv2.destroy_all_windows()


def get_random_image():
    """
    Load a random image pair from the dataset.
    
    Returns:
        tuple: (color_image, depth_image, image_name) for a valid image pair
        
    Raises:
        ValueError: If no valid image pairs are found in the dataset
    """
    # Define dataset paths
    dataset_path = 'Test_dataset'
    color_dir = os.path.join(dataset_path, 'color_images')
    depth_dir = os.path.join(dataset_path, 'depth_images')
    
    # Get list of available images
    images = [f for f in os.listdir(color_dir) if f.endswith('.jpg')]
    if not images:
        raise ValueError("No images found in the color_images directory.")
    
    # Try to find a valid image pair
    while images:
        random_image = random.choice(images)
        color_image_path = os.path.join(color_dir, random_image)
        depth_image_path = os.path.join(depth_dir, random_image[:-4] + '.npy')
        
        # Load color image
        color_image = cv2.imread(color_image_path)
        if color_image is None:
            images.remove(random_image)
            continue
        
        # Check if depth file exists
        if not os.path.exists(depth_image_path):
            images.remove(random_image)
            continue
        
        # Load depth image
        depth_image = np.load(depth_image_path)
        print(f'Selected image: {random_image}')
        
        return color_image, depth_image, random_image[:-4]
    
    raise ValueError("No valid image and depth pairs found in the dataset directory.")


def get_all_images(dataset_path):
    """
    Generator that yields all valid image pairs sequentially.
    
    Yields:
        tuple: (color_image, depth_image, image_name) for each valid image pair
        
    Raises:
        ValueError: If dataset directories are not found or no images exist
    """
    # Define dataset paths
    color_dir = os.path.join(dataset_path, 'color_images')
    depth_dir = os.path.join(dataset_path, 'depth_images')
    
    # Validate dataset directories
    if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
        raise ValueError(f"Dataset directories not found: {color_dir} or {depth_dir}")
    
    # Get sorted list of images
    images = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    if not images:
        raise ValueError("No images found in the color_images directory.")
    
    # Process each image
    valid_count = 0
    for image_name in images:
        color_image_path = os.path.join(color_dir, image_name)
        depth_image_path = os.path.join(depth_dir, image_name[:-4] + '.npy')
        
        # Check if corresponding depth file exists
        if not os.path.exists(depth_image_path):
            print(f"Skipping {image_name}: No corresponding depth file")
            continue
            
        # Load color image
        color_image = cv2.imread(color_image_path)
        if color_image is None:
            print(f"Skipping {image_name}: Could not load color image")
            continue
            
        # Load depth image
        try:
            depth_image = np.load(depth_image_path)
        except Exception as e:
            print(f"Skipping {image_name}: Could not load depth image - {e}")
            continue
        
        # Successfully loaded both images
        valid_count += 1
        yield color_image, depth_image, image_name[:-4]

def get_frames_from_bag_files(bag_folder_path):
    """
    Generator that yields frames from all bag files in the specified folder.
    
    Args:
        bag_folder_path (str): Path to the folder containing bag files
        
    Yields:
        tuple: (color_image, depth_image, image_name) for each frame in the bag files
        
    Raises:
        ValueError: If no bag files are found in the specified folder
    """
    pipeline = rs.pipeline()
    config = rs.config()
    # Validate bag folder
    if not os.path.exists(bag_folder_path):
        raise ValueError(f"Bag folder not found: {bag_folder_path}")
    
    # Get list of bag files
    bag_files = [f for f in os.listdir(bag_folder_path) if f.endswith('.bag')]
    if not bag_files:
        raise ValueError("No bag files found in the specified folder.")
    
    # Process each bag file
    for bag_file in sorted(bag_files):
        print(f"Processing bag file: {bag_file}")
        
        # Create full path to bag file
        bag_file_path = os.path.join(bag_folder_path, bag_file)
        
        # Enable playback from file
        config.enable_device_from_file(bag_file_path, repeat_playback=False)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get playback device
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)  # Allow manual control of playback speed
        try:
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    break  # No more frames
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                #convert BGR to RGB
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Create image name based on bag file and frame number
                frame_number = depth_frame.frame_number
                image_name = f"{bag_file[:-4]}_frame{frame_number}.jpg"
                
                yield color_image, depth_image, image_name
                
        except Exception as e:
            print(f"Error processing bag file {bag_file}: {e}")
        
        finally:
            pipeline.stop()


if __name__ == "__main__":
    main()
