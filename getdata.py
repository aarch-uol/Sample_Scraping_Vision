import pyrealsense2 as rs
import cv2
import numpy as np
import os

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    # Set the visual preset (e.g., HIGH_ACCURACY)
    try:
        depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
        print("Visual preset set to High Density")
    except Exception as e:
        print(f"Failed to set visual preset: {e}")
    
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("No frames received. Exiting...")
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        rect = cv2.selectROI("Select ROI", color_image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")  # Close the ROI window after selection
        if rect[2] == 0 or rect[3] == 0:
            print("No ROI selected. Exiting...")
            return
        return rect, color_image, depth_image
        
    finally:
        
        pipeline.stop()
        cv2.destroyAllWindows()

def get_random_image():
    import random
    import os
    dataset_path = 'Dataset2'
    color_dir = os.path.join(dataset_path, 'color_images')
    depth_dir = os.path.join(dataset_path, 'depth_images')
    images = [f for f in os.listdir(color_dir) if f.endswith('.jpg')]
    if not images:
        raise ValueError("No images found in the color_images directory.")
    while images:
        random_image = random.choice(images)
        color_image_path = os.path.join(color_dir, random_image)
        depth_image_path = os.path.join(depth_dir, random_image[:-4] + '.npy')
        color_image = cv2.imread(color_image_path)
        if color_image is None:
            images.remove(random_image)
            continue
        if not os.path.exists(depth_image_path):
            images.remove(random_image)
            continue
        depth_image = np.load(depth_image_path)
        print('image:', random_image)
        return color_image, depth_image, random_image[:-4]
    raise ValueError("No valid image and depth pairs found in the dataset directory.")

def get_all_images():
    """Generator that yields all valid image pairs sequentially"""
    dataset_path = 'Dataset2'
    color_dir = os.path.join(dataset_path, 'color_images')
    depth_dir = os.path.join(dataset_path, 'depth_images')
    
    if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
        raise ValueError(f"Dataset directories not found: {color_dir} or {depth_dir}")
    
    images = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    if not images:
        raise ValueError("No images found in the color_images directory.")
    
    valid_count = 0
    for image_name in images:
        color_image_path = os.path.join(color_dir, image_name)
        depth_image_path = os.path.join(depth_dir, image_name[:-4] + '.npy')
        
        # Check if both files exist and are valid
        if not os.path.exists(depth_image_path):
            print(f"Skipping {image_name}: No corresponding depth file")
            continue
            
        color_image = cv2.imread(color_image_path)
        if color_image is None:
            print(f"Skipping {image_name}: Could not load color image")
            continue
            
        try:
            depth_image = np.load(depth_image_path)
        except Exception as e:
            print(f"Skipping {image_name}: Could not load depth image - {e}")
            continue
        
        valid_count += 1
        print(f"Processing image {valid_count}: {image_name}")
        yield color_image, depth_image, image_name[:-4]
