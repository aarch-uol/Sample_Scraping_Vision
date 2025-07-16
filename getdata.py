import pyrealsense2 as rs
import cv2
import numpy as np

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
