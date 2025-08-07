import cv2
import os
import glob
import argparse
from pathlib import Path

def create_video_from_images(input_folder='results', output_path='results_video.mp4', fps=2.0, pattern='*_results.jpg'):
    """
    Create a video from a series of images
    
    Parameters:
        input_folder (str): Folder containing the images
        output_path (str): Output video file path
        fps (float): Frames per second for the video
        pattern (str): File pattern to match (e.g., '*_results.jpg')
    """
    
    # Get all image files matching the pattern
    image_pattern = os.path.join(input_folder, pattern)
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"No images found matching pattern: {image_pattern}")
        return False
    
    print(f"Found {len(image_files)} images")
    print(f"Creating video with {fps} FPS")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width, layers = first_image.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        # Resize image if dimensions don't match (shouldn't happen with your results)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        # Write frame to video
        video_writer.write(image)
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video created successfully: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create video from result images')
    parser.add_argument('--input', '-i', default='results', 
                       help='Input folder containing images (default: results)')
    parser.add_argument('--output', '-o', default='results_video.mp4',
                       help='Output video filename (default: results_video.mp4)')
    parser.add_argument('--fps', '-f', type=float, default=2.0,
                       help='Frames per second (default: 2.0)')
    parser.add_argument('--pattern', '-p', default='*_results.jpg',
                       help='File pattern to match (default: *_results.jpg)')
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided
    if len(os.sys.argv) == 1:
        print("=== Video Creator for Pipeline Results ===")
        
        # Get input folder
        input_folder = input(f"Enter input folder (default: results): ").strip()
        if not input_folder:
            input_folder = 'results'
        
        # Get output filename
        output_file = input(f"Enter output filename (default: results_video.mp4): ").strip()
        if not output_file:
            output_file = 'results_video.mp4'
        
        # Get FPS
        while True:
            fps_input = input(f"Enter frames per second (default: 2.0): ").strip()
            if not fps_input:
                fps = 2.0
                break
            try:
                fps = float(fps_input)
                if fps > 0:
                    break
                else:
                    print("FPS must be positive")
            except ValueError:
                print("Please enter a valid number")
        
        # Get pattern
        pattern = input(f"Enter file pattern (default: *_results.jpg): ").strip()
        if not pattern:
            pattern = '*_results.jpg'
        
        create_video_from_images(input_folder, output_file, fps, pattern)
    
    else:
        # Command line mode
        create_video_from_images(args.input, args.output, args.fps, args.pattern)

if __name__ == "__main__":
    main()