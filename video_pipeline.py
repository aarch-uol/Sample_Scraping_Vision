import getdata
import saving
from image_data import ImageData
import os
import csv


def process_all_bag_videos(bag_folder_path, process_single_image_func, frame_skip=1, max_frames_per_bag=None):
    """
    Process all .bag video files sequentially without interactive display.
    
    Args:
        bag_folder_path (str): Path to folder containing .bag files
        process_single_image_func (function): Function to process each frame
        frame_skip (int): Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames_per_bag (int): Maximum frames to process per bag file (None = all frames)
    """
    print(f"\n=== Processing All .bag Video Files from {bag_folder_path} ===")
    
    # Statistics tracking
    successful_frames = 0
    failed_frames = 0
    total_bags_processed = 0
    failed_frames_list = []
    
    # Validate bag folder
    if not os.path.exists(bag_folder_path):
        print(f"Error: Bag folder not found: {bag_folder_path}")
        return
    
    try:
        frame_counter = 0
        current_bag = ""
        bag_frame_count = 0
        
        for color_image, depth_image, image_name in getdata.get_frames_from_bag_files(bag_folder_path):
            try:
                # Track which bag file we're processing
                bag_name = image_name.split('_frame')[0]
                if bag_name != current_bag:
                    current_bag = bag_name
                    total_bags_processed += 1
                    bag_frame_count = 0
                    
                
                # Frame skipping logic
                if frame_counter % frame_skip != 0:
                    frame_counter += 1
                    continue
                
                # Check if we've reached max frames for this bag
                if max_frames_per_bag is not None and bag_frame_count >= max_frames_per_bag:
                    frame_counter += 1
                    continue
                
                bag_frame_count += 1
                
                # Create ImageData object and process frame
                image_data = ImageData(color_image, depth_image, image_name)
                success, processed_image_data = process_single_image_func(image_data)
                
                if success:
                    successful_frames += 1
                    
                    # Save results for this frame
                    output_path = saving.save_simplified_results(processed_image_data, results_path='video_results2')
                    
                    # Save coordinates CSV for this frame
                    if hasattr(processed_image_data, 'cropped_coords') and processed_image_data.cropped_coords is not None:
                        #save_frame_coordinates_csv(processed_image_data, 'video_results')
                        pass
                        
                else:
                    failed_frames += 1
                    
                frame_counter += 1
                
            except Exception as e:
                print(f"Error processing frame {image_name}: {e}")
                failed_frames += 1
                failed_frames_list.append(f"{image_name} - Error: {str(e)}")
                frame_counter += 1
                continue
        
        # Final bag completion message
        if current_bag:
            print(f"Completed processing {current_bag}")
                
    except Exception as e:
        print(f"Error in bag video processing: {e}")
        return
    
    # Save failed frames log
    if failed_frames_list:
        save_failed_frames_log(failed_frames_list)
    
    # Print final statistics
    print(f"\n=== Video Processing Complete ===")
    print(f"Total .bag files processed: {total_bags_processed}")
    print(f"Successfully processed frames: {successful_frames}")
    print(f"Failed to process frames: {failed_frames}")
    print(f"Total frames attempted: {successful_frames + failed_frames}")
    print(f"Frame skip rate: {frame_skip} (processing every {frame_skip} frame(s))")
    
    if max_frames_per_bag:
        print(f"Max frames per bag limit: {max_frames_per_bag}")
    
    if failed_frames_list:
        print(f"Failed frames logged to: video_results2/failed_frames.txt")

    print(f"Results saved to: video_results2/")


def process_single_bag_file(bag_file_path, process_single_image_func, frame_skip=1, max_frames=None):
    """
    Process a single .bag file.
    
    Args:
        bag_file_path (str): Path to specific .bag file
        process_single_image_func (function): Function to process each frame
        frame_skip (int): Process every Nth frame
        max_frames (int): Maximum frames to process (None = all frames)
    """
    print(f"\n=== Processing Single Bag File: {os.path.basename(bag_file_path)} ===")
    
    successful_frames = 0
    failed_frames = 0
    failed_frames_list = []
    
    try:
        frame_counter = 0
        
        # Create a temporary folder with just this bag file for compatibility
        bag_folder = os.path.dirname(bag_file_path)
        
        for color_image, depth_image, image_name in getdata.get_frames_from_bag_files(bag_folder):
            try:
                # Only process frames from the specific bag file
                if not image_name.startswith(os.path.splitext(os.path.basename(bag_file_path))[0]):
                    continue
                
                # Frame skipping logic
                if frame_counter % frame_skip != 0:
                    frame_counter += 1
                    continue
                
                # Check if we've reached max frames
                if max_frames is not None and successful_frames >= max_frames:
                    break
                
                # Process frame
                image_data = ImageData(color_image, depth_image, image_name)
                success, processed_image_data = process_single_image_func(image_data)
                
                if success:
                    successful_frames += 1
                    # Save results
                    saving.save_simplified_results(processed_image_data, results_path='video_results2')
                    #save_frame_coordinates_csv(processed_image_data, 'video_results')
                    
                else:
                    failed_frames += 1
                    failed_frames_list.append(f"{image_name} - Processing failed")
                
                frame_counter += 1
                
            except Exception as e:
                print(f"Error processing frame {image_name}: {e}")
                failed_frames += 1
                failed_frames_list.append(f"{image_name} - Error: {str(e)}")
                frame_counter += 1
                continue
                
    except Exception as e:
        print(f"Error processing bag file: {e}")
        return
    
    # Save results
    if failed_frames_list:
        save_failed_frames_log(failed_frames_list)
    
    print(f"\n=== Single Bag Processing Complete ===")
    print(f"Successfully processed: {successful_frames} frames")
    print(f"Failed to process: {failed_frames} frames")
    print(f"Total attempted: {successful_frames + failed_frames} frames")


def save_frame_coordinates_csv(image_data, results_path):
    """
    Save frame coordinates to CSV in the video results folder.
    
    Args:
        image_data (ImageData): Processed image data with coordinates
        results_path (str): Base results folder path
    """
    if not hasattr(image_data, 'cropped_coords') or image_data.cropped_coords is None:
        return
    
    # Create coordinates subfolder
    coords_folder = os.path.join(results_path, 'coordinates')
    if not os.path.exists(coords_folder):
        os.makedirs(coords_folder)
    
    # Save coordinates CSV
    csv_filename = f"{image_data.image_name}_coordinates.csv"
    csv_path = os.path.join(coords_folder, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Depth'])
            
            for coord in image_data.cropped_coords:
                writer.writerow([int(coord[0]), int(coord[1]), int(coord[2])])
                
    except Exception as e:
        print(f"Error saving coordinates CSV for {image_data.image_name}: {e}")


def save_failed_frames_log(failed_frames_list):
    """
    Save list of failed frames to a log file.
    
    Args:
        failed_frames_list (list): List of failed frame descriptions
    """
    # Create results directory if it doesn't exist
    results_dir = 'video_results2'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Write failed frames log
    log_path = os.path.join(results_dir, 'failed_frames.txt')
    try:
        with open(log_path, 'w') as f:
            f.write("Failed Frames Log\n")
            f.write("==================\n\n")
            for failed_frame in failed_frames_list:
                f.write(f"{failed_frame}\n")
        print(f"Failed frames log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving failed frames log: {e}")


def main():
    """
    Main function to run video processing with different options.
    """
    # Import the process_single_image function from main
    from main import process_single_image
    
    print("Video Processing Options:")
    print("1. Process all .bag files in folder")
    print("2. Process all .bag files (every 5th frame - faster)")
    print("3. Process all .bag files (max 50 frames per bag)")
    print("4. Process single .bag file")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    bag_folder = 'Bag_files'  # Default folder
    
    if choice == '1':
        # Process all frames from all bag files
        process_all_bag_videos(bag_folder, process_single_image)
        
    elif choice == '2':
        # Process every 5th frame for faster processing
        process_all_bag_videos(bag_folder, process_single_image, frame_skip=30)
        
    elif choice == '3':
        # Process max 50 frames per bag
        process_all_bag_videos(bag_folder, process_single_image, max_frames_per_bag=50)
        
    elif choice == '4':
        # Process single bag file
        bag_file = input("Enter bag file name (e.g., 'recording.bag'): ").strip()
        bag_path = os.path.join(bag_folder, bag_file)
        
        if os.path.exists(bag_path):
            process_single_bag_file(bag_path, process_single_image, frame_skip=1, max_frames=100)
        else:
            print(f"Bag file not found: {bag_path}")
            
    else:
        print("Invalid choice. Processing all bag files with default settings.")
        process_all_bag_videos(bag_folder, process_single_image)


if __name__ == "__main__":
    main()