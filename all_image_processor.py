import getdata
import saving
from image_data import ImageData



def process_all_images(dataset_path, process_single_image_func):
    """Process all images sequentially without interactive display."""
    print("\n=== Processing All Images Sequentially ===")
    
    successful_count = 0
    failed_count = 0
    failed_images = []
    
    
    try:
        for color_image, depth_image, image_name in getdata.get_all_images(dataset_path):
            try:
                # Create ImageData object to hold results
                image_data = ImageData(color_image, depth_image, image_name)
                success, image_data = process_single_image_func(image_data)
                if success:
                    successful_count += 1
                    saving.save_combined_results(image_data)
                         
                else:
                    failed_count += 1
                    failed_images.append(f"{image_name} - Vial detection failed")
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                failed_count += 1
                failed_images.append(f"{image_name} - Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        return
    
    saving.create_failed_images_log(failed_images)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total attempted: {successful_count + failed_count} images")
    
    if failed_images:
        print(f"Failed images logged to: results/failed_images.txt")