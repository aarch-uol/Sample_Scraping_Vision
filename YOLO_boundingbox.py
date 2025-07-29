from ultralytics import YOLO
import math

# Initialize YOLO model
model = YOLO("best.pt")


def find_bounding_box(image):
    """
    Find the central-most bounding box for vials detected by YOLO.
    
    Args:
        image (np.ndarray): Input image for object detection
        
    Returns:
        tuple: Bounding box coordinates (x1, y1, x2, y2) or None if no valid detection
    """
    # Run YOLO prediction
    results = model.predict(image)
    result = results[0]
    
    # Check if any boxes were detected
    if result.boxes is None or len(result.boxes) == 0:
        print("No vial detected in the image.")
        return None
    
    # Get image center for distance calculations
    image_height, image_width = image.shape[:2]
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    valid_boxes = []
    
    # Collect all valid boxes with class_id == '0'
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        
        if class_id == '0':
            # Extract box coordinates and confidence
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            
            # Calculate box center point
            box_center_x = (cords[0] + cords[2]) / 2
            box_center_y = (cords[1] + cords[3]) / 2
            
            # Calculate distance from image center
            distance = math.sqrt((box_center_x - image_center_x)**2 + (box_center_y - image_center_y)**2)
            
            # Store box information
            valid_boxes.append({
                'coords': cords,
                'confidence': conf,
                'distance': distance,
                'center': (box_center_x, box_center_y)
            })
    
    # Check if any valid boxes were found
    if not valid_boxes:
        print("No vial detected in the image.")
        return None
    
    # Find the box closest to the image center
    central_box = min(valid_boxes, key=lambda x: x['distance'])
    
    # Log selection information
    print(f"Selected central box - Probability: {central_box['confidence']}, "
          f"Distance from center: {central_box['distance']:.1f}")
    
    # Return bounding box coordinates
    rect = (central_box['coords'][0], central_box['coords'][1], 
            central_box['coords'][2], central_box['coords'][3])
    
    return rect


