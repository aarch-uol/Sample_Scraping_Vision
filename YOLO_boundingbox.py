from ultralytics import YOLO
import math

model = YOLO("best.pt")

def find_bounding_box(image):
    results = model.predict(image)
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print("No vial detected in the image.")
        return None
    
    # Get image center
    image_height, image_width = image.shape[:2]
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    valid_boxes = []
    
    # Collect all valid boxes with class_id == '0'
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        if class_id == '0':
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            
            # Calculate box center
            box_center_x = (cords[0] + cords[2]) / 2
            box_center_y = (cords[1] + cords[3]) / 2
            
            # Calculate distance from image center
            distance = math.sqrt((box_center_x - image_center_x)**2 + (box_center_y - image_center_y)**2)
            
            valid_boxes.append({
                'coords': cords,
                'confidence': conf,
                'distance': distance,
                'center': (box_center_x, box_center_y)
            })
    
    if not valid_boxes:
        print("No vial detected in the image.")
        return None
    
    # Find the box closest to the center
    central_box = min(valid_boxes, key=lambda x: x['distance'])
    
    print(f"Selected central box - Probability: {central_box['confidence']}, Distance from center: {central_box['distance']:.1f}")
    
    rect = (central_box['coords'][0], central_box['coords'][1], 
            central_box['coords'][2], central_box['coords'][3])
    
    return rect


