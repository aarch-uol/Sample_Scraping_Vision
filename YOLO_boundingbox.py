from ultralytics import YOLO
model = YOLO("best.pt")

def find_bounding_box(image):
    results = model.predict(image)
    result = results[0]
    len(result.boxes)
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        if class_id == 'vial' or class_id == 'vial_with_crystal' or class_id == 'empty_vial':
            print("Probability:", conf)
            rect = (cords[0], cords[1], cords[2], cords[3])
    return rect


