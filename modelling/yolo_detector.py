from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path):
        print(f"Loading YOLO detector from {model_path}")
        try:
            self.model = YOLO(model_path)
        except:
            print("Custom YOLO not found")
            self.model = YOLO('yolov11n.pt')

    def detect(self, image, conf=0.5):
        """
        Trả về list các bounding box [x1, y1, x2, y2]
        """
        results = self.model(image, verbose=False, conf=conf)
        boxes = []
        for result in results:
            for box in result.boxes:
                coords = list(map(int, box.xyxy[0]))
                boxes.append(coords)
        return boxes