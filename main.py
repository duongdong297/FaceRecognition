import time
from datetime import date
import cv2
import yaml
import torch
from PIL import Image
import torch.nn.functional as F

from modelling.face_recognizer import FaceRecognizerSystem
from database.face_database import FaceDatabase
from utils.preprocessing import preprocess_face
from database.csv_logger import CSVLogger

# 1. Load Config
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] != 'cpu' else "cpu")

def main():
    # 2. Init Modules
    system = FaceRecognizerSystem(cfg)
    db = FaceDatabase(cfg['paths']['database'])
    logger = CSVLogger(cfg['paths']['log_csv'])

    last_logged_date = {}
    # 3. Open Camera
    cap = cv2.VideoCapture(0)
    print("System Started. Press 'q' to quit, 'r' to register new face")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detect Faces
        results = system.recognize_image(frame, db)
        current_date = date.today()

        for res in results:
            x1, y1, x2, y2 = res['box']
            name = res['name']
            score = res['score']

            # Vẽ box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            

            # Ghi Log (Nếu nhận diện được)
            if name != "Unknown":
                text_label = f"{name} ({score:.2f})"
                color = (0, 255, 0)
                cv2.putText(frame, "PASS", (50, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            else:
                text_label = f"Unknown ({score:.2f})"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            

            if name != "Unknown":
                last_log_day = last_logged_date.get(name, date.min)

                if current_date > last_log_day:
                    # 1. Ghi log vào file CSV
                    logger.log_access(name)
                    
                    # 2. Cập nhật ngày điểm danh cuối cùng cho người này
                    last_logged_date[name] = current_date 
                    
                    print(f"{name} đã điểm danh thành công vào ngày {current_date}.")
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Face Recognition System', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()