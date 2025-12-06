import cv2
import sys
import os
import yaml
import torch
import numpy as np 
from datetime import datetime
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelling.face_recognizer import FaceRecognizerSystem
from database.face_database import FaceDatabase

def register_from_camera(name):
    # Load config
    with open("config/config.yaml", "r") as f: cfg = yaml.safe_load(f)
    
    # Khởi tạo Hệ thống và Database
    system = FaceRecognizerSystem(cfg)
    db = FaceDatabase(cfg['paths']['database'])
    SAVE_DIR = cfg['paths'].get('registered_images', 'database/raw_images/')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print(f"📷 Look at the camera to register: {name}. Press 's' to save or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect & Show (Dùng recognize_image chỉ để hiển thị box)
        results = system.recognize_image(frame, db)
        
        # NOTE: Hiển thị box và tên người lạ (để user biết hệ thống đang hoạt động)
        for res in results:
             x1, y1, x2, y2 = res['box']
             color = (0, 255, 0) # Màu xanh lá cây
             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Register", display_frame)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'): # Nhấn 's' để Save
            
            # 2. Chạy lại Detect để lấy bounding boxes
            raw_boxes = system.detector.detect(frame)
            
            if raw_boxes:
                # 3. Tìm box lớn nhất (Giả sử đó là khuôn mặt cần đăng ký)
                # Tính diện tích box: (x2-x1)*(y2-y1)
                largest_box = max(raw_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                x1, y1, x2, y2 = largest_box

                # 4. Crop khuôn mặt
                face_crop = frame[y1:y2, x1:x2]
                
                # 5. Lấy vector Embedding
                if face_crop.size > 0:
                    embedding = system.get_embedding(face_crop)
                    
                    # 6. Lưu vào DB
                    db.add_person(name, embedding) 

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_DIR, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(filename, face_crop)
                    print(f"Registration Success! Added '{name}' to database.")
                    break
                else:
                    print("Vùng crop không hợp lệ. Thử lại.")
            else:
                print("Không tìm thấy khuôn mặt nào. Hãy đưa mặt vào giữa khung hình.")
                
        elif key & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_face.py <name>")
    else:
        register_from_camera(sys.argv[1])