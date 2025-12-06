import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from .yolo_detector import FaceDetector
from .arcface import build_model
from utils.preprocessing import preprocess_face


class FaceRecognizerSystem:
    def __init__(self, config):
        """
        Khởi tạo hệ thống nhận diện
        args:
            config: dict load từ config.yaml
        """
        self.config = config
        self.device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Face Recognition System on {self.device}...")
        
        # 1. Load Detector (YOLO)
        self.detector = FaceDetector(config['model']['yolo_path'])
        
        # 2. Load Backbone (ArcFace/MobileFaceNet)
        self.backbone = build_model(config['model']['arcface_path'], self.device)
        
    def get_embedding(self, face_img_bgr):
        """
        Hàm phụ trợ: Lấy embedding từ một ảnh khuôn mặt đã cắt (cropped image)
        Input: Numpy array (BGR - OpenCV format)
        Output: Numpy vector (normalized)
        """
        if face_img_bgr.size == 0:
            return None
            
        # Convert BGR (OpenCV) -> RGB (PIL) -> Tensor
        face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Preprocess (Resize 112x112 + Normalize)
        tensor = preprocess_face(face_pil, self.device)
        
        # Inference
        with torch.no_grad():
            feat = self.backbone(tensor)
            # L2 Normalize (Quan trọng cho Cosine Similarity)
            feat = F.normalize(feat, p=2, dim=1).cpu().numpy()[0]
            
        return feat

    def recognize_image(self, image, database):
        """
        Pipeline đầy đủ: Detect -> Crop -> Embed -> Match
        Input: Ảnh gốc từ Camera (OpenCV Image)
        Output: List kết quả
        """
        # 1. Detect Faces
        boxes = self.detector.detect(image, conf=0.5)
        results = []
        h, w = image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box
            
            #Safe crop
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0 or (x2-x1) < 20 or (y2-y1) < 20: 
                continue

            # 2. Get Embedding
            feat = self.get_embedding(face_img)
            if feat is None: continue

            # 3. Match with Database
            name, score = database.find_match(feat, threshold=self.config['model']['threshold'])

            results.append({
                "box": [x1, y1, x2, y2],
                "name": name,
                "score": score
            })
            
        return results