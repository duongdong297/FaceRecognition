import cv2
import numpy as np
from skimage import transform as trans

def simple_crop_face(img, box, image_size=(112, 112)):
    """
    Crop và Resize đơn giản khi chỉ có Bounding Box
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # Mở rộng box một chút để lấy hết khuôn mặt (Padding)
    pad_w = (x2 - x1) * 0.1
    pad_h = (y2 - y1) * 0.1
    
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(w, int(x2 + pad_w))
    y2 = min(h, int(y2 + pad_h))
    
    face_img = img[y1:y2, x1:x2]
    
    # Resize về 112x112
    if face_img.size != 0:
        face_img = cv2.resize(face_img, image_size)
        
    return face_img