import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transform(image_size=(112, 112)):
    """Trả về transform pipeline giống lúc training"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def preprocess_face(face_img_pil, device):
    """Chuẩn bị ảnh crop để đưa vào model"""
    transform = get_transform()
    tensor = transform(face_img_pil).unsqueeze(0) # Add batch dim
    return tensor.to(device)