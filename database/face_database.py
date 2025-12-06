import json
import os
import numpy as np
import torch
from utils.preprocessing import preprocess_face
from PIL import Image

class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.database = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                # Convert list back to numpy array for calculation
                for key in data:
                    data[key] = np.array(data[key], dtype=np.float32)
            print(f"Database loaded: {len(data)} people.")
            return data
        return {}

    def save_db(self):
        # Convert numpy array to list for JSON serialization
        serializable_db = {k: v.tolist() for k, v in self.database.items()}
        print(f"DEBUG: Attempting to save to path: {os.path.abspath(self.db_path)}")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            with open(self.db_path, 'w') as f:
                json.dump(serializable_db, f, indent=4) 
            print("Database saved.")
        except Exception as e:
            print(f"LỖI KHÔNG THỂ LƯU FILE! Lỗi: {e}")

    def add_person(self, name, embedding):
        self.database[name] = embedding
        self.save_db()

    def delete_person(self, name):
        """Xóa người dùng khỏi database dựa trên tên."""
        if name in self.database:
            del self.database[name]
            self.save_db() # Gọi hàm lưu lại DB sau khi xóa
            print(f"User '{name}' successfully deleted.")
            return True
        else:
            print(f"Error: User '{name}' not found in database.")
            return False
    def find_match(self, embedding, threshold=0.305):
        max_score = -1.0
        best_name = "Unknown"

        for name, db_emb in self.database.items():
            # Cosine Similarity
            score = np.dot(embedding, db_emb)
            if score > max_score:
                max_score = score
                best_name = name
        
        if max_score > threshold:
            return best_name, max_score
        return "Unknown", max_score