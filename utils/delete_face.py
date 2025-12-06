# scripts/delete_face.py
import sys
import yaml
import os

# Cần thêm đường dẫn để import được các module khác
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.face_database import FaceDatabase

def delete_user_cli(name):
    print(f" Attempting to delete user: {name}")
    
    # Load config
    try:
        with open("config/config.yaml", "r") as f: 
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: Cannot find config/config.yaml. Run from the project root.")
        return

    db = FaceDatabase(cfg['paths']['database'])
    
    # Delete and report status
    db.delete_person(name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_face.py <name>")
        print("Example: python scripts/delete_face.py NguyenVanA")
    else:
        delete_user_cli(sys.argv[1])