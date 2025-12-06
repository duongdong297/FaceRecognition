# database/csv_logger.py
import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.fieldnames = ['STT', 'Ten', 'ThoiGian']
        self._initialize_csv()

    def _initialize_csv(self):
        """Tạo file và header nếu chưa tồn tại"""
        if not os.path.exists(self.log_file_path):
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            with open(self.log_file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_access(self, name):
        """Ghi một dòng log vào CSV"""
        try:
            # 1. Tính STT (đếm số dòng hiện tại)
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Trừ 1 cho dòng header, nếu file rỗng thì stt=1
                row_count = sum(1 for row in f) 
                stt = row_count if row_count > 0 else 1

            # 2. Lấy thời gian
            now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            # 3. Ghi file
            with open(self.log_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow({'STT': stt, 'Ten': name, 'ThoiGian': now})
            
            print(f"[LOG SAVED] {name} at {now}")
            
        except Exception as e:
            print(f"Error logging to CSV: {e}")