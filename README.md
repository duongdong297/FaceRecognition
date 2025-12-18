# FaceRecognition
# 📸 Hệ thống Điểm danh & Nhận diện Khuôn mặt (Face Recognition Attendance)

Hệ thống điểm danh thông minh thời gian thực (Real-time) sử dụng **Deep Learning**. Dự án kết hợp tốc độ của **YOLO** để phát hiện khuôn mặt và độ chính xác của **ArcFace (MobileFaceNet)** để nhận diện danh tính.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLO](https://img.shields.io/badge/Model-YOLOv11%2F11-green)
![Gradio](https://img.shields.io/badge/Web-Gradio_Streaming-yellow)

## ✨ Tính Năng Nổi Bật

* 🚀 **Real-time Performance:** Tốc độ nhận diện cực nhanh trên CPU/GPU.
* 🔍 **Dual-Model Architecture:**
    * **Detector:** YOLOv11n-face (hoặc YOLOv8n-face) để bắt khuôn mặt ở nhiều góc độ.
    * **Recognizer:** MobileFaceNet (ArcFace Loss) trích xuất đặc trưng 512 chiều.
* 📝 **Smart Attendance (Điểm danh thông minh):**
    * Tự động ghi log vào file Excel/CSV (`STT`, `Tên`, `Thời gian`).
    * **Cơ chế Cooldown:** Ngăn chặn spam log (mỗi người chỉ điểm danh 1 lần/ngày hoặc theo thời gian cài đặt).
* 📸 **Auto-Capture:** Tự động chụp và lưu ảnh khuôn mặt khi độ tin cậy (score) vượt ngưỡng an toàn.
* 🌐 **Multi-Platform:**
    * **Desktop App:** Chạy cửa sổ OpenCV truyền thống.
    * **Web App:** Giao diện Web (Gradio) hỗ trợ Live Streaming qua mạng LAN.

## 📂 Cấu Trúc Dự Án

```text
FaceRecognition/
├── config/
│   └── config.yaml             # File cấu hình (ngưỡng, đường dẫn, device)
├── database/
│   ├── embeddings/             # Lưu trữ vector đặc trưng (.npy)
│   ├── raw_images/             # Ảnh gốc khi đăng ký
│   ├── capture_logs/           # Ảnh chụp tự động khi điểm danh
│   ├── access_log.csv          # File lịch sử ra/vào
│   ├── face_database.py        # Quản lý thêm/xóa/load dữ liệu
│   └── csv_logger.py           # Quản lý ghi log CSV
├── modelling/
│   ├── arcface.py              # Kiến trúc mạng MobileFaceNet
│   ├── yolo_detector.py        # Wrapper cho YOLO
│   └── face_recognizer.py      # Hệ thống chính (System Wrapper)
├── scripts/
│   ├── register_face.py        # Script đăng ký khuôn mặt mới
│   └── delete_face.py          # Script xóa người dùng
├── weights/                    # Chứa các trọng số mô hình
├── main.py                      # Ứng dụng Desktop (OpenCV)
├── web_app_demo.py             # Ứng dụng Web (Gradio Live)
└── requirements.txt            # Các thư viện cần thiết

