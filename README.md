# FaceRecognition
# Hệ thống Điểm danh & Nhận diện Khuôn mặt (Face Recognition Attendance)

Hệ thống điểm danh thông minh thời gian thực (Real-time) sử dụng **Deep Learning**. Dự án kết hợp tốc độ của **YOLO** để phát hiện khuôn mặt và độ chính xác của **ArcFace (MobileFaceNet)** để nhận diện danh tính.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLO](https://img.shields.io/badge/Model-YOLOv11%2F11-green)
![Gradio](https://img.shields.io/badge/Web-Gradio_Streaming-yellow)

## ✨ Tính Năng

* 🚀 **Real-time Performance:** Tốc độ nhận diện cực nhanh trên đa dạng phần cứng.
* 🔍 **Dual-Model Architecture:**
    * **Detector:** YOLOv11n-face (hoặc YOLOv8n-face) để bắt khuôn mặt ở nhiều góc độ.
    * **Recognizer:** Arcface trích xuất đặc trưng 512 chiều.
* 📝 **Smart Attendance (Điểm danh thông minh):**
    * Tự động ghi log vào file Excel/CSV (`STT`, `Tên`, `Thời gian`).
    * **Cooldown:** Ngăn chặn spam log (mỗi người chỉ điểm danh 1 lần/ngày hoặc theo thời gian cài đặt).
* 📸 **Auto-Capture:** Tự động chụp và lưu ảnh khuôn mặt khi độ tin cậy (score) vượt ngưỡng hệ thống.
* 🌐 **Multi-Platform:**
    * **Desktop App:** Chạy OpenCV Windows thông qua Webcam hay các thiết bị kết nối với máy tính.
    * **Web App:** Giao diện Web (Gradio) hỗ trợ Capture, Live Streaming qua mạng LAN.

## 📂 Cấu Trúc Dự Án

```text
FaceRecognitio
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
├── utils/
│   ├── register_face.py        # Script đăng ký khuôn mặt mới
│   └── delete_face.py          # Script xóa người dùng
├── weights/                    # Chứa các trọng số mô hình
├── main.py                     # Ứng dụng Desktop (OpenCV)
├── web_app_demo.py             # Ứng dụng Web (Gradio Live)
└── requirements.txt            # Các thư viện cần thiết
```
 ## ⚙️ Cài Đặt

### 1. Yêu cầu hệ thống
* **Python:**>= 3.8
* **Webcam:** Cần kết nối Webcam USB hoặc Camera tích hợp sẵn trên laptop.
### 2. Clone dự án
```bash
git clone https://github.com/duongdong297/FaceRecognition.git
```
### 3. Cài đặt thư viện
Tại thư mục gốc của dự án, mở Terminal và chạy lệnh sau để cài đặt toàn bộ các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```
### 4. Chuẩn bị Model
Tìm trong thư mục ```text /weights``` và điều chỉnh đường dẫn trong ```text config/config.yaml```

## 🚀 Hướng dẫn sử dụng
### 1. Đăng ký khuôn mặt (Lưu Features Vectors, ảnh,...)
Sử dụng lệnh sau:
```bash
python utils/register_face.py "Ten_Nguoi_Dung"
```
### 2. Chạy chương trình
```bash
python main.py
```
* Box xanh lá: Nhận diện người đã đăng kí
* Nhận diện thành công sẽ ghi log vào: ```text database/access_log.csv```
### 3. Chạy trên giao diện web app
```bash
python web_app_demo.py
```
* Truy cập IP hiện trên Terminal
* Bấm nút chụp ảnh để lưu lại và chờ xử lý
* Ghi lại kết quả sau khi xử lý vào ```text database/access_log.csv```

## 🔧 Cấu Hình Hệ Thống
Bạn có thể tùy chỉnh các tham số trong file ```text config/config.yaml:```
```YAML
system:
  device: "cuda"
  image_size: [112, 112]

model:
  yolo_path: "weights/yolov11n-face.pt"  
  arcface_path: "weights/arcface_best_v2.pth"
  threshold: 0.305 

paths:
  database: "database/embeddings/face_db.json"
  raw_images: "database/raw_images" 
  log_csv: 'database/access_log.csv'
```
### Kết luận
Đây là DEMO.
