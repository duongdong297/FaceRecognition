import gradio as gr
import cv2
import yaml
import numpy as np
import time
import os
from datetime import datetime
from modelling.face_recognizer import FaceRecognizerSystem
from database.face_database import FaceDatabase
from database.csv_logger import CSVLogger 

# Load Config
try:
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy config/config.yaml.")
    exit()

# Khởi tạo modules
SYSTEM = FaceRecognizerSystem(cfg)
DB = FaceDatabase(cfg['paths']['database'])
CAPTURE_DIR = "database/capture_logs/"
os.makedirs(CAPTURE_DIR, exist_ok=True)
LOGGER = CSVLogger(cfg['paths'].get('log_csv', 'database/access_log.csv'))

# Biến global cho Cooldown
WEB_COOLDOWN_DICT = {} 
WEB_COOLDOWN_TIME_SECONDS = 10.0
CAPTURE_SCORE_THRESHOLD = 0.85

# ====================================================================
# 2. HÀM CORE LOGIC CHO GRADIO (Xử lý ảnh đầu vào)
# ====================================================================

def live_recognition_web(image_input_rgb: np.ndarray) -> tuple[np.ndarray, str, str]:
    """
    Thực hiện nhận diện trên ảnh (NumPy array RGB) từ Gradio.
    """
    global WEB_COOLDOWN_DICT
    
    # Gradio trả về ảnh là NumPy array ở định dạng RGB
    # OpenCV (hệ thống nhận diện) cần định dạng BGR
    frame_bgr = cv2.cvtColor(image_input_rgb, cv2.COLOR_RGB2BGR)

    # 1.(Detector + Recognizer)
    results = SYSTEM.recognize_image(frame_bgr, DB)
    
    # 2. Xử lý kết quả (Vẽ và Ghi log)
    annotated_frame = frame_bgr
    log_status_text = "Status: Idle"
    current_time = time.time()
    
    # Vẽ kết quả lên ảnh BGR
    for res in results:
        name = res['name']
        score = res['score']
        
        # Vẽ box
        x1, y1, x2, y2 = res['box']
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # BGR colors
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{name} ({score:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Logic Ghi Log (Cooldown theo session)
        if name != "Unknown":
            time_since_last_log = current_time - WEB_COOLDOWN_DICT.get(name, 0)
            
            if time_since_last_log > WEB_COOLDOWN_TIME_SECONDS:
                #Auto Capture
                """
                if score >= CAPTURE_SCORE_THRESHOLD:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(CAPTURE_DIR, f"{name}_PASS_{timestamp}.jpg")
                    cv2.imwrite(filename, annotated_frame)
                    log_status_text = f"PASS & Captured: {name} ({score:.2f})"
                """
                # Ghi log
                LOGGER.log_access(name)
                WEB_COOLDOWN_DICT[name] = current_time # Cập nhật thời điểm log
                log_status_text = f"PASS! Welcome, {name}."
            else:
                log_status_text = f"Recognized {name}. Cooldown active."
    
    # Chuyển BGR trở lại RGB cho Gradio hiển thị
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, log_status_text


# ====================================================================
# 3. ĐỊNH NGHĨA GIAO DIỆN GRADIO
# ====================================================================
#Live
""" custom_css = "footer {visibility: hidden}"

with gr.Blocks(title="Face Attendance") as demo:
    gr.Markdown("#Face Recognition Attendance Demo (YOLO + ArcFace)")
    gr.Markdown("Ứng dụng nhận diện khuôn mặt sử dụng ArcFace và YOLO")
    
    with gr.Row():
        with gr.Column():
            input_camera = gr.Image(
                label="Camera Input", 
                sources=["webcam"], 
                streaming=True, 
                type="numpy"
            )
        
        
        with gr.Column():
            output_image = gr.Image(label="Output", type="numpy")
            output_status = gr.Textbox(label="System Status")

    input_camera.stream(
        fn=live_recognition_web, 
        inputs=input_camera, 
        outputs=[output_image, output_status],
    )

if __name__ == "__main__":
    demo.launch() """


#Capture
# Khai báo các component
live_input = gr.Image(sources=["webcam"], type="numpy", label="Camera Feed (Webcam)")
output_img = gr.Image(label="Face Recognition Output", type="numpy")
output_text = gr.Textbox(label="System Status / Last Log")

# Tạo Interface
iface = gr.Interface(
    fn=live_recognition_web,
    inputs=live_input,
    outputs=[output_img, output_text],
    title="Face Recognition Attendance Demo (YOLO + ArcFace)",
    description="Ứng dụng nhận diện khuôn mặt sử dụng  ArcFace và YOLO."
)

if __name__ == "__main__":
    iface.launch()