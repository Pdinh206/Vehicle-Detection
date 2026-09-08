import os
import torch
from ultralytics import YOLO

# ================= CẤU HÌNH INPUT =================
VIDEO_PATH = "vh1.mp4"  
MODEL_PATH = "best.pt"      
CONF_THRESHOLD = 0.35       
# ==================================================


device = 0 if torch.cuda.is_available() else "cpu"

if not os.path.exists(VIDEO_PATH):
    print(f"Lỗi: Không tìm thấy file video '{VIDEO_PATH}'!")
elif not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file mô hình '{MODEL_PATH}'!")
else:
    print(f"--> Đang chạy '{VIDEO_PATH}' trên thiết bị [{device}]")
    print("Mẹo: Nhấn phím 'q' trên cửa sổ video để tắt bất kỳ lúc nào.")
    
    model = YOLO(MODEL_PATH)

    model.track(
        source=VIDEO_PATH,
        conf=CONF_THRESHOLD,
        device=device,
        persist=True,
        show=True,
        save=False
    )
