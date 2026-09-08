import os
from ultralytics import YOLO

# Nạp model
model = YOLO("yolo11n_4cars.pt")

# Nhập tên file từ bàn phím
video_name = input("Nhập tên file video (vd: test.mp4): ").strip()

# Kiểm tra file có tồn tại hay không trước khi chạy
if not os.path.exists(video_name):
    print(f"Lỗi: Không tìm thấy file '{video_name}' trong thư mục này!")
else:
    print(f"Đang xử lý {video_name}...")
    results = model.predict(source=video_name, conf=0.35, device="cpu", save=True)
    print("Xử lý xong! Kết quả nằm trong thư mục 'runs/detect/predict'")
