
# Vehicle Tracking Project

## 1. Models
- yolo11n.pt: Mô hình gốc của yolo từ tập coco
- bestrbl.pt: Mô hình train từ dữ liệu của Roboflow kết hợp với train từ video thực tế. 
- bestultraly.pt: Mô hình train từ dữ liệu của ultralytic.

## 2. Dữ liệu Video Test
Do giới hạn dung lượng GitHub, các video test được lưu trữ ngoài:
- Thư mục test ảnh trên github: test model trên ảnh.
- Thư mục test video: https://drive.google.com/drive/u/0/folders/1jXRbfL7takDqWV_jUsSjmmFXj998NloE
Bao gồm các file:
- vh.mp4: Video giao thông góc rộng ban ngày.
- vh1.mp4: Video kiểm tra mật độ xe cao.
- 220480.mp4
*Lưu ý: Tải các file video, di chuyển các video trong thư mục test_video vừa tải vào cùng 1 thư mục với các model trên rồi chạy.*


## 3. Cách chạy:

# python yolo_detect.py
<!-- # Tracking bằng model rbl
yolo track model=bestrbl.pt source=vh1.mp4 show=True 
# Hoặc có thể chạy code python
python yolo_detect.py --model bestrbl.pt --source test/6.png

# Tracking bằng model ultr
yolo track model=bestultraly.pt source=vh1.mp4 show=True 
# Hoặc
python yolo_detect.py --model bestultraly.pt --source vh.mp4
-->


