# Vehicle Tracking Project

## 1. Models
- yolo11n.pt: Mô hình gốc của yolo từ tập coco
- bestrbl.pt: Mô hình train từ dữ liệu của Roboflow kết hợp với train từ video thực tế. 
- bestultraly.pt: Mô hình train từ dữ liệu của ultralytic.

## 2. Dữ liệu Video Test
Do giới hạn dung lượng GitHub, các video test được lưu trữ ngoài:
- vh.mp4: Video giao thông góc rộng ban ngày.
- vh1.mp4: Video kiểm tra mật độ xe cao.
- 220480.mp4
- thư mục test: test model trên ảnh. (đã có sẵn)
*Lưu ý: Tải các file video, di chuyển các video từ thư mục test vừa tải vào cùng 1 thư mục với các model trên rồi chạy.*

## 3. Cách chạy:
python yolo_detect.py --model bestrbl.pt --source vh.mp4
python yolo_detect.py --model bestultraly.pt --source vh.mp4
python yolo_detect.py --model yolo11n.pt --source vh.mp4
