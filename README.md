# Vehicle Tracking Project

Dự án nhận diện, phân vùng (segmentation) và theo dõi phương tiện giao thông (ByteTrack) kết hợp giải thuật bao lồi trực giao (Orthogonal Convex Hull).

---

## 1. Yêu cầu hệ thống
- Đã cài đặt **Python** (khuyến nghị Python 3.10 - 3.12).
- Khi cài Python trên Windows, nhớ tích chọn: **"Add python.exe to PATH"**.

---

## 2. Hướng dẫn chạy nhanh (1-Click Run cho Windows)

Người dùng khác sau khi tải source code về chỉ cần:

1. **Nhấp đúp vào file `run.bat`**:
   - Lần đầu mở: Script sẽ **tự động tạo môi trường ảo (`venv`)** và **tự động cài đặt đầy đủ thư viện** từ `requirements.txt`.
   - Các lần tiếp theo: Chạy thẳng vào chương trình trong vòng 1 giây.
2. Nhập tên file video test (ví dụ `vh.mp4`) khi chương trình yêu cầu và xem kết quả.
3. Bấm phím **`q`** trên bàn phím để dừng/thoát video.

*(Dành cho Linux/macOS: chạy lệnh `./run.sh`)*

---

## 3. Hướng dẫn chạy thủ công bằng `venv`

Nếu muốn tự chạy bằng dòng lệnh:

```bash
# Bước 1: Tạo môi trường ảo venv
python -m venv venv

# Bước 2: Kích hoạt môi trường ảo
# Trên Windows (Command Prompt hoặc PowerShell):
.\venv\Scripts\activate

# Bước 3: Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt

# Bước 4: Chạy chương trình
python yolo_detect.py
```

---

## 4. Dữ liệu Video Test & Weights
- **Model weights**: `bestseg.pt`, `yolo11n.pt`
- **Video test**: Do giới hạn dung lượng GitHub, các video test được lưu trữ ngoài:
  - Thư mục Google Drive: [Tải video tại đây](https://drive.google.com/drive/u/0/folders/1jXRbfL7takDqWV_jUsSjmmFXj998NloE)
  - Bao gồm: `vh.mp4`, `vh1.mp4`, `tht.mp4`
  - *Lưu ý: Tải các file video và đặt vào cùng thư mục với `yolo_detect.py` trước khi chạy.*
