# Vehicle Tracking Project

Dự án nhận diện, phân vùng (segmentation) và theo dõi phương tiện giao thông (ByteTrack) kết hợp giải thuật bao lồi trực giao (Orthogonal Convex Hull).

---

## 1. Yêu cầu hệ thống
- Đã cài đặt **Python** (khuyến nghị Python 3.10 - 3.12 hoặc Anaconda).
- Hỗ trợ cả **CPU** và **GPU NVIDIA** (CUDA 12.8+, tương thích RTX series kể cả RTX 40/50 series).

---

## 2. Hướng dẫn chạy nhanh (cho Windows/Mac/Linux)

Người dùng sau khi tải hoặc clone source code về chỉ cần:

1. **Nhấp đúp chuột vào file `run.bat`**:
   - **Nếu máy đã có sẵn thư viện** (như máy đang dùng Anaconda): Script sẽ nhận diện và mở chương trình ngay lập tức.
   - **Nếu máy mới chưa có thư viện**: Script sẽ **tự động tạo môi trường ảo (`venv`)** và **tự động cài đặt đầy đủ thư viện hỗ trợ GPU/CPU** từ `requirements.txt` trong lần chạy đầu tiên.
2. Khi terminal hỏi `Nhập tên file video (vd: test.mp4):`, nhập tên file video (ví dụ: `vh.mp4`) và nhấn **Enter**.
3. Cửa sổ video nhận diện sẽ xuất hiện. Bấm phím **`q`** trên bàn phím để dừng/thoát video.

*(Dành cho Linux / macOS: mở terminal và chạy lệnh `./run.sh`)*

---

## 3. Hướng dẫn chạy thủ công bằng dòng lệnh (`venv`)

Nếu bạn muốn tự thao tác từng bước bằng dòng lệnh:

```bash
# Bước 0: Clone project
git clone https://github.com/Pdinh206/Vehicle-Detection.git; cd Vehicle-Detection

# Bước 1: Tạo môi trường ảo venv
python -m venv venv

# Bước 2: Kích hoạt môi trường ảo
# Trên Windows (CMD hoặc PowerShell):
.\venv\Scripts\activate

# Bước 3: Cài đặt các thư viện cần thiết (bao gồm PyTorch CUDA 12.8)
python -m pip install --upgrade pip
pip install -r requirements.txt

# Bước 4: Chạy chương trình
python yolo_detect.py
```

Khi không dùng nữa, gõ `deactivate` để thoát môi trường ảo.

---

## 4. Tùy chọn thiết bị chạy (GPU hoặc CPU)

Trong file `yolo_detect.py`, bạn có thể dễ dàng chuyển đổi thiết bị tại dòng cấu hình `DEVICE`:

- **Chạy bằng GPU (NVIDIA)**:
  ```python
  DEVICE = 0
  ```
- **Chạy bằng CPU**:
  ```python
  DEVICE = "cpu"
  ```

> **Lưu ý**: Hệ thống đã tích hợp cơ chế bảo vệ an toàn. Nếu bạn cấu hình `DEVICE = 0` nhưng máy chưa có GPU phù hợp hoặc driver CUDA gặp lỗi, chương trình sẽ tự động chuyển về chạy trên CPU (`device='cpu'`) để đảm bảo không bị văng hay ngắt chương trình.

---

## 5. Dữ liệu Video Test & Weights
- **Model weights**: `bestseg.pt`, `yolo11n.pt`
- **Video test**: `vh.mp4`, `vh1.mp4`, `tht.mp4`
