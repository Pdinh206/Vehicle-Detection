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
# Cách A: Chạy cửa sổ Desktop (OpenCV)
python yolo_detect.py

# Cách B: Chạy giao diện Web (vừa chạy vừa xuất màn hình trực tiếp - khuyên dùng cho Colab/Cloud GPU)
python app.py
```

Khi không dùng nữa, gõ `deactivate` để thoát môi trường ảo.

---

## 4. Giao diện Web Trực Tiếp (Live Stream với Gradio)

Khi chạy `python app.py`:
- Trình duyệt sẽ tự động mở giao diện điều khiển hiện đại.
- **Vừa chạy vừa xuất màn hình**: Khung hình video được stream trực tiếp lên web theo thời gian thực cùng bảng thống kê số lượng phương tiện (Ô tô, Xe máy, Xe buýt, Xe tải).

---

## 5. Tùy chọn thiết bị chạy (GPU NVIDIA hoặc CPU)

Dự án hiện tại đã **hoàn toàn tự động nhận diện thiết bị** (`DEVICE = 0 if torch.cuda.is_available() else "cpu"`):
- **Máy có card rời NVIDIA (RTX/GTX)** + đã cài PyTorch CUDA: Tự động kích hoạt GPU 0 để đạt tốc độ xử lý tối đa (FPS cao).
- **Máy không có card rời (chỉ có CPU / GPU Onboard Intel/AMD)**: Tự động chạy trên CPU mà không hề bị lỗi hay văng chương trình.

### Cách xử lý khi máy có card rời nhưng vẫn báo chạy CPU:
Hiện tượng này xảy ra khi môi trường Python đang cài nhầm bản **PyTorch CPU** (`torch-x.x.x+cpu`). Để chuyển sang chạy GPU:
1. Gỡ bản PyTorch CPU cũ:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```
2. Cài bản PyTorch hỗ trợ CUDA (khuyến nghị CUDA 12.4 hoặc 12.8):
   ```bash
   # Dành cho CUDA 12.4 (ổn định nhất trên Python 3.10 - 3.12):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

   # Hoặc dành cho CUDA 12.8 / RTX 50 series:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. Kiểm tra lại:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```
   Khi kết quả hiển thị `CUDA: True` và tên card NVIDIA của bạn, dự án sẽ tự động chạy 100% bằng GPU.

---

## 6. Dữ liệu Video Test & Weights
- **Model weights**: `bestseg.pt`, `yolo11n.pt`
- **Video test**: `vh.mp4`, `vh1.mp4`, `tht.mp4`
