import os
import time
import tempfile
from pathlib import Path
from itertools import chain

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import gradio as gr

from yolo_detect import findOrthogonalConvexHull, inside

# ==============================================================================
# CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
MODEL_PATH = "bestseg.pt"
CLASSES = [2, 3, 5, 7]  # car: 2, motorcycle: 3, bus: 5, truck: 7
CLASS_NAMES_VI = {
    2: "Ô tô",
    3: "Xe máy",
    5: "Xe buýt",
    7: "Xe tải",
}

COLOR_MAP = {
    2: (255, 255, 0),    # car: cyan
    3: (255, 255, 255),  # motorcycle: white
    5: (0, 0, 255),      # bus: red
    7: (0, 255, 255),    # truck: yellow
}

TRACKER_CONFIG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.20,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.20,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
}

# Khởi tạo tracker YAML
TRACKER_PATH = "my_bytetrack.yaml"
with open(TRACKER_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(TRACKER_CONFIG, f, sort_keys=False)

# Tải trước model
print(f"Đang tải model {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Tải model thành công!")


# ==============================================================================
# HÀM XỬ LÝ VIDEO STREAMING REAL-TIME
# ==============================================================================
def process_video_stream(
    video_path,
    conf_thresh=0.05,
    imgsz=960,
    draw_orthogonal_hull=True,
    device_choice="cpu",
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("Vui lòng chọn hoặc tải lên một file video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Không thể mở file video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or not np.isfinite(fps):
        fps = 25.0

    # Chuẩn bị file output
    out_dir = Path(tempfile.gettempdir()) / "vehicle_detection_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(out_dir / f"output_{Path(video_path).stem}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Xử lý thiết bị (GPU / CPU)
    actual_device = "0" if "0" in str(device_choice) else "cpu"

    try:
        results = model.track(
            source=video_path,
            classes=CLASSES,
            conf=float(conf_thresh),
            imgsz=int(imgsz),
            max_det=300,
            device=actual_device,
            persist=True,
            stream=True,
            tracker=TRACKER_PATH,
            verbose=False,
        )
        results_iter = iter(results)
        first_result = next(results_iter, None)
    except Exception as e:
        # Tự động fallback về CPU nếu GPU không khả dụng
        actual_device = "cpu"
        results = model.track(
            source=video_path,
            classes=CLASSES,
            conf=float(conf_thresh),
            imgsz=int(imgsz),
            max_det=300,
            device="cpu",
            persist=True,
            stream=True,
            tracker=TRACKER_PATH,
            verbose=False,
        )
        results_iter = iter(results)
        first_result = next(results_iter, None)

    if first_result is None:
        raise gr.Error("Không đọc được dữ liệu từ video.")

    stream_results = chain([first_result], results_iter)

    frame_idx = 0
    start_time = time.time()
    tracked_unique_ids = set()

    for result in stream_results:
        frame_idx += 1
        frame = result.orig_img.copy()
        current_counts = {cid: 0 for cid in CLASSES}
        boxes = result.boxes
        polygons = result.masks.xy if result.masks is not None else []

        if boxes is not None:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                current_counts[class_id] = current_counts.get(class_id, 0) + 1
                color = COLOR_MAP.get(class_id, (0, 255, 255))
                confidence = float(box.conf[0].item())

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    tracked_unique_ids.add(track_id)

                class_name = CLASS_NAMES_VI.get(class_id, model.names[class_id])
                label = f"{class_name} {confidence:.2f}"
                if track_id is not None:
                    label = f"#{track_id} {label}"

                top_y = int(box.xyxy[0][1].item())
                if i < len(polygons):
                    polygon = np.asarray(polygons[i], dtype=np.int32)
                    if len(polygon) >= 3:
                        if draw_orthogonal_hull and len(polygon) >= 4:
                            points = list(map(tuple, polygon.tolist()))
                            hull = findOrthogonalConvexHull(points)
                            if len(hull) >= 4:
                                polygon = np.asarray(hull, dtype=np.int32)
                        cv2.polylines(
                            frame,
                            [polygon.reshape(-1, 1, 2)],
                            isClosed=True,
                            color=color,
                            thickness=2,
                        )
                        top_y = int(polygon[:, 1].min())
                    else:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                x1 = max(0, int(box.xyxy[0][0].item()))
                cv2.putText(
                    frame,
                    label,
                    (x1, max(15, top_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Thanh thông tin thống kê trên khung hình
        summary_text = (
            f"O to: {current_counts.get(2, 0)} | Xe may: {current_counts.get(3, 0)} | "
            f"Xe buyt: {current_counts.get(5, 0)} | Xe tai: {current_counts.get(7, 0)}"
        )
        cv2.putText(
            frame,
            summary_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        # Tính FPS xử lý
        elapsed = time.time() - start_time
        curr_fps = frame_idx / elapsed if elapsed > 0 else 0

        # Cứ mỗi 2 frame xuất hình ảnh trực tiếp lên trình duyệt (stream live)
        if frame_idx % 2 == 0 or frame_idx == 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stats_markdown = (
                f"### 📊 Thống kê thời gian thực (Frame {frame_idx}/{total_frames if total_frames > 0 else '?'})\n"
                f"- **Tốc độ xử lý:** `{curr_fps:.1f} FPS` | **Thiết bị:** `{actual_device.upper()}`\n"
                f"- **Ô tô (Cars):** `{current_counts.get(2, 0)}`\n"
                f"- **Xe máy (Motorcycles):** `{current_counts.get(3, 0)}`\n"
                f"- **Xe buýt (Buses):** `{current_counts.get(5, 0)}`\n"
                f"- **Xe tải (Trucks):** `{current_counts.get(7, 0)}`\n"
                f"- **Tổng số lượt xe đã theo dõi (ByteTrack IDs):** `{len(tracked_unique_ids)}`"
            )
            yield frame_rgb, stats_markdown, None

    writer.release()

    # Kết thúc xử lý, trả về video hoàn chỉnh
    final_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    final_stats = (
        f"### ✅ ĐÃ XỬ LÝ HOÀN TẤT ({frame_idx} frames)!\n"
        f"- **Tốc độ trung bình:** `{curr_fps:.1f} FPS`\n"
        f"- **Thiết bị sử dụng:** `{actual_device.upper()}`\n"
        f"- **Tổng phương tiện ghi nhận:** `{len(tracked_unique_ids)}`\n"
        f"- **Video đã sẵn sàng tải xuống hoặc xem bên dưới!**"
    )
    yield final_rgb, final_stats, output_video_path


# ==============================================================================
# GIAO DIỆN WEB GRADIO
# ==============================================================================
custom_css = """
#main-container { max-width: 1200px; margin: 0 auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.stat-box { background: rgba(0, 0, 0, 0.05); padding: 15px; border-radius: 8px; }
"""

with gr.Blocks(title="Vehicle Tracking & Orthogonal Convex Hull", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚗 Hệ Thống Nhận Diện & Theo Dõi Phương Tiện Giao Thông
        ### Kết hợp YOLOv11-Seg + ByteTrack + Thuật toán Bao Lồi Trực Giao (Orthogonal Convex Hull)
        *Vừa chạy xử lý bằng GPU/CPU vừa xuất trực tiếp màn hình thời gian thực lên web.*
        """,
        elem_classes="header-box",
    )

    with gr.Row():
        # Cột điều khiển bên trái
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cấu Hình Tham Số")
            input_video = gr.Video(label="Chọn hoặc Tải Video Lên (MP4/AVI)")

            # Các video mẫu có sẵn trong thư mục
            sample_videos = []
            for v in ["vh.mp4", "vh1.mp4"]:
                if Path(v).exists():
                    sample_videos.append([v])
            if sample_videos:
                gr.Examples(
                    examples=sample_videos,
                    inputs=input_video,
                    label="🎬 Video Mẫu Có Sẵn",
                )

            device_dropdown = gr.Dropdown(
                choices=["0 (NVIDIA GPU)", "cpu"],
                value="0 (NVIDIA GPU)",
                label="Thiết bị thực thi (Device)",
                info="Tự động lùi về CPU nếu máy ảo/máy tính chưa có GPU CUDA",
            )

            draw_hull_checkbox = gr.Checkbox(
                value=True,
                label="Vẽ Bao Lồi Trực Giao (Orthogonal Hull)",
                info="Bao lồi góc vuông ôm sát thân xe thay vì hộp chữ nhật",
            )

            conf_slider = gr.Slider(
                minimum=0.01,
                maximum=0.50,
                value=0.05,
                step=0.01,
                label="Ngưỡng tin cậy (Confidence Threshold)",
            )

            imgsz_slider = gr.Slider(
                minimum=480,
                maximum=1280,
                value=960,
                step=32,
                label="Kích thước ảnh inference (Image Size)",
            )

            btn_run = gr.Button("🚀 BẮT ĐẦU XỬ LÝ (LIVE STREAM)", variant="primary", size="lg")

        # Cột hiển thị màn hình trực tiếp bên phải
        with gr.Column(scale=2):
            gr.Markdown("### 📺 Màn Hình Trực Tiếp (Live Real-Time View)")
            live_image = gr.Image(label="Live Frame Stream", elem_classes="stat-box")
            stats_output = gr.Markdown("### 📊 Thống kê sẽ hiển thị ở đây khi video bắt đầu chạy...")
            final_video_output = gr.Video(label="📥 Video Kết Quả Hoàn Chỉnh (Tải xuống)")

    btn_run.click(
        fn=process_video_stream,
        inputs=[
            input_video,
            conf_slider,
            imgsz_slider,
            draw_hull_checkbox,
            device_dropdown,
        ],
        outputs=[
            live_image,
            stats_output,
            final_video_output,
        ],
    )

if __name__ == "__main__":
    # share=True để tự động tạo link public (rất tiện khi chạy trên Google Colab / Kaggle)
    demo.queue().launch(share=True, inbrowser=True)
