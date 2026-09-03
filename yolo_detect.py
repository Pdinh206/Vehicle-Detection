import cv2
import argparse
import time
from collections import defaultdict, Counter
from ultralytics import YOLO

def create_custom_botsort_yaml():
    """Cấu hình BoT-SORT đầy đủ tham số tương thích Ultralytics mới nhất"""
    yaml_content = """tracker_type: botsort
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.20
track_buffer: 150
match_thresh: 0.50
fuse_score: True

gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
model: auto
"""
    tracker_path = "custom_botsort.yaml"
    with open(tracker_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return tracker_path

def is_image_file(file_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    return file_path.lower().endswith(valid_extensions)

def process_image(model, args, colors):
    img = cv2.imread(args.source)
    if img is None:
        print(f"Lỗi: Không thể đọc file ảnh '{args.source}'")
        return

    results = model(img, conf=args.thresh, iou=args.iou, verbose=False)
    boxes = results[0].boxes

    detected_count = 0
    if boxes is not None:
        coords = boxes.xyxy.cpu().numpy().astype(int)
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        detected_count = len(coords)

        for (xmin, ymin, xmax, ymax), cls_id, conf in zip(coords, classes, confs):
            cls_name = model.names.get(cls_id, f"cls_{cls_id}")
            color = colors[cls_id % len(colors)]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{cls_name} {int(conf * 100)}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 8)

            cv2.rectangle(
                img, 
                (xmin, label_ymin - label_size[1] - 6),
                (xmin + label_size[0], label_ymin + 4), 
                color, 
                cv2.FILLED
            )
            cv2.putText(
                img, 
                label, 
                (xmin, label_ymin - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )

    print(f"✓ Đã phát hiện {detected_count} phương tiện trong ảnh.")

    if args.save:
        cv2.imwrite("output.jpg", img)
        print("✓ Đã lưu ảnh kết quả: 'output.jpg'")

    cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Detection", 1024, 576)
    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, args, colors):
    tracker_config = create_custom_botsort_yaml()
    
    source_input = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video '{args.source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        print("✓ Đang ghi video kết quả: 'output.mp4'")

    cv2.namedWindow("Vehicle Tracking (BoT-SORT)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Tracking (BoT-SORT)", 1024, 576)

    id_class_history = defaultdict(list)

    while cap.isOpened():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, 
            conf=0.05,
            iou=args.iou,
            persist=True, 
            imgsz=1280,
            tracker=tracker_config,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                if box.id is None:
                    continue

                conf = float(box.conf[0].item())
                if conf < args.thresh:
                    continue

                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0].item())
                raw_cls_id = int(box.cls[0].item())

                id_class_history[track_id].append(raw_cls_id)
                if len(id_class_history[track_id]) > 30:
                    id_class_history[track_id].pop(0)

                stable_cls_id = Counter(id_class_history[track_id]).most_common(1)[0][0]
                cls_name = model.names.get(stable_cls_id, f"cls_{stable_cls_id}")
                color = colors[stable_cls_id % len(colors)]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                label = f"#{track_id} {cls_name} {int(conf * 100)}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 8)

                cv2.rectangle(
                    frame, 
                    (xmin, label_ymin - label_size[1] - 6),
                    (xmin + label_size[0], label_ymin + 4), 
                    color, 
                    cv2.FILLED
                )
                cv2.putText(
                    frame, 
                    label, 
                    (xmin, label_ymin - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )

        proc_fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)
        cv2.putText(frame, f"FPS: {proc_fps:0.1f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if out:
            out.write(frame)

        cv2.imshow("Vehicle Tracking (BoT-SORT)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLO BoT-SORT Tracking")
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn model .pt')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn video / ảnh')
    parser.add_argument('--thresh', type=float, default=0.28, help='Ngưỡng confidence hiển thị')
    parser.add_argument('--iou', type=float, default=0.6, help='Ngưỡng NMS IoU')
    parser.add_argument('--save', action='store_true', help='Lưu file kết quả')
    args = parser.parse_args()

    model = YOLO(args.model)
    colors = [
        (68, 148, 228),   # 0: Car
        (164, 120, 87),   # 1: Motorbike
        (178, 182, 133),  # 2: Truck
        (93, 97, 209),    # 3: Bus
        (88, 159, 106)
    ]

    if is_image_file(args.source):
        process_image(model, args, colors)
    else:
        process_video(model, args, colors)

if __name__ == "__main__":
    main()