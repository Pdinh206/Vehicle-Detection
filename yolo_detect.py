import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

# ==============================================================================
# 1. CÁC HÀM THUẬT TOÁN ORTHOGONAL CONVEX HULL 
# ==============================================================================

def inside(p, p1, p2):
    if p == p1 or p == p2:
        return False
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and \
           min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])

def find_o_hull1(set1, q1, qq1):
    if len(set1) == 0:
        return []
    sort_set1y = sorted(set1, key=lambda p: (-p[1], p[0]))
    new_point11 = sort_set1y[0]
    sort_set1x = sorted(set1, key=lambda p: (p[0], -p[1]))
    new_point12 = sort_set1x[0]
    new_set1 = [p for p in set1 if inside(p, new_point11, new_point12)]
    return [new_point11] + find_o_hull1(new_set1, new_point11, new_point12) + [new_point12]

def find_o_hull2(set2, q2, qq2):
    if len(set2) == 0:
        return []
    sort_set2x = sorted(set2, key=lambda p: (p[0], p[1]))
    new_point21 = sort_set2x[0]
    sort_set2y = sorted(set2, key=lambda p: (p[1], p[0]))
    new_point22 = sort_set2y[0]
    new_set2 = [p for p in set2 if inside(p, new_point21, new_point22)]
    return [new_point21] + find_o_hull2(new_set2, new_point21, new_point22) + [new_point22]

def find_o_hull3(set3, q3, qq3):
    if len(set3) == 0:
        return []
    sort_set3y = sorted(set3, key=lambda p: (p[1], -p[0]))
    new_point31 = sort_set3y[0]
    sort_set3x = sorted(set3, key=lambda p: (-p[0], p[1]))
    new_point32 = sort_set3x[0]
    new_set3 = [p for p in set3 if inside(p, new_point31, new_point32)]
    return [new_point31] + find_o_hull3(new_set3, new_point31, new_point32) + [new_point32]

def find_o_hull4(set4, q4, qq4):
    if len(set4) == 0:
        return []
    sort_set4x = sorted(set4, key=lambda p: (-p[0], -p[1]))
    new_point41 = sort_set4x[0]
    sort_set4y = sorted(set4, key=lambda p: (-p[1], -p[0]))
    new_point42 = sort_set4y[0]
    new_set4 = [p for p in set4 if inside(p, new_point41, new_point42)]
    return [new_point41] + find_o_hull4(new_set4, new_point41, new_point42) + [new_point42]

def findOrthogonalConvexHull(points):
    if len(points) < 4:
        return points

    # 1. Tìm 8 điểm mốc cực trị
    maxY = points[0][1]
    minY = points[0][1]
    maxX = points[0][0]
    minX = points[0][0]
    
    leftPoints = []
    rightPoints = []
    topPoints = []
    bottomPoints = []
    
    for point in points:
        if point[0] < minX: minX = point[0]
        if point[0] > maxX: maxX = point[0]
        if point[1] < minY: minY = point[1]
        if point[1] > maxY: maxY = point[1]

    for point in points:
        if point[0] == minX: leftPoints.append(point)
        if point[0] == maxX: rightPoints.append(point)
        if point[1] == minY: bottomPoints.append(point)
        if point[1] == maxY: topPoints.append(point)

    top = (topPoints[0],) if len(topPoints) == 1 else (sorted(topPoints, key=lambda x: x[0])[0], sorted(topPoints, key=lambda x: x[0])[-1])
    bottom = (bottomPoints[0],) if len(bottomPoints) == 1 else (sorted(bottomPoints, key=lambda x: -x[0])[0], sorted(bottomPoints, key=lambda x: -x[0])[-1])
    right = (rightPoints[0],) if len(rightPoints) == 1 else (sorted(rightPoints, key=lambda x: -x[1])[0], sorted(rightPoints, key=lambda x: -x[1])[-1])
    left = (leftPoints[0],) if len(leftPoints) == 1 else (sorted(leftPoints, key=lambda x: x[1])[0], sorted(leftPoints, key=lambda x: x[1])[-1])

    q1 = top[0]
    qq4 = top[0] if len(top) == 1 else top[1]
    q4 = right[0]
    qq3 = right[0] if len(right) == 1 else right[1]
    q3 = bottom[0]
    qq2 = bottom[0] if len(bottom) == 1 else bottom[1]
    q2 = left[0]
    qq1 = left[0] if len(left) == 1 else left[1]

    # 2. Phân chia 4 tập
    set1 = [a for a in points if inside(a, q1, qq1)]
    set2 = [a for a in points if inside(a, q2, qq2)]
    set3 = [a for a in points if inside(a, q3, qq3)]
    set4 = [a for a in points if inside(a, q4, qq4)]

    arranged_points = []
    arranged_points = arranged_points + [q1] + find_o_hull1(set1, q1, qq1) + [qq1]
    arranged_points = arranged_points + [q2] + find_o_hull2(set2, q2, qq2) + [qq2]
    arranged_points = arranged_points + [q3] + find_o_hull3(set3, q3, qq3) + [qq3]
    arranged_points = arranged_points + [q4] + find_o_hull4(set4, q4, qq4) + [qq4]

    # 3. Bẻ góc 270 độ bằng danh sách phụ S
    arranged_points.append(arranged_points[0])
    S = []
    n = len(arranged_points)

    for i in range(0, n - 1):
        if arranged_points[i+1][0] > arranged_points[i][0] and arranged_points[i+1][1] > arranged_points[i][1]:
            p3 = [arranged_points[i][0], arranged_points[i+1][1]]
            S.append([i + 1, p3])

        elif arranged_points[i+1][0] > arranged_points[i][0] and arranged_points[i+1][1] < arranged_points[i][1]:
            p3 = [arranged_points[i+1][0], arranged_points[i][1]]
            S.append([i + 1, p3])

        elif arranged_points[i+1][0] < arranged_points[i][0] and arranged_points[i+1][1] < arranged_points[i][1]:
            p3 = [arranged_points[i][0], arranged_points[i+1][1]]
            S.append([i + 1, p3])

        elif arranged_points[i+1][0] < arranged_points[i][0] and arranged_points[i+1][1] > arranged_points[i][1]:
            p3 = [arranged_points[i+1][0], arranged_points[i][1]]
            S.append([i + 1, p3])

    for i in range(len(S)):
        arranged_points.insert(S[i][0] + i, S[i][1])

    arranged_points.append(arranged_points[0])
    return arranged_points
# ==============================================================================
# 3. PIPELINE XỬ LÝ VÀ LƯU VIDEO
# ==============================================================================
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7.
MODEL_PATH = "bestseg.pt"
CLASSES = [2, 3, 5, 7]

# Lower conf lets ByteTrack use detections in its low-score association stage.
CONF = 0.05
IMGSZ = 960
MAX_DET = 300
DEVICE = "cpu"  # Change to 0 to use the first CUDA GPU.

# Draw the orthogonal convex hull computed from each YOLO segmentation mask.
DRAW_ORTHOGONAL_HULL = True

TRACKER_PATH = "my_bytetrack.yaml"
TRACKER_CONFIG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.20,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.20,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
}

# OpenCV colors are BGR.
COLOR_MAP = {
    2: (255, 255, 0),    # car: cyan
    3: (255, 255, 255),  # motorcycle: white
    5: (0, 0, 255),      # bus: red
    7: (0, 255, 255),    # truck: yellow
}


def main():
    video_name = input("Nhập tên file video (vd: test.mp4): ").strip().strip('"')
    if not video_name:
        raise ValueError("Bạn chưa nhập tên file video.")

    video_path = Path(video_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_name}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Không đọc được kích thước video: {video_name}")
    if not fps or not np.isfinite(fps):
        fps = 25.0

    output_name = f"output_{video_path.name}"
    writer = cv2.VideoWriter(
        output_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được video đầu ra: {output_name}")

    frame_idx = 0
    try:
        with open(TRACKER_PATH, "w", encoding="utf-8") as tracker_file:
            yaml.safe_dump(TRACKER_CONFIG, tracker_file, sort_keys=False)

        model = YOLO(MODEL_PATH)
        print(f"Đang xử lý '{video_name}' (bấm 'q' để dừng)...")
        cv2.namedWindow("Vehicle Segmentation Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vehicle Segmentation Tracking", 1280, 720)
        results = model.track(
            source=str(video_path),
            classes=CLASSES,
            conf=CONF,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            persist=True,
            stream=True,
            tracker=TRACKER_PATH,
            verbose=False,
            device=DEVICE,
        )

        for result in results:
            frame = result.orig_img.copy()
            counts = {class_id: 0 for class_id in CLASSES}
            boxes = result.boxes
            polygons = result.masks.xy if result.masks is not None else []

            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0].item())
                    counts[class_id] = counts.get(class_id, 0) + 1
                    color = COLOR_MAP.get(class_id, (0, 255, 255))
                    confidence = float(box.conf[0].item())

                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    class_name = model.names[class_id]
                    label = f"{class_name} {confidence:.2f}"
                    if track_id is not None:
                        label = f"#{track_id} {label}"

                    top_y = int(box.xyxy[0][1].item())
                    if i < len(polygons):
                        polygon = np.asarray(polygons[i], dtype=np.int32)
                        if len(polygon) >= 3:
                            if DRAW_ORTHOGONAL_HULL and len(polygon) >= 4:
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
                            x1, y1, x2, y2 = map(
                                int, box.xyxy[0].tolist()
                            )
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

            summary = (
                f"car:{counts.get(2, 0)}  moto:{counts.get(3, 0)}  "
                f"bus:{counts.get(5, 0)}  truck:{counts.get(7, 0)}"
            )
            cv2.putText(
                frame,
                summary,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}: {summary}")
            frame_idx += 1

            writer.write(frame)
            cv2.imshow("Vehicle Segmentation Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        writer.release()
        cv2.destroyAllWindows()
        print(f"Xong! Video đã lưu tại: {os.path.abspath(output_name)}")


if __name__ == "__main__":
    main()
