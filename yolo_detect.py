import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO


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

# Bảng màu chuẩn theo 4 ID của COCO (BGR):
COLOR_MAP = {
    2: (255, 255, 0),      # 2: car        -> Xanh lơ
    3: (255, 255, 255),    # 3: motorcycle -> Trắng
    5: (0, 0, 255),        # 5: bus        -> Đỏ
    7: (0, 255, 255),      # 7: truck      -> Vàng
}

model = YOLO("yolo11n.pt")

video_name = input("Nhập tên file video (vd: test.mp4): ").strip()
cap = cv2.VideoCapture(video_name)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()

output_name = f"output_{os.path.basename(video_name)}"
writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

print(f"Đang xử lý '{video_name}' (Bấm 'q' để dừng)...")

try:
    # Lọc 4 lớp xe: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    for result in model.track(source=video_name, classes=[2, 3, 5, 7], conf=0.2, persist=True, stream=True):
        frame = result.orig_img.copy()

        if result.boxes is not None:
            for b in result.boxes:
                # Lấy thông tin xe từ YOLO
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                color = COLOR_MAP.get(cls_id, (0, 255, 255))
                label = f"{model.names[cls_id]} {float(b.conf[0]):.2f}"

                # BƯỚC 1: Cắt vùng xe (ROI) và trích xuất điểm cạnh bằng Canny
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                y_idx, x_idx = np.where(edges > 0)

                # BƯỚC 2: Lấy mẫu điểm (nhảy bước 10 để thuật toán chạy nhanh)
                pts = [[int(x + x1), int(y + y1)] for x, y in zip(x_idx[::10], y_idx[::10])]

                # BƯỚC 3: Tính và vẽ Bao lồi trực giao (Orthogonal Convex Hull)
                if len(pts) >= 4:
                    hull = findOrthogonalConvexHull(pts)
                    poly = np.array(hull, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

                    # BƯỚC 4: Ghi nhãn phía trên đỉnh cao nhất của đa giác
                    min_y = min(p[1] for p in hull)
                    cv2.putText(frame, label, (x1, max(15, min_y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(frame)
        cv2.imshow("Tracking & Hull", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    writer.release()
    cv2.destroyAllWindows()
    print(f"Xong! Video đã lưu tại: {output_name}")
