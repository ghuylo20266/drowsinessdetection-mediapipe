#!/usr/bin/env python
import cv2
import mediapipe as mp
import numpy as np
import time

# Khởi tạo bộ giải pháp lưới khuôn mặt thế hệ mới của Google
print("[INFO] Đang khởi tạo mô hình lõi Google MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Bật quét chi tiết mắt siêu độ phân giải
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mở luồng Video từ Webcam
print("[INFO] Đang kích hoạt camera cảm biến...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Cấu hình kích thước khung hình chuẩn tương đương file gốc
frame_width = 1024
frame_height = 576

# =========================================================================
# ĐỊNH NGHĨA CÁC CHỈ SỐ ĐIỂM (INDICES) CHUẨN GOOGLE MEDIAPIPE
# =========================================================================
# Thứ tự 6 điểm vùng mắt phục vụ công thức EAR
RIGHT_EYE_INDICES = 
LEFT_EYE_INDICES  = 

# Thứ tự các điểm bờ môi trong phục vụ công thức MAR
MOUTH_INDICES = 

# 6 Điểm mấu chốt để định hình mô hình không gian 2D (Tương ứng 6 điểm dlib cũ)
# Thứ tự nghiêm ngặt: [Chóp mũi, Cằm, Khóe mắt trái, Khóe mắt phải, Khóe miệng trái, Khóe miệng phải]
HEAD_POSE_INDICES = 

# Các hằng số tọa độ mô hình 3D chuẩn thế giới (World 3D Model Points)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Chóp mũi (Nose tip)
    (0.0, -330.0, -65.0),        # Cằm (Chin)
    (-225.0, 170.0, -135.0),     # Góc ngoài mắt trái (Left eye corner)
    (225.0, 170.0, -135.0),      # Góc ngoài mắt phải (Right eye corner)
    (-150.0, -150.0, -125.0),    # Khóe miệng trái (Left Mouth corner)
    (150.0, -150.0, -125.0)      # Khóe miệng phải (Right mouth corner)
], dtype="double")

# Cấu hình các ngưỡng kiểm soát động hành vi tương tự file gốc
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

def calculate_ear(landmarks, eye_indices, w, h):
    """Tính toán Tỷ lệ khép mắt (Eye Aspect Ratio) theo khoảng cách Euclid"""
    pts = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in eye_indices]
    p2_p6 = np.linalg.norm(pts - pts)
    p3_p5 = np.linalg.norm(pts - pts)
    p1_p4 = np.linalg.norm(pts - pts)
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear, pts

def calculate_mar(landmarks, mouth_indices, w, h):
    """Tính toán Tỷ lệ mở miệng (Mouth Aspect Ratio) để xác định trạng thái ngáp"""
    pts = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in mouth_indices]
    v1 = np.linalg.norm(pts - pts)
    v2 = np.linalg.norm(pts - pts)
    h_dist = np.linalg.norm(pts - pts)
    mar = (v1 + v2) / (2.0 * h_dist)
    return mar, pts

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Chuẩn hóa kích thước khung hình và chuyển màu sang RGB
    frame = cv2.resize(frame, (frame_width, frame_height))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Đẩy dữ liệu qua mạng nơ-ron MediaPipe xử lý
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Hiển thị số lượng khuôn mặt tìm thấy
        text = f"{len(results.multi_face_landmarks)} face(s) found"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # -----------------------------------------------------------------
            # 1. KIỂM TRA TRẠNG THÁI MẮT (EAR) - BÁO ĐỘNG NGỦ GẬT
            # -----------------------------------------------------------------
            leftEAR, left_pts = calculate_ear(landmarks, LEFT_EYE_INDICES, w, h)
            rightEAR, right_pts = calculate_ear(landmarks, RIGHT_EYE_INDICES, w, h)
            ear = (leftEAR + rightEAR) / 2.0

            # Vẽ đường bao mắt dạng đa giác (Thay thế cho ConvexHull dlib)
            cv2.polylines(frame, [np.array(left_pts, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_pts, dtype=np.int32)], True, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes Closed!", (500, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            # -----------------------------------------------------------------
            # 2. KIỂM TRA TRẠNG THÁI MIỆNG (MAR) - BÁO ĐỘNG NGÁP
            # -----------------------------------------------------------------
            mar, mouth_pts = calculate_mar(landmarks, MOUTH_INDICES, w, h)
            cv2.polylines(frame, [np.array(mouth_pts, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (650, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # -----------------------------------------------------------------
            # 3. TRÍCH XUẤT ĐIỂM 2D & ƯỚC LƯỢNG GÓC ĐẦU (HEAD POSE - SOLVEPNP)
            # -----------------------------------------------------------------
            image_points = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in HEAD_POSE_INDICES], dtype="double")
            
            # Khởi tạo thông số nội tại camera (Camera Matrix) giả định
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center], [0, focal_length, center]], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Giả định thấu kính không biến dạng

            # Giải bài toán PnP toán học biến đổi không gian hình học
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Vẽ và đánh số ID cho 6 điểm mấu chốt cấu hình không gian giống file gốc
            for key_idx, idx in enumerate(HEAD_POSE_INDICES):
                cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(key_idx + 1), (cx - 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Vẽ các chấm đỏ đại diện cho các mốc điểm hình học chính xác lên mặt tài xế
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # -----------------------------------------------------------------
            # 4. TÍNH TOÁN VÀ VẼ ĐƯỜNG ĐỊNH HƯỚNG KHÔNG GIAN (HEAD TILT VECTOR)
            # -----------------------------------------------------------------
            # Chiếu điểm 3D từ mũi ra không gian ngoài để tạo các trục Vector chiếu
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (nose_end_point2D_alt, _) = cv2.projectPoints(np.array([(0.0, 1000.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Xác định các điểm đầu và điểm cuối trên ma trận hiển thị phẳng 2D
            p1 = (int(image_points[0][0]), int(image_points[0][1])) # Gốc mũi
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) # Trục đâm thẳng hướng nhìn
            p3 = (int(nose_end_point2D_alt[0][0][0]), int(nose_end_point2D_alt[0][0][1])) # Trục đứng vuông góc

            # Vẽ các đường thẳng mô phỏng hướng góc nghiêng không gian
            cv2.line(frame, p1, p2, (255, 0, 0), 2)  # Đường màu xanh dương hướng nhìn
            cv2.line(frame, p1, p3, (0, 0, 255), 2)  # Đường màu đỏ hướng trục đầu

            # Tính toán góc cúi/ngửa gật đầu (Pitch Angle) dựa trên ma trận xoay Rodrigues
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            head_tilt_degree = angles

            # Đưa thông số góc nghiêng đầu thời gian thực lên màn hình cho Ban Giám Khảo xem
            cv2.putText(frame, f"Head Tilt Degree: {head_tilt_degree:.2f}", (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # -----------------------------------------------------------------
            # 5. HIỂN THỊ MẠNG LƯỚI ĐIỂM PHỤ TRÌNH DIỄN (VISUAL EFFECTS)
            # -----------------------------------------------------------------
            # Vẽ các điểm lưới liti màu đỏ khác để tăng độ ấn tượng mặt trực quan
            for i, lm in enumerate(landmarks):
                if i not in HEAD_POSE_INDICES and i % 7 == 0:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)

    cv2.imshow("AI Anti-Sleep - MediaPipe 2026", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
