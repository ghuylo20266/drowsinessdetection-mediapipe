#!/usr/bin/env python
"""
Driver Drowsiness Detection - MediaPipe Face Landmarker (Tasks API)
Chay duoc voi mediapipe 0.10.31 (ban nay da bo mp.solutions).

Cai dat:
    pip install opencv-python numpy mediapipe==0.10.31
Model face_landmarker.task se duoc tu dong tai ve neu chua co (can Internet lan dau).
"""
import os
import sys
import tempfile
import time
import unicodedata
import urllib.request
import wave
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("[INFO] Dang tai model face_landmarker.task ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise SystemExit(f"[LOI] Khong tai duoc model: {e}\n"
                         f"Hay tai thu cong tu {MODEL_URL} va de canh file .py")

print("[INFO] Dang khoi tao MediaPipe Face Landmarker...")
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,   # điểm chớp mắt 0..1 chính xác hơn EAR
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.FaceLandmarker.create_from_options(options)
start_time = time.time()
fps, fps_t, fps_n = 0.0, time.time(), 0

print("[INFO] Dang mo camera...")
# Windows: backend DirectShow + MJPG thường cho FPS cao hơn mặc định (MSMF)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # giảm độ trễ hình
if not cap.isOpened():  # FIX: kiểm tra camera mở được hay không
    raise SystemExit("[LOI] Khong mo duoc camera. Kiem tra lai thiet bi / quyen truy cap.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)

# ---------------------------------------------------------------------------
# CHỈ SỐ ĐIỂM
# ---------------------------------------------------------------------------
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [263, 387, 385, 362, 373, 380]
MOUTH_INDICES = [78, 81, 13, 308, 14, 87]
# [Chóp mũi, Cằm, Khóe mắt (ảnh) trái, Khóe mắt (ảnh) phải, Khóe miệng trái, Khóe miệng phải]
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype="double")

# ---------------------------------------------------------------------------
# NGƯỠNG (FIX: dùng THỜI GIAN thay vì số frame để không phụ thuộc FPS)
# ---------------------------------------------------------------------------
EYE_BLINK_STRONG = 0.55     # Blink cao -> chắc chắn nhắm, bất kể góc đầu
EYE_BLINK_WEAK = 0.35       # Blink vừa -> chỉ tính là nhắm nếu EAR cũng tụt so với lúc mở
EYE_REL_WEAK = 0.75         # EAR / EAR_lúc_mở < 0.75
EYE_REL_NOBLINK = 0.65      # dự phòng khi không có blendshape
BASELINE_SECONDS = 5.0      # EAR "lúc mở mắt" lấy từ 5s gần nhất (tự thích nghi theo góc đầu)
EYE_CLOSED_SECONDS = 0.8    # nhắm mắt quá 1s mới báo (chớp mắt bình thường ~0.1-0.4s)
MOUTH_AR_THRESH = 0.60      # FIX: 0.79 quá cao với MAR môi trong, gần như không bao giờ kích hoạt
YAWN_SECONDS = 1.0
HEAD_NOD_DEG = 15.0         # pitch lệch so với tư thế bình thường của bạn (độ)
HEAD_NOD_RATIO = 0.15       # hoặc tỉ lệ vị trí mũi trên mặt lệch quá mức này
HEAD_NOD_SECONDS = 1.0
HEAD_BASELINE_SECONDS = 10.0  # tư thế "bình thường" lấy từ 10s gần nhất


def calculate_ear(landmarks, eye_indices, w, h):
    """EAR tính trong không gian 3D (có z) để đỡ bị méo khi quay/nghiêng đầu."""
    p3 = [np.array([landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w]) for i in eye_indices]
    v1 = np.linalg.norm(p3[1] - p3[5])
    v2 = np.linalg.norm(p3[2] - p3[4])
    hd = np.linalg.norm(p3[0] - p3[3])
    pts2d = [p[:2] for p in p3]                      # dùng để vẽ
    return (v1 + v2) / (2.0 * hd + 1e-6), pts2d


def calculate_mar(landmarks, mouth_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in mouth_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hd = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hd + 1e-6), pts


def normalize_pitch(angle):
    """FIX: RQDecomp3x3 với model này hay trả về quanh ±180 độ -> đưa về [-90, 90]."""
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


# Bộ đếm thời gian bắt đầu của từng trạng thái (None = không đang xảy ra)
DEBUG = False   # False: chỉ hiện cảnh báo. Nhấn phím 'd' để bật/tắt điểm + số liệu


# ---------------------------------------------------------------------------
# CÒI BÁO ĐỘNG "ò e ò e"
# ---------------------------------------------------------------------------
class Siren:
    """Còi hai âm 'ò e' lặp lại. Windows dùng winsound; hệ khác dùng pygame (nếu có)."""

    def __init__(self, low=640, high=900, beat=0.40, rate=22050):
        self.path = os.path.join(tempfile.gettempdir(), "drowsy_siren.wav")
        self.playing = False
        self.muted = False
        self.backend = None
        self._make_wav(low, high, beat, rate)
        if sys.platform.startswith("win"):
            import winsound
            self.winsound = winsound
            self.backend = "winsound"
        else:
            try:
                import pygame
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(self.path)
                self.backend = "pygame"
            except Exception:
                print("[CANH BAO] Khong co am thanh. Cai: pip install pygame")

    def _make_wav(self, low, high, beat, rate):
        parts = []
        for freq in (low, high):
            t = np.arange(int(rate * beat)) / rate
            tone = np.sin(2 * np.pi * freq * t)
            fade = int(rate * 0.01)                       # 10ms mờ dần, tránh tiếng "tạch"
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
            parts.append(tone)
        data = (np.concatenate(parts) * 0.8 * 32767).astype("<i2")
        with wave.open(self.path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(data.tobytes())

    def start(self):
        if self.playing or self.muted or self.backend is None:
            return
        if self.backend == "winsound":
            ws = self.winsound
            ws.PlaySound(self.path, ws.SND_FILENAME | ws.SND_ASYNC | ws.SND_LOOP)
        else:
            self.sound.play(loops=-1)
        self.playing = True

    def stop(self):
        if not self.playing:
            return
        if self.backend == "winsound":
            self.winsound.PlaySound(None, self.winsound.SND_PURGE)
        else:
            self.sound.stop()
        self.playing = False


siren = Siren()
alert_this_frame = False
last_alert_time = 0.0


# ---------------------------------------------------------------------------
# HIỂN THỊ TIẾNG VIỆT (cv2.putText không vẽ được chữ có dấu -> dùng Pillow)
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return None


FONT_BIG = load_font(30)
FONT_SMALL = load_font(20)
if FONT_BIG is None:
    print("[CANH BAO] Khong tim thay font Unicode -> hien chu khong dau.")


def strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


pending_frame, pending_D = [], []   # (text, x, y, big, color_rgb)


def alert(text, y):
    global alert_this_frame
    alert_this_frame = True
    item = (text, 10, y, True, (255, 0, 0))
    pending_frame.append(item)
    pending_D.append(item)


def dtext(text, x, y, color=(255, 0, 0)):
    pending_D.append((text, x, y, False, color))


def render_texts(img, items):
    """Vẽ toàn bộ chữ tiếng Việt lên img (BGR) trong một lần chuyển đổi."""
    if not items:
        return
    if FONT_BIG is None:   # dự phòng: bỏ dấu, dùng OpenCV
        for text, x, y, big, rgb in items:
            cv2.putText(img, strip_accents(text), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 if big else 0.5, rgb[::-1], 2)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, big, rgb in items:
        draw.text((x, y - 24), text, font=FONT_BIG if big else FONT_SMALL, fill=rgb,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


eye_start = None
ear_hist = deque()
eye_open_since = None
no_face_since = None
yawn_start = None
nod_start = None
nod_hist = deque()


def duration(start):
    return 0.0 if start is None else time.time() - start


try:
    while True:
        success, frame = cap.read()
        if not success:
            print("[LOI] Khong doc duoc frame tu camera.")
            break

        # FIX: KHÔNG ép về 1024x576 (làm méo tỉ lệ 4:3 -> sai EAR/MAR/solvePnP)
        h, w = frame.shape[:2]
        D = frame.copy()   # lớp vẽ điểm/số liệu (chỉ hiện khi DEBUG)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)  # phải tăng dần
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            no_face_since = None
            dtext("Phát hiện 1 khuôn mặt", 10, 20)

            # ---------------- 1. MẮT (EAR) ----------------
            leftEAR, left_pts = calculate_ear(landmarks, LEFT_EYE_INDICES, w, h)
            rightEAR, right_pts = calculate_ear(landmarks, RIGHT_EYE_INDICES, w, h)
            ear = (leftEAR + rightEAR) / 2.0

            cv2.polylines(D, [np.array(left_pts, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(D, [np.array(right_pts, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.putText(D, f"EAR: {ear:.2f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Điểm chớp mắt từ blendshape (0 = mở hoàn toàn, 1 = nhắm hoàn toàn)
            blink = None
            if results.face_blendshapes:
                scores = {c.category_name: c.score for c in results.face_blendshapes[0]}
                blink = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
                cv2.putText(D, f"Blink: {blink:.2f}", (150, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # EAR "lúc mở mắt" của chính tư thế hiện tại: phân vị 80 của 5s gần nhất
            tnow = time.time()
            vals = [e for _, e in ear_hist]
            baseline = float(np.percentile(vals, 80)) if len(vals) >= 15 else 0.30
            baseline = max(baseline, 0.15)
            rel = ear / baseline
            dtext(f"Rel: {rel:.2f}", 480, 45)

            if blink is not None:
                eyes_closed = (blink > EYE_BLINK_STRONG) or \
                              (blink > EYE_BLINK_WEAK and rel < EYE_REL_WEAK)
            else:
                eyes_closed = rel < EYE_REL_NOBLINK

            # chỉ học baseline khi mắt đang mở, để lúc nhắm lâu không kéo baseline xuống
            if not eyes_closed and eye_start is None:
                ear_hist.append((tnow, ear))
            while ear_hist and tnow - ear_hist[0][0] > BASELINE_SECONDS:
                ear_hist.popleft()

            now = time.time()
            if eyes_closed:
                if eye_start is None:
                    eye_start = now
                eye_open_since = None
            else:
                # cho phép "hở" tối đa 0.3s (nhiễu 1-2 frame) trước khi reset
                if eye_open_since is None:
                    eye_open_since = now
                if now - eye_open_since > 0.3:
                    eye_start = None

            closed_for = duration(eye_start)
            dtext(f"Nhắm mắt: {closed_for:.1f}s", 330, 45)
            if closed_for >= EYE_CLOSED_SECONDS:
                alert("BUỒN NGỦ: Đang nhắm mắt! - Closing eyes!", 80)

            # ---------------- 2. MIỆNG (MAR) ----------------
            mar, mouth_pts = calculate_mar(landmarks, MOUTH_INDICES, w, h)
            cv2.polylines(D, [np.array(mouth_pts, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.putText(D, f"MAR: {mar:.2f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                if yawn_start is None:
                    yawn_start = time.time()
                if duration(yawn_start) >= YAWN_SECONDS:
                    alert("Đang ngáp! - Yawning!", 110)
            else:
                yawn_start = None

            # ---------------- 3. TƯ THẾ ĐẦU (solvePnP) ----------------
            image_points = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_POSE_INDICES],
                dtype="double",
            )
            focal_length = w
            camera_matrix = np.array(
                [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
                dtype="double",
            )
            dist_coeffs = np.zeros((4, 1))

            ok, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            for idx, p in enumerate(image_points):
                cv2.circle(D, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                cv2.putText(D, str(idx + 1), (int(p[0]) - 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            if ok:  # FIX: chỉ dùng kết quả khi solvePnP thành công
                axis_z, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                              rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)
                axis_y, _ = cv2.projectPoints(np.array([(0.0, 1000.0, 0.0)]),
                                              rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(axis_z[0][0][0]), int(axis_z[0][0][1]))
                p3 = (int(axis_y[0][0][0]), int(axis_y[0][0][1]))
                cv2.line(D, p1, p2, (255, 0, 0), 2)
                cv2.line(D, p1, p3, (0, 0, 255), 2)

                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = cv2.RQDecomp3x3(rmat)[0]
                head_tilt_degree = normalize_pitch(angles[0])

                # Tỉ lệ vị trí mũi giữa đường mắt và cằm (không phụ thuộc quy ước góc)
                eye_mid_y = (landmarks[33].y + landmarks[263].y) / 2.0
                face_h = landmarks[152].y - eye_mid_y
                nose_ratio = (landmarks[1].y - eye_mid_y) / (face_h + 1e-6)

                # Tư thế "bình thường" của chính người dùng (trung vị 10s gần nhất)
                tn = time.time()
                if len(nod_hist) >= 20:
                    base_pitch = float(np.median([x[0] for x in nod_hist]))
                    base_ratio = float(np.median([x[1] for x in nod_hist]))
                    d_pitch = head_tilt_degree - base_pitch
                    d_ratio = nose_ratio - base_ratio
                    nodding = abs(d_pitch) > HEAD_NOD_DEG or abs(d_ratio) > HEAD_NOD_RATIO
                    dtext(f"Góc đầu: {head_tilt_degree:.1f} (lệch {d_pitch:+.1f}) | mũi {d_ratio:+.2f}", 10, 135)
                else:
                    nodding = False   # đang học tư thế bình thường (~1 giây đầu)
                    dtext("Đang hiệu chỉnh tư thế đầu...", 10, 135)

                if nodding:
                    if nod_start is None:
                        nod_start = tn
                    if duration(nod_start) >= HEAD_NOD_SECONDS:
                        alert("BUỒN NGỦ: Gục đầu! - Head drooping!", 170)
                else:
                    nod_start = None
                    nod_hist.append((head_tilt_degree, nose_ratio, tn))
                while nod_hist and tn - nod_hist[0][2] > HEAD_BASELINE_SECONDS:
                    nod_hist.popleft()

            # ---------------- 4. LƯỚI ĐIỂM TRANG TRÍ ----------------
            for i, lm in (enumerate(landmarks) if DEBUG else ()):
                if i not in HEAD_POSE_INDICES and i % 7 == 0:
                    cv2.circle(D, (int(lm.x * w), int(lm.y * h)), 1, (0, 0, 255), -1)
        else:
            # Khi nhắm chặt mắt, mô hình đôi khi mất mặt: vẫn giữ bộ đếm mắt
            # trong 3 giây (và vẫn cảnh báo), sau đó mới reset.
            now = time.time()
            if no_face_since is None:
                no_face_since = now
            yawn_start = None
            if nod_start is not None and now - no_face_since < 3.0:
                if duration(nod_start) >= HEAD_NOD_SECONDS:
                    alert("BUỒN NGỦ: Gục đầu! - Head drooping!", 170)
            else:
                nod_start = None
            if eye_start is not None and now - no_face_since < 3.0:
                if duration(eye_start) >= EYE_CLOSED_SECONDS:
                    alert("BUỒN NGỦ: Đang nhắm mắt! - Closing eyes!", 80)
            else:
                eye_start = None
            dtext("Không thấy khuôn mặt", 10, 20, (255, 165, 0))

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps, fps_t, fps_n = fps_n / (time.time() - fps_t), time.time(), 0
        if DEBUG:
            dtext(f"FPS: {fps:.0f}", 10, 200)
        out = D if DEBUG else frame
        render_texts(out, pending_D if DEBUG else pending_frame)
        pending_frame.clear()
        pending_D.clear()

        # Còi: bật khi có cảnh báo, tắt khi hết cảnh báo hơn 0.6 giây
        if alert_this_frame:
            last_alert_time = time.time()
            siren.start()
        elif time.time() - last_alert_time > 0.6:
            siren.stop()
        alert_this_frame = False
        cv2.imshow("AI Anti-Sleep - MediaPipe", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            DEBUG = not DEBUG
        if key == ord("m"):          # tắt/bật tiếng còi
            siren.muted = not siren.muted
            if siren.muted:
                siren.stop()
finally:
    siren.stop()
    # FIX: luôn giải phóng tài nguyên kể cả khi có lỗi
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
