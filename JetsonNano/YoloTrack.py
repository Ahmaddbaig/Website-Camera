import cv2
from adafruit_servokit import ServoKit
import time
from flask import Flask, render_template, Response
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import onnxruntime

# Load ONNX model ONCE at startup (use GPU provider if available)
session = onnxruntime.InferenceSession(
    "yolo11n.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# COCO classes (yolo11n is trained on COCO, adjust if needed)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# ONNX input processing (letterbox resize)
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def run_yolo(frame, conf_threshold=0.5):
    img, ratio, (dw, dh) = letterbox(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]  # shape (1, 3, 640, 640)
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    pred = outputs[0][0]  # (84, 8400)
    boxes = []
    for i in range(pred.shape[1]):
        conf = pred[4, i]
        if conf > conf_threshold:
            class_id = np.argmax(pred[5:, i])
            x, y, w, h = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
            # Scale coords back to original image size
            x = (x - dw) / ratio[0]
            y = (y - dh) / ratio[1]
            w = w / ratio[0]
            h = h / ratio[1]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            boxes.append({
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(frame.shape[1], x2), "y2": min(frame.shape[0], y2),
                "confidence": float(conf),
                "class": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
            })
    return boxes

# ========== SERVO/TRACKING CODE BELOW (UNCHANGED, MINOR PATCH) ==========

app = Flask(__name__)
kit = ServoKit(channels=16)
pan_servo_channel = 0
tilt_servo_channel = 1
PAN_INVERTED = False
TILT_INVERTED = False
pan_angle = 90
tilt_angle = 90
kit.servo[pan_servo_channel].angle = pan_angle
kit.servo[tilt_servo_channel].angle = tilt_angle
default_pan_step = 1.5
default_tilt_step = 1.0
max_pan_step = 5.0
max_tilt_step = 3.0
tilt_min_angle = 60
tilt_max_angle = 120
pan_left_limit = 0
pan_right_limit = 180
smoothing_factor = 0.4
movement_threshold = 5
confidence_threshold = 0.6
centering_tolerance = 15
aggressive_centering_distance = 50
detection_history = deque(maxlen=5)
min_consistent_detections = 2
last_detection_time = time.time()
timeout_duration = 2
scanning = False
scan_direction = 1
scan_speed = 2
frame_queue = queue.Queue(maxsize=3)
detection_results = []
detection_lock = threading.Lock()
last_fps_time = time.time()
frame_count = 0
current_fps = 0
frame_skip_counter = 0
FRAME_SKIP_RATE = 3
current_target = None
target_lock = threading.Lock()
last_servo_update = time.time()
servo_update_interval = 0.05
no_detection_counter = 0
NO_DETECTION_THRESHOLD = 5

class PositionFilter:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = self.alpha * self.x + (1 - self.alpha) * x
            self.y = self.alpha * self.y + (1 - self.alpha) * y
        return self.x, self.y

    def reset(self):
        self.x = None
        self.y = None

position_filter = PositionFilter(alpha=0.7)

# --- USE USB CAMERA INSTEAD OF GStreamer ---
def init_camera():
    cap = cv2.VideoCapture(1)  # Use your working USB camera index
    if not cap.isOpened():
        raise Exception("Failed to open USB camera at index 2!")
    print("USB camera opened successfully at index 2!")
    ret, frame = cap.read()
    if ret:
        print(f"First frame size: {frame.shape}")
    return cap

camera = init_camera()

def is_detection_stable(new_detection):
    if len(detection_history) < min_consistent_detections:
        return True
    recent_centers = []
    for det in detection_history:
        if det and len(det) > 0:
            d = det[0]
            cx = (d['x1'] + d['x2']) // 2
            cy = (d['y1'] + d['y2']) // 2
            recent_centers.append((cx, cy))
    if not recent_centers:
        return True
    avg_x = sum(c[0] for c in recent_centers) / len(recent_centers)
    avg_y = sum(c[1] for c in recent_centers) / len(recent_centers)
    new_cx = (new_detection['x1'] + new_detection['x2']) // 2
    new_cy = (new_detection['y1'] + new_detection['y2']) // 2
    distance = np.sqrt((new_cx - avg_x)**2 + (new_cy - avg_y)**2)
    size = (new_detection['x2'] - new_detection['x1']) * (new_detection['y2'] - new_detection['y1'])
    max_allowed_distance = 100 + (size / 1000)
    return distance < max_allowed_distance


def calculate_servo_movement(center_x, center_y, object_width, object_height):
    global pan_angle, tilt_angle, frame_count
    frame_center_x = 320
    frame_center_y = 240
    offset_x = center_x - frame_center_x
    offset_y = center_y - frame_center_y
    distance_factor_x = abs(offset_x) / frame_center_x
    distance_factor_y = abs(offset_y) / frame_center_y
    size = object_width * object_height
    size_factor = min(1.0, 20000 / size) if size > 0 else 1.0
    pan_movement = 0
    tilt_movement = 0
    if abs(offset_x) > 5:
        base_movement = default_pan_step + (distance_factor_x * 2.0)
        if PAN_INVERTED:
            pan_movement = np.sign(offset_x) * base_movement * size_factor
        else:
            pan_movement = -np.sign(offset_x) * base_movement * size_factor
        pan_movement = np.clip(pan_movement, -max_pan_step, max_pan_step)
    if abs(offset_y) > 5:
        base_movement = default_tilt_step + (distance_factor_y * 1.5)
        if TILT_INVERTED:
            tilt_movement = np.sign(offset_y) * base_movement * size_factor
        else:
            tilt_movement = -np.sign(offset_y) * base_movement * size_factor
        tilt_movement = np.clip(tilt_movement, -max_tilt_step, max_tilt_step)
    return pan_movement, tilt_movement

def move_servos_smooth(pan_movement, tilt_movement):
    global pan_angle, tilt_angle
    pan_angle += pan_movement * smoothing_factor
    tilt_angle += tilt_movement * smoothing_factor
    pan_angle = max(pan_left_limit, min(pan_right_limit, pan_angle))
    tilt_angle = max(tilt_min_angle, min(tilt_max_angle, tilt_angle))
    try:
        kit.servo[pan_servo_channel].angle = pan_angle
        kit.servo[tilt_servo_channel].angle = tilt_angle
    except Exception as e:
        print(f"Servo error: {e}")

def scan_for_target():
    global pan_angle, tilt_angle, scan_direction
    pan_angle += scan_speed * scan_direction
    if pan_angle >= pan_right_limit - 5 or pan_angle <= pan_left_limit + 5:
        scan_direction *= -1
    tilt_angle = 90
    try:
        kit.servo[pan_servo_channel].angle = pan_angle
        kit.servo[tilt_servo_channel].angle = tilt_angle
    except:
        pass


def detection_worker():
    global last_detection_time, scanning, frame_skip_counter, current_target, no_detection_counter
    while True:
        try:
            frame = None
            try:
                while True:
                    try:
                        frame = frame_queue.get_nowait()
                    except queue.Empty:
                        break
                if frame is None:
                    frame = frame_queue.get(timeout=0.1)
            except:
                time.sleep(0.01)
                continue
            frame_skip_counter += 1
            if frame_skip_counter % FRAME_SKIP_RATE != 0:
                continue

            detections = run_yolo(frame)  # <<=== LOCAL YOLO!

            if detections and len(detections) > 0:
                valid_detections = [d for d in detections if d['confidence'] > confidence_threshold]
                if valid_detections:
                    no_detection_counter = 0
                    detection_history.append(valid_detections)
                    best_detection = None
                    best_score = 0
                    for det in valid_detections:
                        size = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
                        score = size
                        if det['class'] == 'person':
                            score *= 2
                        if is_detection_stable(det):
                            if score > best_score:
                                best_score = score
                                best_detection = det
                    if best_detection:
                        with detection_lock:
                            detection_results.clear()
                            detection_results.append(best_detection)
                        with target_lock:
                            current_target = best_detection
                        last_detection_time = time.time()
                        scanning = False
                        cx = (best_detection['x1'] + best_detection['x2']) // 2
                        cy = (best_detection['y1'] + best_detection['y2']) // 2
                        position_filter.update(cx, cy)
                    else:
                        no_detection_counter += 1
                else:
                    no_detection_counter += 1
                    detection_history.append([])
            else:
                no_detection_counter += 1
                detection_history.append([])
            if no_detection_counter >= NO_DETECTION_THRESHOLD:
                detection_history.clear()
                no_detection_counter = 0
                print("Clearing detection history - no valid detections")
                with target_lock:
                    current_target = None
                position_filter.reset()
                with detection_lock:
                    detection_results.clear()
            if time.time() - last_detection_time > timeout_duration:
                if not scanning:
                    print(f"No detection for {timeout_duration} seconds - starting scan mode")
                    detection_history.clear()
                scanning = True
                with target_lock:
                    current_target = None
                position_filter.reset()
                with detection_lock:
                    detection_results.clear()
        except Exception as e:
            print(f"Detection worker error: {e}")
            time.sleep(0.01)

def servo_control_worker():
    global scanning, last_servo_update, frame_count
    while True:
        try:
            current_time = time.time()
            if current_time - last_servo_update < servo_update_interval:
                time.sleep(0.001)
                continue
            last_servo_update = current_time
            if scanning:
                scan_for_target()
            else:
                with target_lock:
                    target = current_target
                if target and position_filter.x is not None:
                    cx, cy = int(position_filter.x), int(position_filter.y)
                    w = target['x2'] - target['x1']
                    h = target['y2'] - target['y1']
                    pan_movement, tilt_movement = calculate_servo_movement(cx, cy, w, h)
                    if abs(pan_movement) > 0.01 or abs(tilt_movement) > 0.01:
                        move_servos_smooth(pan_movement, tilt_movement)
            time.sleep(0.01)
        except Exception as e:
            print(f"Servo control error: {e}")
            time.sleep(0.01)

def calculate_fps():
    global last_fps_time, frame_count, current_fps
    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        current_fps = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time

def gen_frames():
    global frame_count
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    servo_thread = threading.Thread(target=servo_control_worker, daemon=True)
    detection_thread.start()
    servo_thread.start()
    print("\n=== TRACKING SYSTEM STARTED ===")
    print(f"Running YOLO11n locally!")
    while True:
        try:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                continue
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, 480))
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            with detection_lock:
                for detection in detection_results:
                    x1 = detection['x1']
                    y1 = detection['y1']
                    x2 = detection['x2']
                    y2 = detection['y2']
                    label = detection['class']
                    conf = detection['confidence']
                    color = (255, 0, 0) if label == 'person' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label}: {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)
                    if position_filter.x is not None:
                        cv2.circle(frame, (int(position_filter.x), int(position_filter.y)),
                                 8, (255, 255, 0), 2)
                    cv2.putText(frame, "TRACKING", (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            center_x, center_y = 320, 240
            with detection_lock:
                is_centered = False
                if detection_results:
                    det = detection_results[0]
                    obj_center_x = (det['x1'] + det['x2']) // 2
                    obj_center_y = (det['y1'] + det['y2']) // 2
                    if (abs(obj_center_x - center_x) < centering_tolerance and
                        abs(obj_center_y - center_y) < centering_tolerance):
                        is_centered = True
            zone_color = (0, 255, 0) if is_centered else (0, 255, 255)
            cv2.rectangle(frame,
                         (center_x - centering_tolerance, center_y - centering_tolerance),
                         (center_x + centering_tolerance, center_y + centering_tolerance),
                         zone_color, 2)
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), zone_color, 1)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), zone_color, 1)
            cv2.rectangle(frame,
                         (center_x - aggressive_centering_distance, center_y - aggressive_centering_distance),
                         (center_x + aggressive_centering_distance, center_y + aggressive_centering_distance),
                         (128, 128, 128), 1)
            if is_centered:
                cv2.putText(frame, "CENTERED", (center_x - 40, center_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            calculate_fps()
            info_y = 30
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            status_text = "SCANNING" if scanning else "TRACKING"
            status_color = (0, 0, 255) if scanning else (0, 255, 0)
            cv2.putText(frame, f"Mode: {status_text}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            info_y += 30
            pan_warning = ""
            if pan_angle <= pan_left_limit + 1:
                pan_warning = " (LEFT LIMIT!)"
            elif pan_angle >= pan_right_limit - 1:
                pan_warning = " (RIGHT LIMIT!)"
            tilt_warning = ""
            if tilt_angle <= tilt_min_angle + 1:
                tilt_warning = " (DOWN LIMIT!)"
            elif tilt_angle >= tilt_max_angle - 1:
                tilt_warning = " (UP LIMIT!)"
            cv2.putText(frame, f"Pan: {pan_angle:.1f}°{pan_warning}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(frame, f"Tilt: {tilt_angle:.1f}°{tilt_warning}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(frame, f"Detections: {len(detection_results)}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0'})

@app.route('/')
def index():
    return render_template('index.html')

def cleanup():
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        JETSON_IP = '192.168.50.106'
        print(f"Starting camera server on http://{JETSON_IP}:5002")
        print("YOLO11n ONNX inference running locally on Jetson Orin Nano.")
        app.run(host=JETSON_IP, port=5002, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()


