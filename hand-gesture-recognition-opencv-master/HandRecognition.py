import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, Response, jsonify, send_from_directory

# **********************************************
# * Advanced Hand Gesture Recognition v5.0
# * High-Accuracy Landmark Engine + 4s Scan
# **********************************************

app = Flask(__name__)

# --- STATIC ASSETS ---
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', MODEL_PATH)

# --- ENGINE ---
class AdvancedHandEngine:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_time = time.time()
        self.capture_start_time = 0
        self.is_capturing = False
        self.captured_gesture = None
        self.countdown = 4

    def update_state(self, cmd):
        if cmd == "start":
            self.capture_start_time = time.time()
            self.is_capturing = True
            self.captured_gesture = None
            self.countdown = 4

    def process_frame(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.detector.detect(mp_image)
        
        h, w, _ = image.shape
        gesture = "None"
        hand_present = False

        if detection_result.hand_landmarks:
            hand_present = True
            landmarks = detection_result.hand_landmarks[0]
            
            # --- Robust Gesture Logic ---
            gesture = self.detect_gesture_advanced(landmarks, detection_result.handedness[0][0].display_name)
            
            # Landmarks
            for lm in landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (px, py), 4, (255, 0, 255), -1)

        # Handle 4s Scan
        if self.is_capturing:
            elapsed = time.time() - self.capture_start_time
            self.countdown = max(0, 4 - int(elapsed))
            if elapsed >= 4.0:
                self.is_capturing = False
                self.captured_gesture = gesture
                self.countdown = 0
            
            overlay = image.copy()
            cv2.rectangle(overlay, (0,0), (w, h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
            cv2.putText(image, f"ANALYZING: {self.countdown}s", (w//2-180, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            # Show live suggestion during scan
            cv2.putText(image, f"Current: {gesture}", (w//2-150, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        
        if self.captured_gesture and not self.is_capturing:
            cv2.rectangle(image, (w//2-250, h//2-60), (w//2+250, h//2+60), (10,10,10), -1)
            cv2.putText(image, "FINAL DETECTION:", (w//2-220, h//2-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(image, self.captured_gesture, (w//2-220, h//2+30), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 2)

        return image

    def detect_gesture_advanced(self, lm, handedness):
        # Finger states
        f = [0,0,0,0,0] # Thumb, Index, Middle, Ring, Pinky
        
        # Thumb: Tip (4) vs IP (3) + relation to landmark 2
        if handedness == "Right":
            if lm[4].x < lm[3].x and lm[4].x < lm[2].x: f[0] = 1
        else:
            if lm[4].x > lm[3].x and lm[4].x > lm[2].x: f[0] = 1

        # 4 Fingers: Tip vs Joint
        for i, (tip, joint) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
            if lm[tip].y < lm[joint].y: f[i+1] = 1

        # --- Mapping Logic ---
        # Basic Counts
        if sum(f) == 5: return "Palm (Open)"
        if sum(f) == 0: return "Fist"
        
        # ASL & Sign Mapping
        if f == [1, 1, 0, 0, 0]: return "ASL: L / L-Sign"
        if f == [0, 1, 1, 0, 0]: return "Victory / ASL: V"
        if f == [0, 1, 0, 0, 0]: return "Pointing / ASL: D"
        if f == [1, 0, 0, 0, 0]: return "Thumbs Up"
        if f == [0, 0, 0, 0, 1]: return "Pinky Up / ASL: I"
        
        # Numbers
        if f == [0, 1, 1, 1, 0] or f == [1, 1, 0, 0, 0]: # Depending on thumb
             if f == [0, 1, 1, 1, 0]: return "Number: 3"
        if f == [0, 1, 1, 1, 1]: return "Number: 4"
        if f == [1, 1, 1, 1, 0]: return "ASL: B / Number: 3 (Alt)"
        
        # Special
        if f == [0, 1, 0, 0, 1]: return "Rock On (Metal)"
        
        # Close check for 'Okay' (Thumb and Index tips close)
        dist_ok = np.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
        if dist_ok < 0.05 and f[2:] == [1,1,1]:
            return "Okay Sign"

        return f"Active (F:{sum(f)})"

engine = AdvancedHandEngine()
cap = cv2.VideoCapture(0)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return f"""
    <html>
    <head>
        <title>Advanced Hand CV</title>
        <style>
            body {{ margin: 0; background: #0b0e14; color: #00d2ff; font-family: 'Segoe UI', sans-serif; display: flex; height: 100vh; overflow: hidden; }}
            .sidebar {{ width: 350px; background: #151921; border-right: 2px solid #00d2ff; overflow-y: auto; padding: 25px; box-shadow: 10px 0 20px rgba(0,0,0,0.5); }}
            .sidebar h2 {{ font-size: 1.1rem; border-bottom: 2px solid #00d2ff; padding-bottom: 15px; text-align: center; text-transform: uppercase; color:#fff}}
            .ref-grid {{ width: 100%; border-radius: 12px; border: 2px solid #00d2ff88; margin-bottom: 20px; }}
            
            .main {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
            .container {{ border: 6px solid #00d2ff; border-radius: 20px; overflow: hidden; box-shadow: 0 0 60px rgba(0, 210, 255, 0.4); position: relative; }}
            img.feed {{ display: block; max-height: 70vh; width: auto; }}
            
            .controls {{ margin-top: 35px; text-align: center; }}
            button {{ background: #00d2ff; border: none; padding: 18px 50px; border-radius: 35px; font-weight: bold; cursor: pointer; text-transform: uppercase; transition: 0.3s; color:#0e1117; font-size: 1.1rem; letter-spacing: 2px; }}
            button:hover {{ background: #fff; box-shadow: 0 0 30px #00d2ff; transform: translateY(-3px); }}
            .hint {{ margin-top: 15px; opacity: 0.6; font-size: 0.8rem; }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>📜 REFERENCE</h2>
            <img src="/assets/gesture_ref.png" class="ref-grid" alt="Reference Grid">
            <div style="background: #1c232e; padding: 15px; border-radius: 10px; font-size: 0.8rem; line-height: 1.6;">
                <p><b>Tips for Accuracy:</b></p>
                <ul style="padding-left: 20px;">
                    <li>Keep hand in center</li>
                    <li>Ensure good lighting</li>
                    <li>Avoid busy backgrounds</li>
                    <li>Face palm toward camera</li>
                </ul>
            </div>
        </div>
        <div class="main">
            <h1 style="letter-spacing: 8px; margin-bottom: 20px; text-shadow: 0 0 15px #00d2ff55;">G E S T U R E <span style="color:#fff">S C A N N E R</span></h1>
            <div class="container"><img src="/video_feed" class="feed"></div>
            <div class="controls">
                <button onclick="startCapture()">Run Analysis (4s)</button>
                <div class="hint">The scanner will freeze the result after 4 seconds of scanning.</div>
            </div>
        </div>
        <script>function startCapture() {{ fetch('/action/start'); }}</script>
    </body>
    </html>
    """

def gen():
    while True:
        success, frame = cap.read()
        if not success: break
        frame = engine.process_frame(cv2.flip(frame, 1))
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action/<cmd>')
def action(cmd):
    engine.update_state(cmd)
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
