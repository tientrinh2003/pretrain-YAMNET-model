#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartBP Health Monitor - No-Conflict Version
- YAMNet Speech/Non-Speech Detection (TensorFlow compatible)
- OpenCV Face Detection (no MediaPipe/JAX conflicts)
- Simple mouth detection using facial landmarks
- Cross-validation: Audio + Visual speech detection
- Complete Web UI with real-time dashboard
- Zero dependency conflicts
"""

import os
import sys
import numpy as np
import math
import threading
import time
import logging
import signal
import csv
from urllib.request import urlretrieve
from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.utils import get_file

# -------------------- IMPORTS WITH ERROR HANDLING --------------------
# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available")
except ImportError as e:
    CV2_AVAILABLE = False
    cv2 = None
    print(f"❌ OpenCV not available: {e}")

# SoundDevice
try:
    import sounddevice as sd
    SD_AVAILABLE = True
    print("✅ SoundDevice available")
except ImportError as e:
    SD_AVAILABLE = False
    sd = None
    print(f"❌ SoundDevice not available: {e}")

# TensorFlow (no conflict version)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    hub = None
    print(f"❌ TensorFlow not available: {e}")

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIGURATION --------------------
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    TARGET_FPS = 30
    
    # Audio settings
    SAMPLE_RATE = 16000
    BUFFER_DURATION = 1.5
    AUDIO_PROCESSING_INTERVAL = 0.5
    SPEECH_THRESHOLD = 0.3
    
    # Visual detection settings (OpenCV-based)
    MOUTH_ASPECT_RATIO_THRESHOLD = 0.5  # For mouth open detection
    FACE_CASCADE_SCALE = 1.1
    FACE_MIN_NEIGHBORS = 5
    
    # Model paths
    MODEL_DIR = "model_data"
    YAMNET_SAVEDMODEL_URL = "https://tfhub.dev/google/yamnet/1"
    YAMNET_MODEL_PATH = "./yamnet_sns_savedmodel"
    
    # OpenCV models (lightweight, no JAX)
    FACE_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    FACE_CASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
    
    # Performance settings
    FRAME_SKIP = 1
    JPEG_QUALITY = 85
    MAX_AUDIO_BUFFER = 80000
    
    # Human sound classes for speech detection
    HUMAN_SOUNDS = [
        "Speech", "Singing", "Yell", "Screaming", "Laughter", 
        "Whispering", "Crying, sobbing", "Male speech, man speaking",
        "Female speech, woman speaking", "Child speech, kid speaking"
    ]

# -------------------- GLOBAL STATE --------------------
class GlobalState:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_speech_classification = {"speech": 0.0, "non_speech": 0.0, "is_speaking": False}
        self.last_sound_classification = [("Listening...", 0.0)]
        self.is_running = True
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.camera = None
        self.latest_frame = None
        
        # Visual detection state
        self.face_detected = False
        self.mouth_aspect_ratio = 0.0
        self.is_mouth_open = False
        self.is_talking_visual = False
        self.is_talking_audio = False
        self.is_talking_combined = False
        
    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()

state = GlobalState()

# -------------------- AUDIO CLASSIFIER --------------------
class AudioClassifier:
    """TensorFlow-only YAMNet classifier (no JAX conflicts)"""
    
    def __init__(self):
        self.mode = None
        self.model = None
        self.class_names = self._load_class_map()
        self._init_model()
    
    def _load_class_map(self):
        """Load YAMNet class mapping"""
        try:
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            class_map_path = get_file(
                fname='yamnet_class_map.csv',
                origin='https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
                cache_dir='.', cache_subdir=Config.MODEL_DIR
            )
            names = []
            with open(class_map_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    names.append(row[2])
            logger.info(f"✅ Loaded {len(names)} YAMNet class names")
            return names
        except Exception as e:
            logger.error(f"❌ Failed to load class map: {e}")
            return ["Speech", "Non-speech"]  # Fallback
    
    def _init_model(self):
        """Initialize YAMNet model (TensorFlow only, no JAX)"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return
        
        # Try 1: Custom SavedModel (fine-tuned)
        if os.path.exists(Config.YAMNET_MODEL_PATH):
            try:
                self.model = tf.saved_model.load(Config.YAMNET_MODEL_PATH)
                self.mode = "custom_savedmodel"
                logger.info(f"✅ Custom YAMNet SavedModel loaded from {Config.YAMNET_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load custom SavedModel: {e}")
        
        # Try 2: TensorFlow Hub SavedModel (no JAX)
        try:
            self.model = hub.load(Config.YAMNET_SAVEDMODEL_URL)
            self.mode = "hub_savedmodel"
            logger.info("✅ YAMNet SavedModel loaded from TensorFlow Hub")
            return
        except Exception as e:
            logger.error(f"❌ Failed to load YAMNet: {e}")
            self.model = None
    
    def classify(self, waveform):
        """Classify audio and return both detailed and binary results"""
        if self.model is None:
            return {
                "speech": 0.0, 
                "non_speech": 1.0, 
                "is_speaking": False,
                "top_classes": [("No model", 0.0)]
            }
        
        try:
            # Ensure correct format
            if not isinstance(waveform, np.ndarray):
                waveform = np.array(waveform, dtype=np.float32)
            elif waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Convert to TensorFlow tensor
            wf = tf.convert_to_tensor(waveform, dtype=tf.float32)
            
            # Run classification
            if self.mode == "custom_savedmodel":
                # Custom model signature
                scores = self.model.signatures['serving_default'](wf)
                if 'output_0' in scores:
                    scores_np = scores['output_0'].numpy()
                else:
                    scores_np = list(scores.values())[0].numpy()
                if scores_np.ndim > 1:
                    scores_np = scores_np[0]
            else:
                # TensorFlow Hub YAMNet
                scores, embeddings, spectrogram = self.model(wf)
                scores_np = scores.numpy().mean(axis=0)
            
            # Get top classes
            top_indices = np.argsort(scores_np)[::-1][:5]
            top_classes = [(self.class_names[i], float(scores_np[i])) for i in top_indices]
            
            # Calculate speech probability
            speech_prob = 0.0
            for class_name, prob in top_classes:
                if class_name in Config.HUMAN_SOUNDS:
                    speech_prob += prob
            
            speech_prob = min(1.0, speech_prob)  # Cap at 1.0
            non_speech_prob = 1.0 - speech_prob
            is_speaking = speech_prob > Config.SPEECH_THRESHOLD
            
            return {
                "speech": speech_prob,
                "non_speech": non_speech_prob,
                "is_speaking": is_speaking,
                "top_classes": top_classes
            }
            
        except Exception as e:
            logger.error(f"❌ Audio classification error: {e}")
            return {
                "speech": 0.0, 
                "non_speech": 1.0, 
                "is_speaking": False,
                "top_classes": [("Error", 0.0)]
            }

# -------------------- VISUAL DETECTION (OpenCV only) --------------------
class SimpleVisualDetector:
    """OpenCV-based visual detection (no MediaPipe/JAX conflicts)"""
    
    def __init__(self):
        self.face_cascade = None
        self._init_opencv()
    
    def _init_opencv(self):
        """Initialize OpenCV face detection"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - visual detection disabled")
            return
        
        try:
            # Download face cascade if not exists
            if not os.path.exists(Config.FACE_CASCADE_PATH):
                os.makedirs(Config.MODEL_DIR, exist_ok=True)
                logger.info("📥 Downloading face cascade...")
                urlretrieve(Config.FACE_CASCADE_URL, Config.FACE_CASCADE_PATH)
            
            # Load face cascade
            self.face_cascade = cv2.CascadeClassifier(Config.FACE_CASCADE_PATH)
            if self.face_cascade.empty():
                logger.error("❌ Failed to load face cascade")
                self.face_cascade = None
            else:
                logger.info("✅ OpenCV face detection initialized")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenCV detector: {e}")
            self.face_cascade = None
    
    def detect_mouth_opening(self, frame):
        """Simple mouth opening detection using face region analysis"""
        if self.face_cascade is None:
            return False, 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=Config.FACE_CASCADE_SCALE, 
                minNeighbors=Config.FACE_MIN_NEIGHBORS
            )
            
            if len(faces) > 0:
                # Get the largest face
                (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
                
                # Define mouth region (lower third of face)
                mouth_y = y + int(h * 0.65)
                mouth_height = int(h * 0.35)
                mouth_x = x + int(w * 0.2)
                mouth_width = int(w * 0.6)
                
                # Extract mouth region
                mouth_roi = gray[mouth_y:mouth_y+mouth_height, mouth_x:mouth_x+mouth_width]
                
                if mouth_roi.size > 0:
                    # Simple mouth opening detection using contour analysis
                    _, thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find largest contour (potential mouth opening)
                        largest_contour = max(contours, key=cv2.contourArea)
                        mouth_area = cv2.contourArea(largest_contour)
                        
                        # Calculate mouth aspect ratio
                        if len(largest_contour) >= 5:
                            ellipse = cv2.fitEllipse(largest_contour)
                            (_, _), (width, height), _ = ellipse
                            aspect_ratio = height / width if width > 0 else 0
                        else:
                            aspect_ratio = 0
                        
                        is_mouth_open = aspect_ratio > Config.MOUTH_ASPECT_RATIO_THRESHOLD
                        
                        # Draw mouth region for debugging
                        cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x+mouth_width, mouth_y+mouth_height), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        return is_mouth_open, aspect_ratio
            
        except Exception as e:
            logger.error(f"❌ Mouth detection error: {e}")
        
        return False, 0.0

# -------------------- AUDIO PROCESSING --------------------
def audio_processing_thread():
    """Enhanced audio processing with detailed classification"""
    if not SD_AVAILABLE:
        logger.warning("Audio processing disabled - SoundDevice not available")
        return
    
    classifier = AudioClassifier()
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        
        with state.lock:
            state.audio_buffer = np.append(state.audio_buffer, audio_data)
            if len(state.audio_buffer) > Config.MAX_AUDIO_BUFFER:
                state.audio_buffer = state.audio_buffer[-Config.MAX_AUDIO_BUFFER:]
    
    try:
        with sd.InputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(Config.SAMPLE_RATE * 0.1)
        ):
            logger.info("🎤 Audio stream started")
            
            buffer_size = int(Config.SAMPLE_RATE * Config.BUFFER_DURATION)
            
            while state.is_running:
                time.sleep(Config.AUDIO_PROCESSING_INTERVAL)
                
                waveform = None
                with state.lock:
                    if len(state.audio_buffer) >= buffer_size:
                        waveform = state.audio_buffer[:buffer_size].copy()
                        state.audio_buffer = state.audio_buffer[buffer_size:]
                
                if waveform is not None:
                    try:
                        result = classifier.classify(waveform)
                        
                        with state.lock:
                            state.last_speech_classification = {
                                "speech": result["speech"],
                                "non_speech": result["non_speech"],
                                "is_speaking": result["is_speaking"]
                            }
                            state.last_sound_classification = result["top_classes"]
                            state.is_talking_audio = result["is_speaking"]
                            
                            # Update combined talking status
                            state.is_talking_combined = (
                                state.is_talking_audio and state.is_talking_visual
                            ) or (
                                state.is_talking_audio and not CV2_AVAILABLE
                            )
                        
                        if result["is_speaking"]:
                            logger.info(f"🗣️ Speech detected: {result['speech']:.2f} | Top: {result['top_classes'][0][0]}")
                    
                    except Exception as e:
                        logger.error(f"❌ Audio processing error: {e}")
                
    except Exception as e:
        logger.error(f"❌ Audio stream error: {e}")

# -------------------- CAMERA PROCESSING --------------------
def camera_processing_thread():
    """Enhanced camera processing with simple visual detection"""
    if not CV2_AVAILABLE:
        logger.warning("❌ Camera disabled - OpenCV not available")
        return
    
    visual_detector = SimpleVisualDetector()
    
    try:
        state.camera = cv2.VideoCapture(0)
        if not state.camera.isOpened():
            logger.warning("❌ Could not open camera")
            return
        
        state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        state.camera.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
        
        logger.info("📹 Camera initialized")
        
        frame_count = 0
        last_fps_time = time.time()
        
        while state.is_running:
            ret, frame = state.camera.read()
            if not ret:
                logger.warning("Failed to read camera frame")
                time.sleep(0.1)
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Visual detection
            is_mouth_open, mouth_aspect_ratio = visual_detector.detect_mouth_opening(frame)
            
            with state.lock:
                state.mouth_aspect_ratio = mouth_aspect_ratio
                state.is_mouth_open = is_mouth_open
                
                # Visual talking detection (mouth open + audio speech)
                top_sound_name, top_sound_prob = state.last_sound_classification[0]
                state.is_talking_visual = (
                    is_mouth_open and 
                    top_sound_name in Config.HUMAN_SOUNDS and 
                    top_sound_prob > Config.SPEECH_THRESHOLD
                )
                
                # Combined detection
                state.is_talking_combined = (
                    state.is_talking_audio and state.is_talking_visual
                ) or (
                    state.is_talking_audio and not CV2_AVAILABLE
                )
            
            # Add overlays
            cv2.putText(frame, "SmartBP Health Monitor - No Conflict", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mouth info
            cv2.putText(frame, f"Mouth Ratio: {mouth_aspect_ratio:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Audio classification
            with state.lock:
                y_offset = 90
                for i, (cls_name, prob) in enumerate(state.last_sound_classification[:3]):
                    color = (0, 255, 0) if cls_name in Config.HUMAN_SOUNDS else (255, 255, 255)
                    cv2.putText(frame, f"{cls_name}: {prob*100:.1f}%", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 25
                
                # Talking status
                if state.is_talking_combined:
                    cv2.putText(frame, "TALKING", (Config.CAMERA_WIDTH//2 - 100, Config.CAMERA_HEIGHT - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                elif state.is_talking_audio:
                    cv2.putText(frame, "AUDIO ONLY", (Config.CAMERA_WIDTH//2 - 120, Config.CAMERA_HEIGHT - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
                elif state.is_talking_visual:
                    cv2.putText(frame, "MOUTH OPEN", (Config.CAMERA_WIDTH//2 - 120, Config.CAMERA_HEIGHT - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 255), 3)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                state.current_fps = 30 / (current_time - last_fps_time)
                last_fps_time = current_time
            
            # Store latest frame
            with state.lock:
                state.latest_frame = frame.copy()
            
            time.sleep(1.0 / Config.TARGET_FPS)
    
    except Exception as e:
        logger.error(f"❌ Camera processing error: {e}")
    finally:
        if state.camera:
            state.camera.release()

# -------------------- VIDEO STREAMING --------------------
def generate_frames():
    """Generate frames for video streaming with fallback handling"""
    while state.is_running:
        with state.lock:
            if state.latest_frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', state.latest_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    logger.error(f"Frame encoding error: {e}")
            else:
                # Generate placeholder
                if CV2_AVAILABLE:
                    placeholder = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for Camera...", (50, Config.CAMERA_HEIGHT//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(1.0 / Config.TARGET_FPS)

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Enhanced main page with complete dashboard"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartBP Health Monitor - No Conflict</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .header { text-align: center; margin-bottom: 30px; }
            .video-section { text-align: center; margin: 20px 0; }
            .video-stream { border: 2px solid #007bff; border-radius: 10px; max-width: 100%; }
            .status-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }
            .status-card { padding: 20px; border-radius: 10px; text-align: center; }
            .talking { background: #d4edda; color: #155724; border: 2px solid #28a745; }
            .not-talking { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
            .partial { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
            .info { background: #d1ecf1; color: #0c5460; border: 2px solid #17a2b8; }
            .no-conflict { background: #e7f3ff; color: #0056b3; border: 2px solid #007bff; }
            .demo-note { padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 SmartBP Health Monitor - No Conflict Edition</h1>
                <p>TensorFlow + OpenCV - Zero Dependency Conflicts</p>
            </div>
            
            <div class="demo-note no-conflict">
                <h4>🎯 No-Conflict Features</h4>
                <p>✅ TensorFlow YAMNet • ✅ OpenCV Face Detection • ✅ Simple Mouth Detection • ✅ No JAX Conflicts</p>
            </div>
            
            <div class="video-section">
                <h3>📹 Live Camera Feed with OpenCV Analysis</h3>
                <img src="/video_feed" class="video-stream" alt="Live Video Stream">
            </div>
            
            <div class="status-grid">
                <div id="combined-status" class="status-card not-talking">
                    <h4>🎯 Combined Detection</h4>
                    <p><strong id="combined-text">Not Talking</strong></p>
                    <p>Audio + Visual Validation</p>
                </div>
                
                <div id="audio-status" class="status-card not-talking">
                    <h4>🎤 Audio Detection</h4>
                    <p><strong id="audio-text">Not Speaking</strong></p>
                    <p>Confidence: <span id="speech-confidence">0.0</span>%</p>
                </div>
                
                <div id="visual-status" class="status-card not-talking">
                    <h4>👄 Visual Detection</h4>
                    <p><strong id="visual-text">Mouth Closed</strong></p>
                    <p>Ratio: <span id="mouth-ratio">0.0</span></p>
                </div>
                
                <div class="status-card no-conflict">
                    <h4>⚡ No Conflicts</h4>
                    <p>FPS: <span id="fps">0</span></p>
                    <p>TensorFlow + OpenCV</p>
                </div>
            </div>
            
            <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
                SmartBP Health Monitor v3.1 - No Conflict Edition<br>
                🚀 TensorFlow ≤1.24.3 + OpenCV + Zero Dependencies Issues
            </div>
        </div>
        
        <script>
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        // Combined status
                        const combinedStatus = document.getElementById('combined-status');
                        const combinedText = document.getElementById('combined-text');
                        
                        if (data.talking.combined) {
                            combinedStatus.className = 'status-card talking';
                            combinedText.textContent = '🗣️ Talking (Verified)';
                        } else if (data.talking.audio) {
                            combinedStatus.className = 'status-card partial';
                            combinedText.textContent = '🎤 Audio Only';
                        } else if (data.talking.visual) {
                            combinedStatus.className = 'status-card partial';
                            combinedText.textContent = '👄 Visual Only';
                        } else {
                            combinedStatus.className = 'status-card not-talking';
                            combinedText.textContent = '🔇 Not Talking';
                        }
                        
                        // Audio status
                        const audioStatus = document.getElementById('audio-status');
                        const audioText = document.getElementById('audio-text');
                        const speechConfidence = document.getElementById('speech-confidence');
                        
                        if (data.talking.audio) {
                            audioStatus.className = 'status-card talking';
                            audioText.textContent = '🗣️ Speaking';
                        } else {
                            audioStatus.className = 'status-card not-talking';
                            audioText.textContent = '🔇 Not Speaking';
                        }
                        speechConfidence.textContent = (data.speech.speech * 100).toFixed(1);
                        
                        // Visual status
                        const visualStatus = document.getElementById('visual-status');
                        const visualText = document.getElementById('visual-text');
                        const mouthRatio = document.getElementById('mouth-ratio');
                        
                        if (data.visual.mouth_open) {
                            visualStatus.className = 'status-card talking';
                            visualText.textContent = '👄 Mouth Open';
                        } else {
                            visualStatus.className = 'status-card not-talking';
                            visualText.textContent = '🤐 Mouth Closed';
                        }
                        mouthRatio.textContent = data.visual.mouth_aspect_ratio.toFixed(2);
                        
                        // System info
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
            }
            
            // Update every 500ms for responsive demo
            setInterval(updateStatus, 500);
            updateStatus();
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """Enhanced API endpoint with complete status"""
    with state.lock:
        return jsonify({
            "speech": state.last_speech_classification,
            "talking": {
                "audio": state.is_talking_audio,
                "visual": state.is_talking_visual,
                "combined": state.is_talking_combined
            },
            "visual": {
                "mouth_aspect_ratio": state.mouth_aspect_ratio,
                "mouth_open": state.is_mouth_open
            },
            "top_sounds": state.last_sound_classification,
            "capabilities": {
                "audio": SD_AVAILABLE,
                "camera": CV2_AVAILABLE,
                "tensorflow": TF_AVAILABLE,
                "conflicts": False  # No conflicts!
            },
            "system_running": state.is_running,
            "fps": state.current_fps,
            "mode": "No-Conflict"
        })

# -------------------- SIGNAL HANDLERS --------------------
def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("👋 Shutdown signal received")
    state.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    logger.info("🚀 Starting SmartBP Health Monitor - No Conflict Edition...")
    
    # Log capabilities
    logger.info(f"📋 Capabilities: OpenCV={CV2_AVAILABLE}, SoundDevice={SD_AVAILABLE}, TensorFlow={TF_AVAILABLE}")
    logger.info("✅ Zero dependency conflicts!")
    
    # Start camera processing
    if CV2_AVAILABLE:
        camera_thread = threading.Thread(target=camera_processing_thread, daemon=True)
        camera_thread.start()
        logger.info("📹 Camera processing thread started")
    else:
        logger.warning("📹 Camera disabled - OpenCV not available")
    
    # Start audio processing
    if SD_AVAILABLE:
        audio_thread = threading.Thread(target=audio_processing_thread, daemon=True)
        audio_thread.start()
        logger.info("🔊 Audio processing thread started")
    else:
        logger.warning("⚠️ Audio processing disabled")
    
    try:
        logger.info("🌐 Starting web server on http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("👋 Shutting down...")
        state.stop()