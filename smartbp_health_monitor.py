#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartBP Health Monitor - Raspberry Pi 5 Production
- YAMNet Speech/Non-Speech Detection (Fine-tuned model)  
- FaceMesh lip detection for talking verification
- Pose detection with BP cuff placement guide
- Optimized for ARM64 architecture
"""

import os
import sys
import cv2
import numpy as np
import math
import threading
import time
import csv
import logging
import signal
import zipfile
import json
from urllib.request import urlretrieve
from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/pi/smartbp_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIGURATION --------------------
class Config:
    # Hardware constraints for Raspberry Pi 5
    MAX_THREADS = 2
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    TARGET_FPS = 20
    
    # Audio settings for YAMNet
    SAMPLE_RATE = 16000
    BUFFER_DURATION = 0.975  # YAMNet input duration
    AUDIO_PROCESSING_INTERVAL = 0.5
    SPEECH_THRESHOLD = 0.6  # Speech confidence threshold
    
    # Model paths
    MODEL_DIR = "/home/pi/models"
    YAMNET_FINETUNED_ZIP = "/home/pi/yamnet_finetuned_model.zip"  # Your fine-tuned model
    YAMNET_MODEL_PATH = os.path.join(MODEL_DIR, "yamnet_finetuned.tflite")
    
    # Face mesh model URLs
    FACEMESH_URL = "https://tfhub.dev/mediapipe/lite-model/face_landmark/1?lite-format=tflite"
    FACEMESH_PATH = os.path.join(MODEL_DIR, "face_landmark.tflite")
    
    # Pose model URLs (MoveNet Lightning)
    POSE_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/4?lite-format=tflite"
    POSE_PATH = os.path.join(MODEL_DIR, "movenet_lightning.tflite")
    
    # Performance settings
    FRAME_SKIP = 2
    JPEG_QUALITY = 75
    MAX_AUDIO_BUFFER = 80000  # ~5 seconds at 16kHz
    
    # BP Cuff positioning (in inches converted to pixels)
    BP_CUFF_OFFSET_INCHES = 2.0  # 2 inches above elbow
    PIXELS_PER_INCH = 30  # Approximate for typical camera distance

# -------------------- TENSORFLOW SETUP --------------------
def setup_tensorflow():
    """Configure TensorFlow for Raspberry Pi"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(Config.MAX_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(Config.MAX_THREADS)
        tf.config.set_visible_devices([], 'GPU')
        logger.info("TensorFlow configured for Raspberry Pi")
        return tf
    except ImportError:
        logger.error("TensorFlow not available")
        return None

tf = setup_tensorflow()

# -------------------- DEPENDENCY CHECKS --------------------
def check_dependencies():
    """Check and import dependencies"""
    global sd
    
    try:
        import sounddevice as sd
        logger.info("✅ SoundDevice available")
        return True
    except ImportError as e:
        logger.warning(f"SoundDevice not available: {e}")
        return False

AUDIO_AVAILABLE = check_dependencies()

# -------------------- GLOBAL STATE --------------------
class GlobalState:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_speech_classification = {"speech": 0.0, "non_speech": 0.0, "is_speaking": False}
        self.is_running = True
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.camera = None
        self.lip_distance = 0.0
        self.is_talking_face = False
        self.bp_cuff_valid = False
        
    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()

state = GlobalState()

# -------------------- MODEL SETUP --------------------
def extract_finetuned_yamnet():
    """Extract fine-tuned YAMNet model from ZIP"""
    try:
        if os.path.exists(Config.YAMNET_FINETUNED_ZIP) and not os.path.exists(Config.YAMNET_MODEL_PATH):
            logger.info("📦 Extracting fine-tuned YAMNet model...")
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            
            with zipfile.ZipFile(Config.YAMNET_FINETUNED_ZIP, 'r') as zip_ref:
                # Look for .tflite file in the ZIP
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.tflite'):
                        # Extract the TFLite model
                        zip_ref.extract(file_name, Config.MODEL_DIR)
                        # Rename to standard name
                        extracted_path = os.path.join(Config.MODEL_DIR, file_name)
                        os.rename(extracted_path, Config.YAMNET_MODEL_PATH)
                        logger.info(f"✅ Extracted YAMNet model: {file_name}")
                        break
                else:
                    logger.error("❌ No .tflite file found in ZIP")
                    return False
            
            # Also extract class mapping if available
            try:
                with zipfile.ZipFile(Config.YAMNET_FINETUNED_ZIP, 'r') as zip_ref:
                    for file_name in zip_ref.namelist():
                        if 'class' in file_name.lower() and file_name.endswith('.json'):
                            zip_ref.extract(file_name, Config.MODEL_DIR)
                            logger.info(f"✅ Extracted class mapping: {file_name}")
                            break
            except:
                pass
                
        return os.path.exists(Config.YAMNET_MODEL_PATH)
    except Exception as e:
        logger.error(f"❌ Failed to extract YAMNet model: {e}")
        return False

def download_models():
    """Download required models"""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Extract fine-tuned YAMNet
    if not extract_finetuned_yamnet():
        logger.warning("Using default YAMNet model")
    
    # Download FaceMesh model
    if not os.path.exists(Config.FACEMESH_PATH):
        logger.info("📥 Downloading FaceMesh model...")
        try:
            urlretrieve(Config.FACEMESH_URL, Config.FACEMESH_PATH)
            logger.info("✅ FaceMesh model downloaded")
        except Exception as e:
            logger.error(f"❌ Failed to download FaceMesh: {e}")
    
    # Download Pose model
    if not os.path.exists(Config.POSE_PATH):
        logger.info("📥 Downloading Pose model...")
        try:
            urlretrieve(Config.POSE_URL, Config.POSE_PATH)
            logger.info("✅ Pose model downloaded")
        except Exception as e:
            logger.error(f"❌ Failed to download Pose model: {e}")

# -------------------- AUDIO CLASSIFIER (FINE-TUNED YAMNET) --------------------
class FineTunedYAMNetClassifier:
    """Fine-tuned YAMNet for Speech/Non-Speech detection"""
    
    def __init__(self):
        self.model = None
        self.input_length = 15600  # YAMNet standard input length
        self.class_names = ["non_speech", "speech"]  # Binary classification
        self._initialize()
    
    def _initialize(self):
        """Initialize fine-tuned TFLite model"""
        if not tf:
            logger.error("TensorFlow not available")
            return
        
        try:
            if not os.path.exists(Config.YAMNET_MODEL_PATH):
                logger.error(f"Fine-tuned YAMNet model not found: {Config.YAMNET_MODEL_PATH}")
                return
            
            # Load fine-tuned TFLite model
            self.model = tf.lite.Interpreter(
                model_path=Config.YAMNET_MODEL_PATH,
                num_threads=Config.MAX_THREADS
            )
            self.model.allocate_tensors()
            
            # Get input details
            input_details = self.model.get_input_details()[0]
            self.input_length = input_details['shape'][1]
            
            logger.info(f"✅ Fine-tuned YAMNet ready. Input length: {self.input_length}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize YAMNet classifier: {e}")
            self.model = None
    
    def classify(self, waveform):
        """Classify audio as speech/non-speech"""
        if self.model is None:
            return {"speech": 0.0, "non_speech": 1.0, "is_speaking": False}
        
        try:
            # Preprocess audio
            if len(waveform) > self.input_length:
                waveform = waveform[:self.input_length]
            elif len(waveform) < self.input_length:
                waveform = np.pad(waveform, (0, self.input_length - len(waveform)))
            
            # Normalize
            waveform = waveform.astype(np.float32)
            input_data = np.expand_dims(waveform, axis=0)
            
            # Run inference
            self.model.set_tensor(self.model.get_input_details()[0]['index'], input_data)
            self.model.invoke()
            
            # Get output
            scores = self.model.get_tensor(self.model.get_output_details()[0]['index'])
            
            if scores.ndim == 3:  # [batch, time, classes]
                scores = scores[0].mean(axis=0)
            elif scores.ndim == 2:  # [batch, classes]
                scores = scores[0]
            
            # Convert to speech/non-speech probabilities
            if len(scores) == 2:
                non_speech_prob = float(scores[0])
                speech_prob = float(scores[1])
            else:
                # If original YAMNet (521 classes), sum speech-related classes
                speech_indices = [0, 1, 2, 3, 4, 5]  # Approximate speech classes
                speech_prob = float(np.sum(scores[speech_indices]))
                non_speech_prob = 1.0 - speech_prob
            
            is_speaking = speech_prob > Config.SPEECH_THRESHOLD
            
            return {
                "speech": speech_prob,
                "non_speech": non_speech_prob,
                "is_speaking": is_speaking
            }
            
        except Exception as e:
            logger.error(f"❌ Audio classification error: {e}")
            return {"speech": 0.0, "non_speech": 1.0, "is_speaking": False}

# -------------------- FACE MESH DETECTOR --------------------
class FaceMeshDetector:
    """Face landmark detection for lip distance measurement"""
    
    def __init__(self):
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize FaceMesh model"""
        if not tf or not os.path.exists(Config.FACEMESH_PATH):
            logger.error("FaceMesh model not available")
            return
        
        try:
            self.model = tf.lite.Interpreter(
                model_path=Config.FACEMESH_PATH,
                num_threads=Config.MAX_THREADS
            )
            self.model.allocate_tensors()
            logger.info("✅ FaceMesh detector ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FaceMesh: {e}")
            self.model = None
    
    def detect_landmarks(self, frame):
        """Detect face landmarks"""
        if self.model is None:
            return None
        
        try:
            # Preprocess
            input_details = self.model.get_input_details()[0]
            input_size = input_details['shape'][1]
            
            resized = cv2.resize(frame, (input_size, input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
            
            # Run inference
            self.model.set_tensor(input_details['index'], input_data)
            self.model.invoke()
            
            # Get landmarks
            output_details = self.model.get_output_details()[0]
            landmarks = self.model.get_tensor(output_details['index'])
            
            return landmarks[0] if landmarks.shape[0] > 0 else None
            
        except Exception as e:
            logger.error(f"❌ FaceMesh detection error: {e}")
            return None
    
    def calculate_lip_distance(self, landmarks, frame_shape):
        """Calculate mouth opening distance"""
        if landmarks is None:
            return 0.0
        
        try:
            h, w = frame_shape[:2]
            
            # MediaPipe face mesh lip indices
            # Upper lip: points around 12, 15
            # Lower lip: points around 16, 17
            upper_lip_idx = 12
            lower_lip_idx = 16
            
            if len(landmarks) > max(upper_lip_idx, lower_lip_idx):
                upper_lip = landmarks[upper_lip_idx]
                lower_lip = landmarks[lower_lip_idx]
                
                # Convert normalized coordinates to pixels
                upper_lip_px = (upper_lip[0] * w, upper_lip[1] * h)
                lower_lip_px = (lower_lip[0] * w, lower_lip[1] * h)
                
                # Calculate distance
                distance = math.dist(upper_lip_px, lower_lip_px)
                return distance
            
        except Exception as e:
            logger.error(f"❌ Lip distance calculation error: {e}")
        
        return 0.0

# -------------------- POSE DETECTOR --------------------
class PoseDetector:
    """Pose detection for BP cuff placement guidance"""
    
    def __init__(self):
        self.model = None
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self._initialize()
    
    def _initialize(self):
        """Initialize pose model"""
        if not tf or not os.path.exists(Config.POSE_PATH):
            logger.error("Pose model not available")
            return
        
        try:
            self.model = tf.lite.Interpreter(
                model_path=Config.POSE_PATH,
                num_threads=Config.MAX_THREADS
            )
            self.model.allocate_tensors()
            logger.info("✅ Pose detector ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pose detector: {e}")
            self.model = None
    
    def detect_keypoints(self, frame):
        """Detect pose keypoints"""
        if self.model is None:
            return None
        
        try:
            # Preprocess
            input_details = self.model.get_input_details()[0]
            input_size = input_details['shape'][1]
            
            resized = cv2.resize(frame, (input_size, input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
            
            # Run inference
            self.model.set_tensor(input_details['index'], input_data)
            self.model.invoke()
            
            # Get keypoints
            output_details = self.model.get_output_details()[0]
            keypoints = self.model.get_tensor(output_details['index'])
            
            return keypoints[0]  # Shape: [17, 3] - (y, x, confidence)
            
        except Exception as e:
            logger.error(f"❌ Pose detection error: {e}")
            return None

# -------------------- UTILITY FUNCTIONS --------------------
def rotate_point(center, angle, point):
    """Rotate point around center by angle"""
    ox, oy = center
    px, py = point
    s = math.sin(angle)
    c = math.cos(angle)
    px -= ox
    py -= oy
    xnew = px * c - py * s
    ynew = px * s + py * c
    return int(xnew + ox), int(ynew + oy)

def draw_bp_cuff_guide(frame, keypoints, confidence_threshold=0.3):
    """Draw BP cuff placement guide with proper positioning"""
    if keypoints is None:
        return False
    
    h, w = frame.shape[:2]
    
    # Get left arm keypoints (indices: 5=left_shoulder, 7=left_elbow, 9=left_wrist)
    left_shoulder_idx, left_elbow_idx, left_wrist_idx = 5, 7, 9
    
    if (len(keypoints) > max(left_shoulder_idx, left_elbow_idx, left_wrist_idx) and
        keypoints[left_shoulder_idx][2] > confidence_threshold and
        keypoints[left_elbow_idx][2] > confidence_threshold):
        
        # Convert normalized coordinates to pixels
        left_shoulder = (int(keypoints[left_shoulder_idx][1] * w), int(keypoints[left_shoulder_idx][0] * h))
        left_elbow = (int(keypoints[left_elbow_idx][1] * w), int(keypoints[left_elbow_idx][0] * h))
        
        # Calculate arm angle
        arm_vector = (left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1])
        arm_length = math.sqrt(arm_vector[0]**2 + arm_vector[1]**2)
        
        if arm_length > 50:  # Minimum arm length threshold
            angle = math.atan2(arm_vector[1], arm_vector[0])
            
            # Calculate BP cuff position (2 inches above elbow)
            offset_pixels = Config.BP_CUFF_OFFSET_INCHES * Config.PIXELS_PER_INCH
            
            # Move from elbow towards shoulder by offset distance
            unit_vector = (arm_vector[0] / arm_length, arm_vector[1] / arm_length)
            cuff_center = (
                int(left_elbow[0] - unit_vector[0] * offset_pixels),
                int(left_elbow[1] - unit_vector[1] * offset_pixels)
            )
            
            # Calculate cuff dimensions
            cuff_width = int(arm_length * 0.4)  # Width based on arm length
            cuff_height = int(cuff_width * 0.3)  # Height proportional to width
            
            # Create rotated rectangle for cuff
            rect_points = [
                (-cuff_width//2, -cuff_height//2),
                (cuff_width//2, -cuff_height//2),
                (cuff_width//2, cuff_height//2),
                (-cuff_width//2, cuff_height//2)
            ]
            
            # Rotate rectangle
            rotated_points = [rotate_point((0, 0), angle, p) for p in rect_points]
            final_points = [(cuff_center[0] + p[0], cuff_center[1] + p[1]) for p in rotated_points]
            
            # Convert to numpy array for drawing
            pts = np.array(final_points, dtype=np.int32)
            
            # Draw cuff area with transparency effect
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))  # Green fill
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw cuff border
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Draw measurement line (2 inches above elbow)
            line_start = (
                int(left_elbow[0] - unit_vector[0] * (offset_pixels + 20)),
                int(left_elbow[1] - unit_vector[1] * (offset_pixels + 20))
            )
            line_end = (
                int(left_elbow[0] - unit_vector[0] * (offset_pixels - 20)),
                int(left_elbow[1] - unit_vector[1] * (offset_pixels - 20))
            )
            cv2.line(frame, line_start, line_end, (255, 255, 0), 2)
            
            # Add labels
            label_pos = (cuff_center[0] - 60, cuff_center[1] - cuff_height//2 - 15)
            cv2.putText(frame, "BP CUFF ZONE", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Distance label
            distance_label = f"2\" above elbow"
            distance_pos = (cuff_center[0] - 50, cuff_center[1] + cuff_height//2 + 25)
            cv2.putText(frame, distance_label, distance_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw keypoints
            cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
            cv2.circle(frame, left_elbow, 5, (0, 0, 255), -1)
            cv2.line(frame, left_shoulder, left_elbow, (255, 255, 255), 2)
            
            return True
    
    return False

# -------------------- AUDIO PROCESSING --------------------
def audio_callback(indata, frames, time_info, status):
    """Audio stream callback"""
    if status:
        logger.warning(f"Audio status: {status}")
    
    with state.lock:
        state.audio_buffer = np.append(state.audio_buffer, indata[:, 0])
        if len(state.audio_buffer) > Config.MAX_AUDIO_BUFFER:
            state.audio_buffer = state.audio_buffer[-Config.MAX_AUDIO_BUFFER:]

def audio_processing_thread():
    """Background audio processing for speech detection"""
    if not AUDIO_AVAILABLE:
        logger.warning("Audio processing disabled - sounddevice not available")
        return
    
    try:
        classifier = FineTunedYAMNetClassifier()
        if classifier.model is None:
            logger.error("YAMNet classifier initialization failed")
            return
        
        # Initialize audio stream
        stream = None
        for sample_rate in [Config.SAMPLE_RATE, 44100, 48000]:
            try:
                stream = sd.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=audio_callback,
                    blocksize=1024
                )
                stream.start()
                logger.info(f"🔊 Audio stream started at {sample_rate}Hz")
                break
            except Exception as e:
                logger.warning(f"Failed to start audio at {sample_rate}Hz: {e}")
        
        if stream is None:
            logger.error("Could not start audio stream")
            return
        
        # Processing loop
        buffer_size = int(Config.SAMPLE_RATE * Config.BUFFER_DURATION)
        last_process_time = time.time()
        
        while state.is_running:
            try:
                current_time = time.time()
                
                # Throttle processing
                if current_time - last_process_time < Config.AUDIO_PROCESSING_INTERVAL:
                    time.sleep(0.1)
                    continue
                
                # Get audio data
                waveform = None
                with state.lock:
                    if len(state.audio_buffer) >= buffer_size:
                        waveform = state.audio_buffer[:buffer_size].copy()
                        overlap = buffer_size // 4
                        state.audio_buffer = state.audio_buffer[buffer_size - overlap:]
                
                # Classify speech
                if waveform is not None:
                    classification = classifier.classify(waveform)
                    with state.lock:
                        state.last_speech_classification = classification
                
                last_process_time = current_time
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(1)
        
        # Cleanup
        if stream:
            stream.stop()
            stream.close()
            
    except Exception as e:
        logger.error(f"Audio thread error: {e}")

# -------------------- CAMERA INITIALIZATION --------------------
def initialize_camera():
    """Initialize camera with fallback options"""
    for camera_id in range(3):
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = cap.read()
                if ret:
                    logger.info(f"📹 Camera {camera_id} initialized successfully")
                    state.camera = cap
                    return True
                else:
                    cap.release()
            
        except Exception as e:
            logger.warning(f"Camera {camera_id} failed: {e}")
    
    logger.error("No working camera found")
    return False

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

# Initialize components
logger.info("🚀 Initializing SmartBP Health Monitor...")
download_models()
camera_available = initialize_camera()

# Initialize detectors
face_detector = FaceMeshDetector()
pose_detector = PoseDetector()

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SmartBP Health Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f8ff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6; }
        .status { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .status-item { text-align: center; padding: 15px; background: #f8fafc; border-radius: 8px; }
        .status-ok { color: #10b981; font-weight: bold; }
        .status-error { color: #ef4444; font-weight: bold; }
        .video-container { text-align: center; background: #1f2937; padding: 20px; border-radius: 12px; }
        #videoFeed { max-width: 100%; height: auto; border: 3px solid #3b82f6; border-radius: 8px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 15px; }
        .metric { text-align: center; padding: 10px; background: #f1f5f9; border-radius: 6px; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #1e40af; }
        .btn { background: #3b82f6; color: white; border: none; padding: 12px 24px; 
               border-radius: 6px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #2563eb; }
        h1 { color: #1e40af; text-align: center; margin-bottom: 30px; }
        h2 { color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 SmartBP Health Monitor</h1>
        
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="status">
                <div class="status-item">
                    <div>📹 Camera</div>
                    <div class="{{ 'status-ok' if camera_available else 'status-error' }}">
                        {{ '✅ Active' if camera_available else '❌ Not Available' }}
                    </div>
                </div>
                <div class="status-item">
                    <div>🎤 Speech Detection</div>
                    <div class="{{ 'status-ok' if audio_available else 'status-error' }}">
                        {{ '✅ YAMNet Fine-tuned' if audio_available else '❌ Disabled' }}
                    </div>
                </div>
                <div class="status-item">
                    <div>👄 Face Mesh</div>
                    <div class="status-ok">✅ Active</div>
                </div>
                <div class="status-item">
                    <div>🤸 Pose Detection</div>
                    <div class="status-ok">✅ MoveNet</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📹 Live Monitoring</h2>
            <div class="video-container">
                {% if camera_available %}
                <img id="videoFeed" src="/video_feed" alt="Live video feed">
                {% else %}
                <div style="color: white; padding: 50px;">Camera not available</div>
                {% endif %}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div>🗣️ Speech</div>
                    <div class="metric-value" id="speechStatus">--</div>
                </div>
                <div class="metric">
                    <div>👄 Lip Distance</div>
                    <div class="metric-value" id="lipDistance">--</div>
                </div>
                <div class="metric">
                    <div>💪 BP Cuff</div>
                    <div class="metric-value" id="bpStatus">--</div>
                </div>
                <div class="metric">
                    <div>⚡ FPS</div>
                    <div class="metric-value" id="fpsValue">--</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🔧 Controls</h2>
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="updateStatus()">📊 Update Status</button>
            <button class="btn" onclick="window.open('/status', '_blank')">📋 JSON Status</button>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('speechStatus').textContent = 
                        data.speech_classification?.is_speaking ? 'Speaking' : 'Silent';
                    document.getElementById('lipDistance').textContent = 
                        data.lip_distance?.toFixed(1) + 'px';
                    document.getElementById('bpStatus').textContent = 
                        data.bp_cuff_valid ? 'Valid' : 'Invalid';
                    document.getElementById('fpsValue').textContent = 
                        data.fps?.toFixed(1);
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Auto update every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus(); // Initial update
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    """Main web interface"""
    return render_template_string(HTML_TEMPLATE,
        camera_available=camera_available,
        audio_available=AUDIO_AVAILABLE
    )

def generate_frames():
    """Generate video frames with all detections"""
    if not camera_available:
        placeholder = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Available", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        
        while state.is_running:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1 / Config.TARGET_FPS)
        return

    while state.is_running and state.camera and state.camera.isOpened():
        ret, frame = state.camera.read()
        if not ret:
            break

        # FPS calculation
        state.frame_count += 1
        current_time = time.time()
        if current_time - state.last_fps_time >= 1.0:
            state.current_fps = state.frame_count / (current_time - state.last_fps_time)
            state.frame_count = 0
            state.last_fps_time = current_time

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process detections (skip frames for performance)
        if state.frame_count % Config.FRAME_SKIP == 0:
            # Pose detection and BP cuff guide
            try:
                keypoints = pose_detector.detect_keypoints(frame)
                state.bp_cuff_valid = draw_bp_cuff_guide(frame, keypoints)
            except Exception as e:
                logger.error(f"Pose detection error: {e}")
                state.bp_cuff_valid = False
            
            # Face mesh and lip detection
            try:
                landmarks = face_detector.detect_landmarks(frame)
                state.lip_distance = face_detector.calculate_lip_distance(landmarks, frame.shape)
                state.is_talking_face = state.lip_distance > 8.0  # Threshold for talking
            except Exception as e:
                logger.error(f"Face detection error: {e}")
                state.lip_distance = 0.0
                state.is_talking_face = False

        # Display information overlay
        y_offset = 30
        
        # FPS
        cv2.putText(frame, f"FPS: {state.current_fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

        # Speech detection (from audio)
        with state.lock:
            speech_data = state.last_speech_classification
        
        speech_text = f"Speech: {speech_data['speech']*100:.1f}%"
        speech_color = (0, 255, 0) if speech_data['is_speaking'] else (255, 255, 255)
        cv2.putText(frame, speech_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, speech_color, 2)
        y_offset += 30

        # Lip distance (from face)
        lip_text = f"Lip Distance: {state.lip_distance:.1f}px"
        lip_color = (0, 255, 0) if state.is_talking_face else (255, 255, 255)
        cv2.putText(frame, lip_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lip_color, 2)
        y_offset += 30

        # Combined talking detection
        is_talking_combined = speech_data['is_speaking'] or state.is_talking_face
        if is_talking_combined:
            cv2.putText(frame, "🗣️ TALKING DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        y_offset += 35

        # BP cuff status
        if state.bp_cuff_valid:
            cv2.putText(frame, "✅ BP Cuff Position Valid", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "❌ Adjust Arm Position", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Encode frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    """System status API"""
    with state.lock:
        speech_classification = state.last_speech_classification.copy()
    
    return jsonify({
        'camera': camera_available,
        'audio': AUDIO_AVAILABLE,
        'fps': state.current_fps,
        'speech_classification': speech_classification,
        'lip_distance': state.lip_distance,
        'is_talking_face': state.is_talking_face,
        'bp_cuff_valid': state.bp_cuff_valid,
        'models': {
            'yamnet': os.path.exists(Config.YAMNET_MODEL_PATH),
            'facemesh': os.path.exists(Config.FACEMESH_PATH),
            'pose': os.path.exists(Config.POSE_PATH)
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - state.last_fps_time
    })

# -------------------- SIGNAL HANDLING --------------------
def signal_handler(signum, frame):
    """Graceful shutdown"""
    logger.info("Received shutdown signal")
    state.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    logger.info("🚀 Starting SmartBP Health Monitor...")
    
    # Start audio processing for speech detection
    if AUDIO_AVAILABLE:
        audio_thread = threading.Thread(target=audio_processing_thread, daemon=True)
        audio_thread.start()
        logger.info("🔊 Audio processing thread started")
    
    try:
        logger.info("🌐 Starting web server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("👋 Shutting down...")
        state.stop()