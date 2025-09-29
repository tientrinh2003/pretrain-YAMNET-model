import os
import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import csv
from urllib.request import urlretrieve
from flask import Flask, Response
from flask import Flask, Response, jsonify
from flask_cors import CORS
from tensorflow.keras.utils import get_file

# Keras 3: TFSMLayer để load SavedModel inference-only
try:
    from keras.layers import TFSMLayer
except Exception:
    TFSMLayer = None

# -------------------- CẤU HÌNH --------------------
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Đường dẫn model (raw string để tránh \I bị escape)
SPEECH_SAVEDMODEL_PATH = r"D:\IU\THESIS\pretrain\yamnet_sns_savedmodel"  # SavedModel đã train (wrapper)
YAMNET_SAVEDMODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_TFLITE_URL = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"

MODEL_DIR = "model_data"
TFLITE_PATH = os.path.join(MODEL_DIR, "yamnet.tflite")

SAMPLE_RATE = 16000
BUFFER_DURATION_SECONDS = 2.0  # nên khớp đúng 2.0s như lúc train
LIP_DISTANCE_THRESHOLD = 5.0
HUMAN_SOUNDS = ["Speech", "Singing", "Yell", "Screaming", "Laughter", "Whispering", "Crying, sobbing"]
SOUND_CONFIDENCE_THRESHOLD = 0.5  # hợp lý cho binary speech model

# Biến toàn cục
audio_buffer = np.array([], dtype=np.float32)
last_sound_classification = [("Listening...", 0.0)]
is_running = True
lock = threading.Lock()

# -------------------- AUDIO CLASSIFIER --------------------
class AudioClassifier:
    """
    Ưu tiên dùng model speech/non-speech (SavedModel) qua Keras 3 TFSMLayer.
    Fallback: YAMNet SavedModel (TFHub), rồi tới YAMNet TFLite.
    """
    def __init__(self):
        self.mode = None
        self.model = None
        self.required_len = None  # số mẫu waveform mà model cần (vd 32000)
        self.input_details = None
        self.output_details = None
        self.class_names = None
        self.sns_layer = None  # TFSMLayer cho SavedModel nhị phân
        self._init_model()

    def _load_class_map(self):
        # Class map của YAMNet
        os.makedirs(MODEL_DIR, exist_ok=True)
        class_map_path = get_file(
            fname='yamnet_class_map.csv',
            origin='https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
            cache_dir='.', cache_subdir=MODEL_DIR
        )
        names = []
        with open(class_map_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                names.append(row[2])
        return names

    def _init_model(self):
        # 1) Thử load SavedModel speech/non-speech (wrapper) bằng TFSMLayer (Keras 3)
        try:
            if os.path.exists(SPEECH_SAVEDMODEL_PATH):
                if TFSMLayer is None:
                    raise RuntimeError("Keras 3 TFSMLayer chưa khả dụng. Cài đặt: pip install -U keras")
                print(f"[Audio] Đang tải Speech SavedModel (TFSMLayer) từ: {SPEECH_SAVEDMODEL_PATH} ...")
                self.sns_layer = TFSMLayer(SPEECH_SAVEDMODEL_PATH, call_endpoint='serving_default')
                self.mode = "sns"  # speech/non-speech
                self.class_names = ["Non-Speech", "Speech"]
                # Model wrapper của bạn dùng 2.0s @ 16kHz => 32000 mẫu
                self.required_len = int(2.0 * SAMPLE_RATE)
                print(f"[Audio] Đã sẵn sàng Speech model (TFSMLayer). required_len={self.required_len}")
                return
            else:
                print(f"[Audio] Không tìm thấy {SPEECH_SAVEDMODEL_PATH}, sẽ thử YAMNet...")
        except Exception as e:
            print(f"[Audio] Không load được Speech SavedModel (TFSMLayer): {e}")

        # 2) TFHub YAMNet SavedModel
        print("[Audio] Đang tải YAMNet (TFHub SavedModel)...")
        try:
            self.model = hub.load(YAMNET_SAVEDMODEL_URL)
            self.mode = "hub"
            self.class_names = self._load_class_map()
            print("[Audio] YAMNet (SavedModel) đã sẵn sàng.")
            return
        except Exception as e:
            print(f"[Audio] Không load được SavedModel từ TFHub: {e}")

        # 3) TFLite
        print("[Audio] Thử chuyển sang YAMNet TFLite...")
        try:
            if not os.path.exists(TFLITE_PATH):
                os.makedirs(MODEL_DIR, exist_ok=True)
                print(f"[Audio] Đang tải TFLite model tới {TFLITE_PATH} ...")
                urlretrieve(YAMNET_TFLITE_URL, TFLITE_PATH)

            self.model = tf.lite.Interpreter(model_path=TFLITE_PATH)
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            self.mode = "tflite"
            # yêu cầu độ dài đầu vào của TFLite (nếu có)
            self.required_len = None
            inshape = self.input_details[0]["shape"]
            if inshape is not None and len(inshape) == 2 and inshape[1] > 0:
                self.required_len = int(inshape[1])
            self.class_names = self._load_class_map()
            print("[Audio] YAMNet (TFLite) đã sẵn sàng.")
        except Exception as e:
            raise RuntimeError(f"[Audio] Không thể khởi tạo YAMNet TFLite: {e}")

    def _pad_crop(self, waveform: np.ndarray):
        if self.required_len is None:
            return waveform
        if len(waveform) < self.required_len:
            pad = self.required_len - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")
        elif len(waveform) > self.required_len:
            waveform = waveform[:self.required_len]
        return waveform

    # Modes
    def _classify_sns(self, waveform: np.ndarray):
        # Binary speech: trả về vector [non_speech, speech]
        waveform = self._pad_crop(waveform)
        x = waveform.reshape(1, -1).astype(np.float32)
        outs = self.sns_layer(x)  # dict hoặc tensor tùy signature
        if isinstance(outs, dict):
            # lấy output đầu tiên
            y = next(iter(outs.values()))
        else:
            y = outs
        prob = float(tf.convert_to_tensor(y).numpy().reshape(-1)[0])
        return np.array([1.0 - prob, prob], dtype=np.float32)

    def _classify_hub(self, waveform_tf: tf.Tensor):
        scores, embeddings, spectrogram = self.model(waveform_tf)  # [frames, classes]
        return scores.numpy().mean(axis=0)

    def _classify_tflite(self, waveform: np.ndarray):
        waveform = self._pad_crop(waveform)
        x = np.expand_dims(waveform.astype(np.float32), axis=0)
        self.model.set_tensor(self.input_details[0]["index"], x)
        self.model.invoke()
        out0 = self.model.get_tensor(self.output_details[0]["index"])
        scores = out0
        if scores.ndim == 3:
            scores = scores[0]
        if scores.ndim == 2:
            scores = scores.mean(axis=0)
        elif scores.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected TFLite output shape: {out0.shape}")
        return scores

    def classify(self, waveform):
        # Chuẩn hóa dtype
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform, dtype=np.float32)
        elif waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        if self.mode == "sns":
            return self._classify_sns(waveform)
        elif self.mode == "hub":
            wf = tf.convert_to_tensor(waveform, dtype=tf.float32)
            return self._classify_hub(wf)
        elif self.mode == "tflite":
            return self._classify_tflite(waveform)
        else:
            raise RuntimeError("AudioClassifier chưa được khởi tạo đúng cách.")

# -------------------- AUDIO PIPELINE --------------------
def load_yamnet_model():
    try:
        clf = AudioClassifier()
        return clf, clf.class_names
    except Exception as e:
        print(f"[Audio] Lỗi khởi tạo AudioClassifier: {e}")
        return None, None

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        pass
    with lock:
        audio_buffer = np.append(audio_buffer, indata[:, 0])

def audio_processing_thread_func():
    global last_sound_classification, audio_buffer, is_running
    clf, class_names = load_yamnet_model()
    if clf is None:
        is_running = False
        return

    buffer_size = int(SAMPLE_RATE * BUFFER_DURATION_SECONDS)

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
        stream.start()
        print("[Audio] Đã bắt đầu thu âm...")
    except Exception as e:
        print(f"[Audio] Không thể mở microphone: {e}")
        is_running = False
        return

    while is_running:
        waveform = None
        with lock:
            if len(audio_buffer) >= buffer_size:
                waveform = audio_buffer[:buffer_size].copy()
                audio_buffer = audio_buffer[buffer_size:]

        if waveform is not None:
            try:
                scores_np = clf.classify(waveform)  # np.array shape: [num_classes] (2 nếu dùng speech model)
                # Lấy Top-5 (hoặc ít hơn nếu binary)
                top_k = min(5, len(scores_np))
                top_indices = np.argsort(scores_np)[::-1][:top_k]
                top_classes = [(class_names[i], float(scores_np[i])) for i in top_indices]
                with lock:
                    last_sound_classification = top_classes
            except Exception as e:
                print(f"[Audio] Lỗi phân loại: {e}")

        cv2.waitKey(50)

    try:
        stream.stop()
        stream.close()
    except:
        pass
    print("[Audio] Đã dừng luồng âm thanh.")

# -------------------- STREAMING --------------------
app = Flask(__name__)
CORS(app)
cap = cv2.VideoCapture(0)

@app.route("/")
def index():
    # Trang index đơn giản để không 404
    return "<html><body><h3>Video feed</h3><img src='/video_feed' /></body></html>"

def generate_frames():
    global is_running, last_sound_classification

    with mp.solutions.pose.Pose() as pose, \
         mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            h, w, _ = frame.shape
            if pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

            is_talking = False
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    )

                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    lip_distance = math.dist(
                        (upper_lip.x * w, upper_lip.y * h),
                        (lower_lip.x * w, lower_lip.y * h)
                    )

                    with lock:
                        top_sounds = last_sound_classification

                    top_sound_name, top_sound_prob = top_sounds[0] if len(top_sounds) > 0 else ("", 0.0)
                    if (
                        lip_distance > LIP_DISTANCE_THRESHOLD and
                        top_sound_name in HUMAN_SOUNDS and
                        top_sound_prob > SOUND_CONFIDENCE_THRESHOLD
                    ):
                        is_talking = True

                    cv2.putText(frame, f"Lip dist: {lip_distance:.2f}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    y_offset = 80
                    for cls_name, prob in top_sounds:
                        text = f"{cls_name}: {prob * 100:.1f}%"
                        color = (0, 255, 0)
                        if (
                            cls_name == top_sound_name and
                            cls_name in HUMAN_SOUNDS and
                            prob > SOUND_CONFIDENCE_THRESHOLD
                        ):
                            color = (255, 255, 0)
                        cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 30

            if is_talking:
                cv2.putText(frame, "TALKING", (w // 2 - 100, h - 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_processing_thread_func, daemon=True)
    audio_thread.start()
    app.run(host="0.0.0.0", port=5000)