# SmartBP Health Monitor - Raspberry Pi 5 Integration

Tích hợp hoàn chỉnh YAMNet fine-tuned, FaceMesh, và Pose Detection cho đeo vòng đo huyết áp đúng vị trí.

## ✨ Tính năng chính

### 🎤 Speech Detection (YAMNet Fine-tuned)
- Tự động extract và sử dụng model YAMNet đã fine-tuned từ file ZIP
- Phân loại real-time speech/non-speech 
- Tối ưu hóa cho Raspberry Pi 5 ARM64

### 👄 Face Mesh Lip Detection
- Phát hiện khoảng cách 2 môi để xác định mở miệng
- Kết hợp với YAMNet để xác nhận người đang nói chuyện
- Sử dụng TensorFlow Lite cho hiệu suất cao

### 💪 BP Cuff Positioning Guide
- Phát hiện pose và hiển thị vùng đeo vòng huyết áp đúng vị trí
- Tự động tính toán vị trí 2 inch trên khuỷu tay
- Hiển thị hình chữ nhật xoay theo góc cánh tay
- Đường gạch ngang chỉ khoảng cách chính xác

## 🚀 Cài đặt nhanh

### 1. Chuẩn bị Raspberry Pi 5
```bash
# Copy files to Raspberry Pi
scp smartbp_health_monitor.py pi@your-pi-ip:/home/pi/
scp yamnet_finetuned_model.zip pi@your-pi-ip:/home/pi/
scp setup_raspberry_pi.sh pi@your-pi-ip:/home/pi/
scp requirements.txt pi@your-pi-ip:/home/pi/
```

### 2. Chạy script setup
```bash
ssh pi@your-pi-ip
chmod +x setup_raspberry_pi.sh
./setup_raspberry_pi.sh
```

### 3. Khởi động monitor
```bash
cd /home/pi/smartbp_monitor
./start_monitor.sh
```

### 4. Truy cập web interface
Mở browser: `http://your-pi-ip:5000`

## 📁 Cấu trúc project

```
/home/pi/smartbp_monitor/
├── smartbp_health_monitor.py     # Main application
├── yamnet_finetuned_model.zip    # Your fine-tuned model
├── requirements.txt              # Python dependencies
├── start_monitor.sh             # Startup script
├── models/                      # Auto-downloaded models
│   ├── yamnet_finetuned.tflite
│   ├── face_landmark.tflite
│   └── movenet_lightning.tflite
└── venv/                        # Python virtual environment
```

## 🔧 Cấu hình

### Audio Settings
```python
SAMPLE_RATE = 16000              # YAMNet sample rate
SPEECH_THRESHOLD = 0.6           # Speech confidence threshold
AUDIO_PROCESSING_INTERVAL = 0.5  # Processing interval
```

### Camera Settings
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 20
FRAME_SKIP = 2                   # Skip frames for performance
```

### BP Cuff Settings
```python
BP_CUFF_OFFSET_INCHES = 2.0      # 2 inches above elbow
PIXELS_PER_INCH = 30             # Camera distance calibration
```

## 🎯 Cách sử dụng

### 1. Speech Detection
- **YAMNet Fine-tuned**: Tự động phân loại âm thanh thành speech/non-speech
- **Real-time**: Xử lý liên tục với buffer audio
- **Hiển thị**: Phần trăm confidence và trạng thái speaking

### 2. Talking Detection
- **Face Mesh**: Đo khoảng cách 2 môi để phát hiện mở miệng
- **Combined Logic**: Kết hợp audio + visual để xác nhận chính xác
- **Threshold**: Lip distance > 8px = đang nói

### 3. BP Cuff Guide
- **Pose Detection**: Phát hiện vai, khuỷu tay, cổ tay
- **Auto Calculation**: Tự động tính vị trí 2 inch trên khuỷu tay
- **Visual Guide**: Hình chữ nhật xoay + đường kẻ measurement
- **Validation**: Kiểm tra tư thế hợp lệ để đo huyết áp

## 📊 Web Interface

### Dashboard Features
- **Live Video Feed**: Camera stream với tất cả annotations
- **Real-time Metrics**: Speech %, Lip distance, BP cuff status, FPS
- **System Status**: Camera, Audio, Models status
- **Controls**: Refresh, Update, JSON API access

### API Endpoints
- `GET /`: Main dashboard
- `GET /video_feed`: Video stream
- `GET /status`: JSON status data
- `GET /health`: Health check

## 🛠️ Troubleshooting

### Audio Issues
```bash
# Check audio devices
aplay -l
arecord -l

# Test microphone
arecord -d 5 test.wav
aplay test.wav
```

### Camera Issues
```bash
# List cameras
v4l2-ctl --list-devices

# Test camera
libcamera-hello --timeout 5000
```

### Model Issues
```bash
# Check model files
ls -la models/
file models/*.tflite
```

### Performance Issues
```bash
# Monitor system resources
htop
iotop

# Check GPU memory
vcgencmd get_mem gpu
```

## 🔄 Auto-start Service

### Enable service
```bash
sudo systemctl enable smartbp-monitor
sudo systemctl start smartbp-monitor
```

### Check status
```bash
sudo systemctl status smartbp-monitor
sudo journalctl -u smartbp-monitor -f
```

## 📝 Logs

Logs được lưu tại:
- **Application**: `/home/pi/smartbp_monitor.log`
- **System**: `journalctl -u smartbp-monitor`

## 🔧 Customization

### Thay đổi model YAMNet
1. Thay thế file `yamnet_finetuned_model.zip`
2. Restart service hoặc application
3. Model sẽ được extract tự động

### Điều chỉnh thresholds
```python
# Speech detection
Config.SPEECH_THRESHOLD = 0.7  # Tăng để giảm false positive

# Lip detection  
lip_threshold = 10.0           # Tăng để giảm sensitivity

# BP cuff offset
Config.BP_CUFF_OFFSET_INCHES = 1.5  # Thay đổi khoảng cách
```

### Camera calibration
```python
# Điều chỉnh pixels per inch dựa trên khoảng cách camera
Config.PIXELS_PER_INCH = 35    # Camera gần hơn
Config.PIXELS_PER_INCH = 25    # Camera xa hơn
```

## 🎉 Kết quả mong đợi

1. **Speech Detection**: Accuracy > 90% với model fine-tuned
2. **Talking Detection**: Combined accuracy > 95%
3. **BP Cuff Guide**: Chính xác trong 90% cases với pose tốt
4. **Performance**: 15-20 FPS trên Raspberry Pi 5
5. **Latency**: < 500ms cho tất cả detections

## 🆘 Support

Nếu gặp issues:
1. Check logs: `tail -f /home/pi/smartbp_monitor.log`
2. Check system: `sudo systemctl status smartbp-monitor`
3. Verify models: `ls -la models/`
4. Test components riêng biệt

Perfect integration cho hệ thống đo huyết áp thông minh! 🩺✨