# YAMNet Speech/Non-Speech Detection System

A production-ready speech detection system using fine-tuned YAMNet with comprehensive health monitoring integration.

## ✨ Features
- 🎤 **Real-time speech/non-speech classification** using fine-tuned YAMNet
- 📹 **Computer vision integration** with MediaPipe pose estimation  
- 🔄 **Flask web application** with live audio/video feeds
- 🤖 **High-performance model** with 100% validation accuracy
- 📊 **Balanced dataset** with 2400 samples (1200 speech + 1200 non-speech)
- 🏥 **Health monitoring** integration for SmartBP system

## 📊 Model Performance
- **Training Accuracy:** 99.78%
- **Validation Accuracy:** 100%
- **Architecture:** YAMNet (frozen) + Custom Dense Head
- **Input Format:** 2-second audio clips at 16kHz (32,000 samples)
- **Model Size:** ~17MB SavedModel

## 🗂️ Dataset Sources
- **Speech:** LibriSpeech test-clean subset
- **Non-Speech:** ESC-50, UrbanSound8K, MUSAN (music/noise/environmental)

## Quick Start

### 1. Setup Environment
```bash
git clone <your-repo-url>
cd pretrain
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac  
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Application
```bash
python smartbp_health_monitor.py
```
Open browser: `http://localhost:5000`

### 3. Use Pre-trained Model
The repository includes:
- `classifier_on_emb.h5` - Trained classifier on embeddings
- `yamnet_sns_savedmodel/` - End-to-end SavedModel for deployment
- `yamnet_dataset/` - Processed dataset (X.npy, y.npy, X_yamnet_emb.npy)

## Advanced Usage

### Complete Pipeline
```bash
python run_pipeline.py              # Run data preparation + training
python run_pipeline.py --step data  # Data preparation only
python run_pipeline.py --step train # Training only
```

### Manual Steps
#### Prepare New Dataset
```bash
python prepare_yamnet_speech_nonspeech.py
```
This downloads and processes:
- LibriSpeech test-clean (speech)
- ESC-50 (environmental sounds)  
- UrbanSound8K (urban sounds)
- MUSAN (music/noise)

#### Retrain Model
```bash
python finetune_yamnet_speech_nonspeech.py
```

## File Structure
```
├── smartbp_health_monitor.py                 # Main Flask application with YAMNet
├── prepare_yamnet_speech_nonspeech.py        # Dataset preparation
├── finetune_yamnet_speech_nonspeech.py       # Model training
├── run_pipeline.py                           # Complete pipeline runner
├── requirements.txt                          # Dependencies
├── classifier_on_emb.h5                      # Trained classifier
├── yamnet_sns_savedmodel/                    # End-to-end SavedModel
├── yamnet_dataset/                           # Processed dataset
│   ├── X.npy                                # Waveform data
│   ├── y.npy                                # Labels
│   └── X_yamnet_emb.npy                     # Pre-computed embeddings
└── [dataset folders]/                        # Raw audio datasets
```

## Technical Details

### Model Architecture
1. **YAMNet Feature Extraction**: Input waveform → 1024-dim embeddings
2. **Classifier Head**: 
   - Dense(512, ReLU) 
   - Dropout(0.3)
   - Dense(1, Sigmoid)

### Training Process
1. Extract YAMNet embeddings (cached)
2. Train classifier on embeddings (frozen YAMNet)
3. Create end-to-end wrapper model
4. Copy trained weights to wrapper

### Application Features
- Real-time audio processing with 2-second sliding window
- Lip movement detection using MediaPipe
- Combined audio + visual detection for "TALKING" state
- Flask web interface with live video feed

## Configuration
- **Sample Rate**: 16kHz
- **Clip Duration**: 2.0 seconds (32,000 samples)
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Speech Threshold**: 0.5

## Requirements
- Python 3.8+
- TensorFlow 2.12+
- OpenCV
- MediaPipe
- Flask
- NumPy, scikit-learn
- Audio: librosa, sounddevice

## License
Educational/Research purposes. Respect dataset licenses.