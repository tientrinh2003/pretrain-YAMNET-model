# Dataset Setup Instructions

The datasets and trained models are not included in this repository due to their large size. Follow these steps to set them up:

## Option 1: Download and Prepare Dataset (Recommended)

Run the preparation script to automatically download and process the datasets:

```bash
python prepare_yamnet_speech_nonspeech.py
```

This will:
- Download LibriSpeech test-clean (~350MB)
- Download ESC-50 (~600MB) 
- Download UrbanSound8K (~6GB)
- Download MUSAN (~12GB)
- Process and create `yamnet_dataset/` folder with:
  - `X.npy` - Waveform data (2400 samples)
  - `y.npy` - Labels  
  - `X_yamnet_emb.npy` - YAMNet embeddings (created during training)

## Option 2: Train Model

After dataset preparation, train the model:

```bash
python finetune_yamnet_speech_nonspeech.py
```

This creates:
- `classifier_on_emb.h5` - Trained classifier (~6MB)
- `yamnet_sns_savedmodel/` - End-to-end SavedModel (~50MB)

## File Sizes (Approximate)
- **Total raw datasets**: ~19GB
- **Processed dataset**: ~1GB  
- **Trained models**: ~56MB
- **Repository code only**: <1MB

## Storage Requirements
- **Minimum**: 20GB free space for full setup
- **Just for running**: 1GB if you have pre-trained models

## Alternative: Use Pre-trained Models

If you have access to pre-trained models, place them as:
```
├── classifier_on_emb.h5
├── yamnet_sns_savedmodel/
└── yamnet_dataset/
    ├── X.npy
    ├── y.npy  
    └── X_yamnet_emb.npy
```

Then you can run the application directly:
```bash
python app.py
```