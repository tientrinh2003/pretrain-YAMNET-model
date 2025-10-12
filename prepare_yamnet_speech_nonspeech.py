import os
import zipfile
import tarfile
import numpy as np
import librosa
import argparse
import logging
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SAMPLE_RATE = 16000
CLIP_DURATION = 2.0  # seconds
CLIP_SAMPLES = int(CLIP_DURATION * SAMPLE_RATE)
OUT_DIR = "yamnet_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# Data quality thresholds
MIN_AUDIO_DURATION = 0.5  # seconds
MAX_AUDIO_DURATION = 60.0  # seconds
MIN_AMPLITUDE = 1e-6
MAX_SILENCE_RATIO = 0.95

def download(url: str, out_path: str) -> bool:
    """Download file with progress tracking and error handling."""
    if os.path.exists(out_path):
        logger.info(f"[✓] Already exists: {out_path}")
        return True
    
    try:
        logger.info(f"Downloading {url} ...")
        import urllib.request
        
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\rDownload progress: {percent:.1f}%", end="")
        
        urllib.request.urlretrieve(url, out_path, reporthook=show_progress)
        print()  # New line after progress
        logger.info("Download completed.")
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str, check_folder: Optional[str] = None) -> bool:
    """Extract ZIP file with error handling."""
    if check_folder and os.path.exists(check_folder):
        logger.info(f"[✓] Already extracted: {check_folder}")
        return True
    
    try:
        logger.info(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info("Extraction completed.")
        return True
    except Exception as e:
        logger.error(f"Extraction failed for {zip_path}: {e}")
        return False

def extract_tar(tar_path: str, extract_to: str, check_folder: Optional[str] = None) -> bool:
    """Extract TAR file with error handling."""
    if check_folder and os.path.exists(check_folder):
        logger.info(f"[✓] Already extracted: {check_folder}")
        return True
    
    try:
        logger.info(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(extract_to)
        logger.info("Extraction completed.")
        return True
    except Exception as e:
        logger.error(f"Extraction failed for {tar_path}: {e}")
        return False

######################
# LibriSpeech speech #
######################
def prepare_librispeech_speech(n_max=1200):
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    tar_path = "test-clean.tar.gz"
    extract_dir = "LibriSpeech/test-clean"
    download(url, tar_path)
    extract_tar(tar_path, ".", check_folder=extract_dir)
    speech_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".flac"):
                speech_files.append(os.path.join(root, f))
    np.random.shuffle(speech_files)
    return speech_files[:n_max]

#####################
# ESC-50 non-speech #
#####################
def prepare_esc50_non_speech(n_max=800):
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = "ESC-50.zip"
    extract_dir = "ESC-50-master"
    audio_folder = os.path.join(extract_dir, "audio")
    download(url, zip_path)
    extract_zip(zip_path, ".", check_folder=audio_folder)
    meta_file = os.path.join(extract_dir, "meta/esc50.csv")
    import csv
    non_speech_files = []
    with open(meta_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['category'] not in []:
                non_speech_files.append(os.path.join(extract_dir, "audio", row['filename']))
    np.random.shuffle(non_speech_files)
    return non_speech_files[:n_max]

#####################
# UrbanSound8K noise#
#####################
def prepare_urbansound8k_non_speech(n_max=800):
    url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    tar_path = "UrbanSound8K.tar.gz"
    extract_dir = "UrbanSound8K"
    audio_folder = os.path.join(extract_dir, "audio")
    download(url, tar_path)
    extract_tar(tar_path, ".", check_folder=audio_folder)
    meta_file = os.path.join(extract_dir, "metadata/UrbanSound8K.csv")
    import csv
    non_speech_files = []
    speech_labels = ["children_playing"]  # Skip this, rest are non-speech
    with open(meta_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['class'] not in speech_labels:
                path = os.path.join(extract_dir, "audio", f"fold{row['fold']}", row['slice_file_name'])
                non_speech_files.append(path)
    np.random.shuffle(non_speech_files)
    return non_speech_files[:n_max]

##################
# MUSAN noise    #
##################
def prepare_musan_noise(n_max=800):
    url = "http://www.openslr.org/resources/17/musan.tar.gz"
    tar_path = "musan.tar.gz"
    extract_dir = "musan"
    noise_folder = os.path.join(extract_dir, "noise")
    download(url, tar_path)
    extract_tar(tar_path, ".", check_folder=noise_folder)
    musan_noise = []
    for sub in ["music", "noise"]:
        base = os.path.join(extract_dir, sub)
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".wav", ".mp3")):
                    musan_noise.append(os.path.join(root, f))
    np.random.shuffle(musan_noise)
    return musan_noise[:n_max]

#####################
# Audio processing  #
#####################
def validate_audio_quality(audio_path: str) -> Tuple[bool, str]:
    """Validate audio file quality and return (is_valid, reason)."""
    try:
        # Basic file existence check
        if not os.path.exists(audio_path):
            return False, "File not found"
        
        # Try to load with soundfile first (faster)
        try:
            info = sf.info(audio_path)
            duration = info.duration
            sample_rate = info.samplerate
        except:
            # Fallback to librosa
            try:
                y_temp, sr_temp = librosa.load(audio_path, sr=None, duration=0.1)
                duration = librosa.get_duration(path=audio_path)
                sample_rate = sr_temp
            except Exception as e:
                return False, f"Cannot load audio: {e}"
        
        # Duration checks
        if duration < MIN_AUDIO_DURATION:
            return False, f"Too short: {duration:.2f}s"
        if duration > MAX_AUDIO_DURATION:
            return False, f"Too long: {duration:.2f}s"
        
        # Sample a small portion for quality check
        try:
            y_sample, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=1.0)
            
            # Check for silence
            rms = np.sqrt(np.mean(y_sample**2))
            if rms < MIN_AMPLITUDE:
                return False, f"Too quiet: RMS={rms:.2e}"
            
            # Check silence ratio
            silence_threshold = np.percentile(np.abs(y_sample), 10)
            silence_ratio = np.mean(np.abs(y_sample) < silence_threshold)
            if silence_ratio > MAX_SILENCE_RATIO:
                return False, f"Too much silence: {silence_ratio:.2%}"
            
        except Exception as e:
            return False, f"Quality check failed: {e}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def load_and_preprocess(path: str, clip_samples: int, augment: bool = False) -> Optional[np.ndarray]:
    """Enhanced audio loading with validation and optional augmentation."""
    # Validate first
    is_valid, reason = validate_audio_quality(path)
    if not is_valid:
        logger.warning(f"Skipping {path}: {reason}")
        return None
    
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        logger.error(f"librosa error: {e} on {path}")
        return None
    
    # Random crop or pad
    if len(y) >= clip_samples:
        start = np.random.randint(0, len(y) - clip_samples + 1)
        y = y[start : start + clip_samples]
    else:
        y = np.pad(y, (0, clip_samples - len(y)), mode="constant")
    
    # Data augmentation
    if augment:
        y = apply_augmentation(y)
    
    return y.astype(np.float32)

def apply_augmentation(y: np.ndarray) -> np.ndarray:
    """Apply random audio augmentations."""
    augmented = y.copy()
    
    # Random noise (10% chance)
    if np.random.random() < 0.1:
        noise_factor = np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_factor, len(augmented))
        augmented = augmented + noise
    
    # Time shifting (20% chance)
    if np.random.random() < 0.2:
        shift_samples = np.random.randint(-len(y)//10, len(y)//10)
        augmented = np.roll(augmented, shift_samples)
    
    # Volume scaling (30% chance)
    if np.random.random() < 0.3:
        volume_factor = np.random.uniform(0.7, 1.3)
        augmented = augmented * volume_factor
    
    # Ensure we don't clip
    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented

#####################
# Main Extraction   #
#####################
def main():
    """Enhanced main function with argument parsing and comprehensive logging."""
    parser = argparse.ArgumentParser(description="Prepare YAMNet Speech/Non-Speech Dataset")
    parser.add_argument("--speech-samples", type=int, default=800, help="Number of speech samples")
    parser.add_argument("--nonspeech-samples", type=int, default=800, help="Number of non-speech samples")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data")
    parser.add_argument("--output-dir", type=str, default="yamnet_dataset", help="Output directory")
    
    args = parser.parse_args()
    
    # Update global output directory
    global OUT_DIR
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info("YAMNet Speech/Non-Speech Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Speech samples: {args.speech_samples}")
    logger.info(f"Non-speech samples: {args.nonspeech_samples}")
    logger.info(f"Data augmentation: {args.augment}")
    logger.info(f"Output directory: {OUT_DIR}")
    
    start_time = time.time()
    
    # Configurable dataset sizes to prevent overfitting
    SPEECH_SAMPLES = args.speech_samples
    NON_SPEECH_SAMPLES = args.nonspeech_samples
    
    logger.info("Collecting file lists...")
    
    # Speech: LibriSpeech
    speech_files = prepare_librispeech_speech(n_max=SPEECH_SAMPLES)
    
    # Non-speech: ESC-50, UrbanSound8K, MUSAN
    esc50_files = prepare_esc50_non_speech(n_max=min(300, NON_SPEECH_SAMPLES//3))
    urban_files = prepare_urbansound8k_non_speech(n_max=min(400, NON_SPEECH_SAMPLES//2))
    musan_files = prepare_musan_noise(n_max=min(400, NON_SPEECH_SAMPLES//2))
    
    nonspeech_files = esc50_files + urban_files + musan_files
    np.random.shuffle(nonspeech_files)
    nonspeech_files = nonspeech_files[:NON_SPEECH_SAMPLES]
    
    logger.info(f"Collected - Speech: {len(speech_files)}, Non-speech: {len(nonspeech_files)}")
    
    if args.validate_only:
        logger.info("Validation mode - checking audio quality only")
        validate_dataset_quality(speech_files, nonspeech_files)
        return
    
    # Process files with quality validation
    X, y, processing_stats = process_audio_files(speech_files, nonspeech_files, args.augment)
    
    if len(X) == 0:
        logger.error("No valid audio files processed!")
        return
    
    # Convert to numpy arrays
    X = np.stack(X)
    y = np.array(y).reshape(-1, 1)
    
    # Shuffle dataset
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # Save data
    X_path = os.path.join(OUT_DIR, "X.npy")
    y_path = os.path.join(OUT_DIR, "y.npy")
    
    np.save(X_path, X)
    np.save(y_path, y)
    
    # Save metadata
    metadata = {
        "total_samples": len(X),
        "speech_samples": int(np.sum(y)),
        "nonspeech_samples": int(len(y) - np.sum(y)),
        "sample_rate": SAMPLE_RATE,
        "clip_duration": CLIP_DURATION,
        "clip_samples": CLIP_SAMPLES,
        "augmentation_used": args.augment,
        "processing_stats": processing_stats,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": time.time() - start_time
    }
    
    metadata_path = os.path.join(OUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples: {X.shape[0]}")
    logger.info(f"Shape X: {X.shape}, y: {y.shape}")
    logger.info(f"Speech/Non-speech ratio: {np.sum(y)}/{len(y)-np.sum(y)}")
    logger.info(f"Saved to: {X_path}, {y_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Processing stats: {processing_stats}")

def validate_dataset_quality(speech_files: List[str], nonspeech_files: List[str]):
    """Validate dataset quality without processing."""
    logger.info("Validating speech files...")
    speech_valid = 0
    for path in tqdm(speech_files, desc="Speech validation"):
        is_valid, _ = validate_audio_quality(path)
        if is_valid:
            speech_valid += 1
    
    logger.info("Validating non-speech files...")
    nonspeech_valid = 0
    for path in tqdm(nonspeech_files, desc="Non-speech validation"):
        is_valid, _ = validate_audio_quality(path)
        if is_valid:
            nonspeech_valid += 1
    
    logger.info(f"Validation results:")
    logger.info(f"Speech: {speech_valid}/{len(speech_files)} valid ({speech_valid/len(speech_files)*100:.1f}%)")
    logger.info(f"Non-speech: {nonspeech_valid}/{len(nonspeech_files)} valid ({nonspeech_valid/len(nonspeech_files)*100:.1f}%)")

def process_audio_files(speech_files: List[str], nonspeech_files: List[str], augment: bool) -> Tuple[List, List, dict]:
    """Process audio files with comprehensive error tracking."""
    X, y = [], []
    stats = {
        "speech_processed": 0,
        "speech_failed": 0,
        "nonspeech_processed": 0,
        "nonspeech_failed": 0,
        "total_failed": 0
    }
    
    logger.info("Processing speech files...")
    for path in tqdm(speech_files, desc="Speech processing"):
        arr = load_and_preprocess(path, CLIP_SAMPLES, augment)
        if arr is not None:
            X.append(arr)
            y.append(1)
            stats["speech_processed"] += 1
        else:
            stats["speech_failed"] += 1
    
    logger.info("Processing non-speech files...")
    for path in tqdm(nonspeech_files, desc="Non-speech processing"):
        arr = load_and_preprocess(path, CLIP_SAMPLES, augment)
        if arr is not None:
            X.append(arr)
            y.append(0)
            stats["nonspeech_processed"] += 1
        else:
            stats["nonspeech_failed"] += 1
    
    stats["total_failed"] = stats["speech_failed"] + stats["nonspeech_failed"]
    
    logger.info(f"Processing complete: {len(X)} valid samples, {stats['total_failed']} failed")
    return X, y, stats

if __name__ == "__main__":
    main()