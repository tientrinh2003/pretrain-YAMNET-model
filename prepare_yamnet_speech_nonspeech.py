import os
import zipfile
import tarfile
import numpy as np
import librosa
from tqdm import tqdm

SAMPLE_RATE = 16000
CLIP_DURATION = 2.0  # seconds
CLIP_SAMPLES = int(CLIP_DURATION * SAMPLE_RATE)
OUT_DIR = "yamnet_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def download(url, out_path):
    if os.path.exists(out_path):
        print(f"[✓] Already exists: {out_path}")
        return
    print(f"Downloading {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, out_path)
    print("Done.")

def extract_zip(zip_path, extract_to, check_folder=None):
    if check_folder and os.path.exists(check_folder):
        print(f"[✓] Already extracted: {check_folder}")
        return
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Done.")

def extract_tar(tar_path, extract_to, check_folder=None):
    if check_folder and os.path.exists(check_folder):
        print(f"[✓] Already extracted: {check_folder}")
        return
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(extract_to)
    print("Done.")

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
def load_and_preprocess(path, clip_samples, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"librosa error: {e} on {path}")
        return None
    if len(y) >= clip_samples:
        start = np.random.randint(0, len(y) - clip_samples + 1)
        y = y[start : start + clip_samples]
    else:
        y = np.pad(y, (0, clip_samples - len(y)), mode="constant")
    return y.astype(np.float32)

#####################
# Main Extraction   #
#####################
def main():
    print("Collecting file lists...")
    # Speech: LibriSpeech
    speech_files = prepare_librispeech_speech(n_max=1200)
    # Non-speech: ESC-50, UrbanSound8K, MUSAN
    nonspeech_files = (
        prepare_esc50_non_speech(n_max=600)
        + prepare_urbansound8k_non_speech(n_max=600)
        + prepare_musan_noise(n_max=600)
    )
    np.random.shuffle(nonspeech_files)
    nonspeech_files = nonspeech_files[:1200]
    print(f"Speech: {len(speech_files)}, Non-speech: {len(nonspeech_files)}")

    X, y = [], []

    print("Processing speech files...")
    for path in tqdm(speech_files):
        arr = load_and_preprocess(path, CLIP_SAMPLES)
        if arr is not None:
            X.append(arr)
            y.append(1)
    print("Processing non-speech files...")
    for path in tqdm(nonspeech_files):
        arr = load_and_preprocess(path, CLIP_SAMPLES)
        if arr is not None:
            X.append(arr)
            y.append(0)
    X = np.stack(X)
    y = np.array(y).reshape(-1, 1)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    print(f"Saved {X.shape[0]} samples to {OUT_DIR}/X.npy and y.npy")
    print(f"Shape X: {X.shape}, y: {y.shape}")

if __name__ == "__main__":
    main()