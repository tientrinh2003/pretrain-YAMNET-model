import os
import time
import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========= Enhanced Config =========
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YAMNet for Speech/Non-Speech Classification")
    parser.add_argument("--data-dir", type=str, default="yamnet_dataset", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use-scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--model-name", type=str, default="yamnet_finetuned", help="Model name prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

DATA_DIR = args.data_dir
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
EMB_PATH = os.path.join(DATA_DIR, "X_yamnet_emb.npy")

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
DROPOUT = args.dropout
PATIENCE = args.patience
SEED = args.seed

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

logger.info("="*60)
logger.info("YAMNet Fine-tuning Configuration")
logger.info("="*60)
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Learning rate: {LR}")
logger.info(f"Dropout: {DROPOUT}")
logger.info(f"Early stopping patience: {PATIENCE}")
logger.info(f"Use LR scheduler: {args.use_scheduler}")
logger.info(f"Use TensorBoard: {args.use_tensorboard}")
logger.info(f"Random seed: {SEED}")
logger.info("="*60)

# ========= Load data =========
logger.info("Loading waveform data...")

# Check if data exists
if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    logger.error(f"Data files not found! Please run prepare_yamnet_speech_nonspeech.py first")
    logger.error(f"Expected: {X_PATH}, {Y_PATH}")
    exit(1)

X = np.load(X_PATH)  # [N, samples]
y = np.load(Y_PATH)  # [N, 1] hoặc [N,]
if y.ndim == 1:
    y = y.reshape(-1, 1)

logger.info(f"Data loaded: X {X.shape}, y {y.shape}")
logger.info(f"Class distribution: Speech={np.sum(y)}, Non-speech={len(y)-np.sum(y)}")

clip_samples = X.shape[1]

# Load metadata if available
metadata_path = os.path.join(DATA_DIR, "metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Dataset metadata: {metadata.get('created_at', 'Unknown')}")
    logger.info(f"Augmentation used: {metadata.get('augmentation_used', 'Unknown')}")
else:
    logger.warning("No metadata file found")

# ========= Extract embeddings (cache) =========
def extract_embeddings_if_needed(X: np.ndarray, emb_path: str) -> np.ndarray:
    """Extract YAMNet embeddings with progress tracking."""
    if os.path.exists(emb_path):
        logger.info(f"[✓] Found cached embeddings: {emb_path}")
        return np.load(emb_path)

    logger.info("Loading YAMNet from TFHub...")
    try:
        yamnet = hub.load(YAMNET_HUB_URL)
    except Exception as e:
        logger.error(f"Failed to load YAMNet: {e}")
        raise

    logger.info("Extracting YAMNet embeddings (mean-pooled)...")
    embs = []
    failed_count = 0
    
    for i in tqdm(range(len(X)), desc="Extracting embeddings"):
        try:
            wf = tf.convert_to_tensor(X[i], dtype=tf.float32)  # [samples]
            scores, embeddings, spec = yamnet(wf)              # embeddings: [n_patches, 1024]
            emb = tf.reduce_mean(embeddings, axis=0)           # [1024]
            embs.append(emb.numpy())
        except Exception as e:
            logger.warning(f"Failed to extract embedding for sample {i}: {e}")
            # Use zero embedding as fallback
            embs.append(np.zeros(1024, dtype=np.float32))
            failed_count += 1
    
    embs = np.stack(embs)  # [N, 1024]
    np.save(emb_path, embs)
    
    logger.info(f"[✓] Saved embeddings to {emb_path}")
    if failed_count > 0:
        logger.warning(f"Failed to extract {failed_count}/{len(X)} embeddings")
    
    return embs

X_emb = extract_embeddings_if_needed(X, EMB_PATH)
logger.info(f"Embeddings shape: {X_emb.shape}")

# Basic embedding statistics
logger.info(f"Embedding stats - Mean: {X_emb.mean():.4f}, Std: {X_emb.std():.4f}")
logger.info(f"Embedding range - Min: {X_emb.min():.4f}, Max: {X_emb.max():.4f}")

# ========= Enhanced Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y, test_size=0.12, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.12, random_state=SEED, stratify=y_train
)

logger.info(f"Dataset splits:")
logger.info(f"  Train: {X_train.shape[0]} samples")
logger.info(f"  Validation: {X_val.shape[0]} samples") 
logger.info(f"  Test: {X_test.shape[0]} samples")

# Log class distributions
for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    speech_count = np.sum(split_y)
    total_count = len(split_y)
    logger.info(f"  {split_name} distribution: {speech_count} speech, {total_count-speech_count} non-speech")

# ========= Enhanced Model Architecture =========
def build_enhanced_head_model(input_dim: int = 1024, hidden_dim: int = 512, 
                             dropout_rate: float = 0.3, use_batch_norm: bool = True):
    """Build enhanced head classifier with batch normalization and better architecture."""
    inp = tf.keras.Input(shape=(input_dim,), name="emb_input")
    
    # First hidden layer
    x = tf.keras.layers.Dense(hidden_dim, name="head_dense1")(inp)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name="batch_norm1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout1")(x)
    
    # Second hidden layer (smaller)
    x = tf.keras.layers.Dense(hidden_dim // 2, name="head_dense2")(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name="batch_norm2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name="dropout2")(x)
    
    # Output layer
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="head_out")(x)
    
    model = tf.keras.Model(inputs=inp, outputs=out, name="yamnet_head_classifier")
    return model

# Build model
head_model = build_enhanced_head_model(
    hidden_dim=512, 
    dropout_rate=DROPOUT, 
    use_batch_norm=True
)

# Enhanced optimizer with optional learning rate scheduling
if args.use_scheduler:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR,
        first_decay_steps=100,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    logger.info("Using cosine decay learning rate scheduler")
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    logger.info(f"Using fixed learning rate: {LR}")

head_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"]
)

logger.info("Model architecture:")
head_model.summary(print_fn=logger.info)

# ========= Enhanced Training Setup =========
# Prepare callbacks
callbacks = []

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=PATIENCE, 
    restore_best_weights=True,
    verbose=1
)
callbacks.append(early_stop)

# Model checkpoint
checkpoint_path = f"models/best_{args.model_name}.keras"
os.makedirs("models", exist_ok=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
callbacks.append(checkpoint)

# Reduce learning rate on plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)
callbacks.append(reduce_lr)

# TensorBoard logging
if args.use_tensorboard:
    log_dir = f"logs/{args.model_name}_{int(time.time())}"
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

logger.info("Starting training...")
start_time = time.time()

# Train model
history = head_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
logger.info(f"Training completed in {training_time:.2f} seconds")

# ========= Comprehensive Evaluation =========
def plot_training_history(history, save_path: str = "training_plots.png"):
    """Plot training history with multiple metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training plots saved to: {save_path}")
    plt.close()

def evaluate_model_comprehensive(model, X_test, y_test, save_plots: bool = True):
    """Comprehensive model evaluation with multiple metrics and plots."""
    logger.info("="*50)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*50)
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Basic metrics
    logger.info("== Test Metrics ==")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    logger.info(classification_report(y_test, y_pred, digits=4))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Average Precision
    avg_precision = average_precision_score(y_test, y_pred_prob)
    logger.info(f"Average Precision Score: {avg_precision:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Negatives: {cm[0,0]}")
    logger.info(f"  False Positives: {cm[0,1]}") 
    logger.info(f"  False Negatives: {cm[1,0]}")
    logger.info(f"  True Positives: {cm[1,1]}")
    
    if save_plots:
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # Plot Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # Plot Confusion Matrix
        plt.subplot(1, 3, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
        logger.info("Evaluation plots saved to: evaluation_plots.png")
        plt.close()
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm.tolist()
    }

# Plot training history
plot_training_history(history)

# Validation evaluation
logger.info("== Validation Metrics ==")
y_val_predprob = head_model.predict(X_val, verbose=0)
y_val_pred = (y_val_predprob >= 0.5).astype(int)
logger.info(classification_report(y_val, y_val_pred, digits=4))

# Comprehensive test evaluation
test_metrics = evaluate_model_comprehensive(head_model, X_test, y_test)

# ========= Enhanced Model Saving =========
# Save head classifier
HEAD_PATH = f"models/{args.model_name}_head.keras"
head_model.save(HEAD_PATH)
logger.info(f"[✓] Saved head classifier to {HEAD_PATH}")

# Save training history
history_path = f"models/{args.model_name}_history.json"
history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
logger.info(f"[✓] Saved training history to {history_path}")

# ========= Build Enhanced Wrapper Model =========
class YamnetEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, yamnet_model, name="yamnet_embedding"):
        super().__init__(trainable=False, name=name)
        self.yamnet = yamnet_model

    def call(self, waveforms: tf.Tensor):
        # waveforms: [batch, samples]
        # Apply per-example to respect YAMNet signature ([samples],)
        def per_example(wf):
            _, embeddings, _ = self.yamnet(wf)      # [n_patches, 1024]
            return tf.reduce_mean(embeddings, axis=0)  # [1024]
        return tf.map_fn(per_example, waveforms, fn_output_signature=tf.float32)

logger.info("Building enhanced wrapper model (YAMNet frozen + head)...")
yamnet_loaded = hub.load(YAMNET_HUB_URL)
inp_wav = tf.keras.Input(shape=(clip_samples,), dtype=tf.float32, name="waveform")
embs = YamnetEmbeddingLayer(yamnet_loaded, name="yamnet_embedding")(inp_wav)

# Rebuild the head architecture to match trained model
x = tf.keras.layers.Dense(512, name="head_dense1")(embs)
x = tf.keras.layers.BatchNormalization(name="batch_norm1")(x)
x = tf.keras.layers.Activation("relu", name="relu1")(x)
x = tf.keras.layers.Dropout(DROPOUT, name="dropout1")(x)

x = tf.keras.layers.Dense(256, name="head_dense2")(x)
x = tf.keras.layers.BatchNormalization(name="batch_norm2")(x)
x = tf.keras.layers.Activation("relu", name="relu2")(x)
x = tf.keras.layers.Dropout(DROPOUT * 0.5, name="dropout2")(x)

out = tf.keras.layers.Dense(1, activation="sigmoid", name="head_out")(x)

wrapper_model = tf.keras.Model(inputs=inp_wav, outputs=out, name="yamnet_sns_wrapper")

# Initialize model with dummy input
_ = wrapper_model(tf.zeros((1, clip_samples), dtype=tf.float32))

# Copy trained head weights into wrapper
layer_mapping = {
    "head_dense1": "head_dense1",
    "batch_norm1": "batch_norm1", 
    "head_dense2": "head_dense2",
    "batch_norm2": "batch_norm2",
    "head_out": "head_out"
}

for head_layer_name, wrapper_layer_name in layer_mapping.items():
    try:
        head_layer = head_model.get_layer(head_layer_name)
        wrapper_layer = wrapper_model.get_layer(wrapper_layer_name)
        wrapper_layer.set_weights(head_layer.get_weights())
        logger.info(f"Copied weights for layer: {head_layer_name}")
    except Exception as e:
        logger.warning(f"Could not copy weights for layer {head_layer_name}: {e}")

# Compile wrapper model
wrapper_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Quick sanity check
logger.info("Performing sanity check on wrapper model...")
pred_check = wrapper_model.predict(X[:4].astype(np.float32), verbose=0)
logger.info(f"Wrapper sample predictions: {pred_check.ravel()}")

# Save SavedModel for deployment
WRAPPER_DIR = f"models/{args.model_name}_savedmodel"
wrapper_model.export(WRAPPER_DIR)
logger.info(f"[✓] Saved end-to-end model to {WRAPPER_DIR}")

# ========= Create Comprehensive ZIP Package =========
import zipfile
import shutil

ZIP_PATH = f"{args.model_name}_complete.zip"
logger.info(f"Creating comprehensive ZIP package: {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add SavedModel directory
    for root, dirs, files in os.walk(WRAPPER_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arc_name = os.path.relpath(file_path, os.path.dirname(WRAPPER_DIR))
            zipf.write(file_path, arc_name)
    
    # Add head classifier
    zipf.write(HEAD_PATH, f"{args.model_name}_head.keras")
    
    # Add training history
    zipf.write(history_path, f"{args.model_name}_history.json")
    
    # Add plots if they exist
    for plot_file in ["training_plots.png", "evaluation_plots.png"]:
        if os.path.exists(plot_file):
            zipf.write(plot_file, plot_file)
    
    # Add comprehensive metadata
    final_metadata = {
        "model_name": args.model_name,
        "model_type": "YAMNet Speech/Non-Speech Classifier",
        "framework": "TensorFlow",
        "yamnet_hub_url": YAMNET_HUB_URL,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "use_scheduler": args.use_scheduler,
            "use_tensorboard": args.use_tensorboard
        },
        "model_architecture": {
            "input_shape": f"[batch_size, {clip_samples}]",
            "output_shape": "[batch_size, 1]",
            "embedding_dim": 1024,
            "hidden_layers": [512, 256],
            "activation": "relu",
            "output_activation": "sigmoid"
        },
        "dataset_info": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val), 
            "test_samples": len(X_test),
            "sample_rate": 16000,
            "clip_duration": 2.0
        },
        "performance_metrics": test_metrics,
        "training_time_seconds": training_time,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": SEED
    }
    
    zipf.writestr("model_info.json", json.dumps(final_metadata, indent=2))

logger.info(f"[✓] Created comprehensive ZIP package: {ZIP_PATH}")

# ========= Final Summary =========
logger.info("="*60)
logger.info("TRAINING COMPLETE - FINAL SUMMARY")
logger.info("="*60)
logger.info(f"Model name: {args.model_name}")
logger.info(f"Training time: {training_time:.2f} seconds")
logger.info(f"Best validation loss: {min(history.history['val_loss']):.6f}")
logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
logger.info(f"Final test ROC-AUC: {test_metrics['roc_auc']:.4f}")
logger.info(f"Model files created:")
logger.info(f"  - Head classifier: {HEAD_PATH}")
logger.info(f"  - SavedModel: {WRAPPER_DIR}")
logger.info(f"  - Complete package: {ZIP_PATH}")
logger.info(f"  - Training history: {history_path}")
if args.use_tensorboard:
    logger.info(f"  - TensorBoard logs: {log_dir}")
logger.info("="*60)

logger.info("All done! 🎉")