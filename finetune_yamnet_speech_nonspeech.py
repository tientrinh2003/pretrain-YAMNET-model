import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ========= Config =========
DATA_DIR = "yamnet_dataset"
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
EMB_PATH = os.path.join(DATA_DIR, "X_yamnet_emb.npy")

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3  # LR cho head (huấn luyện trên embedding)
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ========= Load data =========
print("Loading waveform data...")
X = np.load(X_PATH)  # [N, samples]
y = np.load(Y_PATH)  # [N, 1] hoặc [N,]
if y.ndim == 1:
    y = y.reshape(-1, 1)
print("Data loaded:", X.shape, y.shape)
clip_samples = X.shape[1]

# ========= Extract embeddings (cache) =========
def extract_embeddings_if_needed(X: np.ndarray, emb_path: str) -> np.ndarray:
    if os.path.exists(emb_path):
        print(f"[✓] Found cached embeddings: {emb_path}")
        return np.load(emb_path)

    print("Loading YAMNet from TFHub...")
    yamnet = hub.load(YAMNET_HUB_URL)

    print("Extracting YAMNet embeddings (mean-pooled)...")
    embs = []
    for i in tqdm(range(len(X))):
        wf = tf.convert_to_tensor(X[i], dtype=tf.float32)  # [samples]
        scores, embeddings, spec = yamnet(wf)              # embeddings: [n_patches, 1024]
        emb = tf.reduce_mean(embeddings, axis=0)           # [1024]
        embs.append(emb.numpy())
    embs = np.stack(embs)  # [N, 1024]
    np.save(emb_path, embs)
    print(f"[✓] Saved embeddings to {emb_path}")
    return embs

X_emb = extract_embeddings_if_needed(X, EMB_PATH)
print("Embeddings shape:", X_emb.shape)

# ========= Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y, test_size=0.12, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.12, random_state=SEED, stratify=y_train
)
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ========= Build head classifier on embeddings =========
def build_head_model():
    inp = tf.keras.Input(shape=(1024,), name="emb_input")
    x = tf.keras.layers.Dense(512, activation="relu", name="head_dense")(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="head_out")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="yamnet_head_classifier")
    return model

head_model = build_head_model()
head_model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
head_model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
print("Training head on embeddings...")
history = head_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# ========= Evaluate =========
print("== Validation metrics ==")
y_val_predprob = head_model.predict(X_val, verbose=0)
y_val_pred = (y_val_predprob >= 0.5).astype(int)
print(classification_report(y_val, y_val_pred, digits=4))

print("== Test metrics ==")
y_test_predprob = head_model.predict(X_test, verbose=0)
y_test_pred = (y_test_predprob >= 0.5).astype(int)
print(classification_report(y_test, y_test_pred, digits=4))

# ========= Save head =========
HEAD_PATH = "classifier_on_emb.h5"
head_model.save(HEAD_PATH)
print(f"[✓] Saved head classifier to {HEAD_PATH}")

# ========= Build wrapper (YAMNet frozen + head) for end-to-end inference =========
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

print("Building wrapper model (YAMNet frozen + head)...")
yamnet_loaded = hub.load(YAMNET_HUB_URL)
inp_wav = tf.keras.Input(shape=(clip_samples,), dtype=tf.float32, name="waveform")
embs = YamnetEmbeddingLayer(yamnet_loaded, name="yamnet_embedding")(inp_wav)
x = tf.keras.layers.Dense(512, activation="relu", name="head_dense")(embs)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid", name="head_out")(x)
wrapper_model = tf.keras.Model(inputs=inp_wav, outputs=out, name="yamnet_sns_wrapper")

# Build models (run a dummy pass) to init weights before setting
_ = wrapper_model(tf.zeros((1, clip_samples), dtype=tf.float32))

# Copy trained head weights into wrapper
wrapper_model.get_layer("head_dense").set_weights(
    head_model.get_layer("head_dense").get_weights()
)
wrapper_model.get_layer("head_out").set_weights(
    head_model.get_layer("head_out").get_weights()
)

# Optional compile for evaluation
wrapper_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Quick sanity check on a small batch
print("Quick sanity check on wrapper...")
pred_check = wrapper_model.predict(X[:4].astype(np.float32), verbose=0)
print("Wrapper sample preds:", pred_check.ravel())

# Save SavedModel for deployment/inference
WRAPPER_DIR = "yamnet_sns_savedmodel"
wrapper_model.save(WRAPPER_DIR)
print(f"[✓] Saved end-to-end model to {WRAPPER_DIR}")

print("All done.")