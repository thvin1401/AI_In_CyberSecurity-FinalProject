# %% [markdown]
# # HAN Email Spam Detection — Colab Training
#
# Trains the Hierarchical Attention Network on preprocessed data synced from
# Google Drive (produced locally by `src/data/preprocess.py` in Phase 2).
#
# **Crash-resilient**: each of the 10 CV folds saves its checkpoint to Drive
# immediately after training. If this session disconnects, re-run all cells —
# folds that already have a saved checkpoint are skipped automatically.
#
# **Setup required once**: your preprocessed data must already be in
# `My Drive/han-spam/data/processed/` (pushed via `drive_sync.py push-data`
# from your local Docker container).

# %% [markdown]
# ## 1. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/han-spam'
DATA_DIR = f'{DRIVE_ROOT}/data/processed'
CKPT_DIR = f'{DRIVE_ROOT}/checkpoints'
RESULTS_DIR = f'{DRIVE_ROOT}/results'

import os
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print('Drive mounted. Expected data at:', DATA_DIR)
assert os.path.exists(DATA_DIR), (
    f"Data not found at {DATA_DIR}. "
    "Run 'python scripts/drive_sync.py push-data' locally first."
)
print('Found:', os.listdir(DATA_DIR))

# %% [markdown]
# ## 2. Check GPU

# %%
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for g in gpus:
    print(' ', g)
if not gpus:
    print('WARNING: No GPU detected. Runtime → Change runtime type → T4 GPU')

# %% [markdown]
# ## 3. Install extra deps (everything else is preinstalled on Colab)

# %%
# !pip install -q pyyaml

# %% [markdown]
# ## 4. Load config + preprocessed data

# %%
import json
import numpy as np
import yaml
from types import SimpleNamespace

def to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, to_ns(v) if isinstance(v, dict) else v)
    return ns

# Config is small — embed inline so the notebook is self-contained
# (mirrors configs/config.yaml; keep these two files in sync manually,
#  or copy config.yaml into DATA_DIR if you prefer single-source-of-truth)
CONFIG_YAML = """
model:
  cnn_filters: [64, 128]
  cnn_kernels: [3, 5]
  word_gru_units: 50
  word_attention_dim: 100
  sent_gru_units: 50
  sent_attention_dim: 100
  dropout_rate: 0.5
  output_activation: sigmoid
  use_tcn: false
  tcn_filters: 64
  tcn_kernel_size: 3
  tcn_dilations: [1, 2, 4, 8]
  tcn_dropout: 0.2
training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.001
  cv_folds: 10
  early_stopping_patience: 3
project:
  seed: 42
"""
CFG = to_ns(yaml.safe_load(CONFIG_YAML))

X_train = np.load(f'{DATA_DIR}/X_train.npy')
y_train = np.load(f'{DATA_DIR}/y_train.npy')
X_test = np.load(f'{DATA_DIR}/X_test.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')
embedding_matrix = np.load(f'{DATA_DIR}/embedding_matrix.npy')

with open(f'{DATA_DIR}/manifest.json') as f:
    manifest = json.load(f)

print('X_train:', X_train.shape)
print('X_test :', X_test.shape)
print('embedding_matrix:', embedding_matrix.shape)
print('manifest:', manifest)

# %% [markdown]
# ## 5. Build the HAN model
#
# (This mirrors `src/models/han.py` from the local repo — Phase 3. If you've
# built that file already, you can instead `!cp` it from a GitHub repo or
# paste its contents here. For now this is a self-contained definition.)

# %%
from tensorflow.keras import layers, models, initializers
import tensorflow.keras.backend as K


class AttentionLayer(layers.Layer):
    """Word/sentence-level additive attention (Yang et al. 2016 HAN attention)."""

    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_W', shape=(input_shape[-1], self.attention_dim),
            initializer=initializers.GlorotUniform(), trainable=True)
        self.b = self.add_weight(
            name='att_b', shape=(self.attention_dim,),
            initializer='zeros', trainable=True)
        self.u = self.add_weight(
            name='att_u', shape=(self.attention_dim, 1),
            initializer=initializers.GlorotUniform(), trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            ait = ait * K.cast(mask, K.floatx())
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        ait = K.expand_dims(ait)
        weighted = x * ait
        return K.sum(weighted, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_word_encoder(max_words, vocab_size, embed_dim, embedding_matrix, cfg):
    word_input = layers.Input(shape=(max_words,), dtype='int32')
    embed = layers.Embedding(
        vocab_size, embed_dim, weights=[embedding_matrix],
        input_length=max_words, mask_zero=True, trainable=False,
    )(word_input)

    # Multi-kernel CNN bank (concatenated)
    conv_outputs = []
    for filters, kernel in zip(cfg.model.cnn_filters, cfg.model.cnn_kernels):
        c = layers.Conv1D(filters, kernel, padding='same', activation='relu')(embed)
        conv_outputs.append(c)
    cnn_out = layers.Concatenate()(conv_outputs) if len(conv_outputs) > 1 else conv_outputs[0]

    gru_out = layers.Bidirectional(
        layers.GRU(cfg.model.word_gru_units, return_sequences=True)
    )(cnn_out)
    sent_vec = AttentionLayer(cfg.model.word_attention_dim)(gru_out)
    return models.Model(word_input, sent_vec, name='word_encoder')


def build_han_model(max_sentences, max_words, vocab_size, embed_dim, embedding_matrix, cfg):
    word_encoder = build_word_encoder(max_words, vocab_size, embed_dim, embedding_matrix, cfg)

    doc_input = layers.Input(shape=(max_sentences, max_words), dtype='int32')
    sent_encoded = layers.TimeDistributed(word_encoder)(doc_input)

    sent_gru_out = layers.Bidirectional(
        layers.GRU(cfg.model.sent_gru_units, return_sequences=True)
    )(sent_encoded)
    doc_vec = AttentionLayer(cfg.model.sent_attention_dim)(sent_gru_out)

    doc_vec = layers.Dropout(cfg.model.dropout_rate)(doc_vec)
    output = layers.Dense(1, activation=cfg.model.output_activation)(doc_vec)

    model = models.Model(doc_input, output, name='HAN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model


print('Model builder ready.')

# %% [markdown]
# ## 6. 10-fold CV training — crash-resilient
#
# Each fold:
# 1. Checks if `{CKPT_DIR}/fold_{i}.weights.h5` already exists → skip if so
# 2. Trains with early stopping
# 3. Saves weights + fold metrics to Drive immediately
#
# Re-running this cell after a disconnect picks up where it left off.

# %%
import time
from sklearn.model_selection import StratifiedKFold

vocab_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]
max_sentences = manifest['max_sentences']
max_words = manifest['max_words']

skf = StratifiedKFold(
    n_splits=CFG.training.cv_folds, shuffle=True, random_state=CFG.project.seed
)

fold_results = []
metrics_path = f'{RESULTS_DIR}/cv_metrics.json'

# Resume: load any previously saved fold results
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        fold_results = json.load(f)
    print(f'Resuming — {len(fold_results)} folds already completed.')

completed_folds = {r['fold'] for r in fold_results}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_num = fold_idx + 1
    ckpt_path = f'{CKPT_DIR}/fold_{fold_num:02d}.weights.h5'

    if fold_num in completed_folds and os.path.exists(ckpt_path):
        print(f'[fold {fold_num}/10] already done — skipping')
        continue

    print(f'\n{"="*60}')
    print(f'[fold {fold_num}/10] training...')
    print(f'{"="*60}')

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    tf.keras.backend.clear_session()
    model = build_han_model(max_sentences, max_words, vocab_size, embed_dim, embedding_matrix, CFG)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max',
            patience=CFG.training.early_stopping_patience,
            restore_best_weights=True,
        ),
        # Belt-and-braces: also checkpoint mid-training in case of crash
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_auc', mode='max',
            save_best_only=True, save_weights_only=True,
        ),
    ]

    start = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CFG.training.epochs,
        batch_size=CFG.training.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    # Final save (in case ModelCheckpoint didn't fire, e.g. only 1 epoch ran)
    model.save_weights(ckpt_path)

    val_metrics = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    fold_result = {
        'fold': fold_num,
        'elapsed_sec': elapsed,
        'epochs_trained': len(history.history['loss']),
        **{k: float(v) for k, v in val_metrics.items()},
    }
    fold_results.append(fold_result)

    # Save metrics to Drive immediately — this is what makes it crash-resilient
    with open(metrics_path, 'w') as f:
        json.dump(fold_results, f, indent=2)

    print(f'[fold {fold_num}/10] done in {elapsed:.0f}s — val_auc={fold_result.get("auc", 0):.4f}')

print('\nAll folds complete.')

# %% [markdown]
# ## 7. Cross-validation summary

# %%
import pandas as pd

results_df = pd.DataFrame(fold_results)
print(results_df[['fold', 'auc', 'accuracy', 'precision', 'recall', 'epochs_trained']])
print('\nMean ± std across folds:')
for metric in ['auc', 'accuracy', 'precision', 'recall']:
    if metric in results_df.columns:
        print(f'  {metric}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}')

# %% [markdown]
# ## 8. Final evaluation on held-out test set
#
# Uses the best-performing fold's weights (highest val_auc).

# %%
best_fold = results_df.loc[results_df['auc'].idxmax(), 'fold']
best_ckpt = f'{CKPT_DIR}/fold_{int(best_fold):02d}.weights.h5'
print(f'Loading best fold: {int(best_fold)} ({best_ckpt})')

tf.keras.backend.clear_session()
final_model = build_han_model(max_sentences, max_words, vocab_size, embed_dim, embedding_matrix, CFG)
final_model.load_weights(best_ckpt)

test_metrics = final_model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print('\nTest set performance:')
for k, v in test_metrics.items():
    print(f'  {k}: {v:.4f}')

with open(f'{RESULTS_DIR}/test_metrics.json', 'w') as f:
    json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

print(f'\nAll checkpoints and results saved to: {CKPT_DIR} and {RESULTS_DIR}')
print('Pull them locally with: python scripts/drive_sync.py pull-checkpoints')
print('                        python scripts/drive_sync.py pull-results')
