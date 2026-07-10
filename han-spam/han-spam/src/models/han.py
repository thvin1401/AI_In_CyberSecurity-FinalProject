from __future__ import annotations

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import initializers, layers, models
import tensorflow.keras.backend as K

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger("han")


# ──────────────────────────────────────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W = self.add_weight(name="att_W",
            shape=(hidden_dim, self.attention_dim),
            initializer=initializers.GlorotUniform(), trainable=True)
        self.b = self.add_weight(name="att_b",
            shape=(self.attention_dim,),
            initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u",
            shape=(self.attention_dim, 1),
            initializer=initializers.GlorotUniform(), trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.squeeze(K.dot(uit, self.u), axis=-1)
        ait = K.exp(ait)
        if mask is not None:
            ait = ait * K.cast(mask, K.floatx())
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        return K.sum(x * K.expand_dims(ait, axis=-1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attention_dim": self.attention_dim})
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Word Encoder — subclassed for Keras 3 TimeDistributed compatibility
# ──────────────────────────────────────────────────────────────────────────────
class WordEncoder(models.Model):
    def __init__(self, vocab_size, embed_dim, embedding_matrix,
                 cnn_filters, cnn_kernels, gru_units, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

        self.embedding = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim,
            weights=[embedding_matrix],
            mask_zero=True, trainable=False, name="embedding")

        self.conv1d = layers.Conv1D(
            cnn_filters[0], cnn_kernels[0],
            padding="same", activation="relu", name="conv1d")
        self.conv1d_1 = layers.Conv1D(
            cnn_filters[1], cnn_kernels[1],
            padding="same", activation="relu", name="conv1d_1")
        self.concatenate = layers.Concatenate(name="concatenate")
        self.lambda_layer = layers.Lambda(lambda t: t, name="lambda")

        self.bidirectional = layers.Bidirectional(
            layers.GRU(gru_units, return_sequences=True),
            name="bidirectional")

        self.attention_layer = AttentionLayer(attention_dim, name="attention_layer")

    def call(self, x, training=False):
        emb = self.embedding(x)
        c0  = self.conv1d(emb)
        c1  = self.conv1d_1(emb)
        cat = self.concatenate([c0, c1])
        cat = self.lambda_layer(cat)
        gru = self.bidirectional(cat, training=training)
        return self.attention_layer(gru)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.attention_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Build model
# ──────────────────────────────────────────────────────────────────────────────
def build_han_model(max_sentences, max_words, vocab_size, embed_dim,
                    embedding_matrix, cfg=None):
    if cfg is None:
        cfg = CFG

    word_encoder = WordEncoder(
        vocab_size=vocab_size, embed_dim=embed_dim,
        embedding_matrix=embedding_matrix,
        cnn_filters=cfg.model.cnn_filters,
        cnn_kernels=cfg.model.cnn_kernels,
        gru_units=cfg.model.word_gru_units,
        attention_dim=cfg.model.word_attention_dim,
        name="word_encoder",
    )

    doc_input = layers.Input(
        shape=(max_sentences, max_words), dtype="int32", name="input_layer")

    sent_vectors = layers.TimeDistributed(
        word_encoder, name="time_distributed")(doc_input)

    sent_gru = layers.Bidirectional(
        layers.GRU(cfg.model.sent_gru_units, return_sequences=True),
        name="bidirectional")(sent_vectors)

    doc_vec = AttentionLayer(
        cfg.model.sent_attention_dim, name="attention_layer")(sent_gru)

    x      = layers.Dropout(cfg.model.dropout_rate, name="dropout")(doc_vec)
    output = layers.Dense(1, activation=cfg.model.output_activation, name="dense")(x)

    model = models.Model(inputs=doc_input, outputs=output, name="HAN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    log.info(f"Built HAN | params={model.count_params():,}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Load weights manually from h5 paths (avoids layer ordering mismatch)
# ──────────────────────────────────────────────────────────────────────────────
def _set_weights_from_h5(model, h5_path: str):
    """
    Load weights by explicit h5 dataset path instead of relying on
    Keras layer ordering — prevents mismatch when subclassed models
    are wrapped in TimeDistributed.
    """
    with h5py.File(h5_path, "r") as f:
        td = f["layers/time_distributed/layer/layers"]
        sent = f["layers"]

        # ── Word encoder weights ─────────────────────────────────────────────
        we = model.get_layer("time_distributed").layer

        # embedding
        we.embedding.set_weights([
            td["embedding/vars/0"][:]
        ])
        # conv1d
        we.conv1d.set_weights([
            td["conv1d/vars/0"][:],
            td["conv1d/vars/1"][:]
        ])
        # conv1d_1
        we.conv1d_1.set_weights([
            td["conv1d_1/vars/0"][:],
            td["conv1d_1/vars/1"][:]
        ])
        # word BiGRU (bidirectional)
        we.bidirectional.set_weights([
            td["bidirectional/forward_layer/cell/vars/0"][:],
            td["bidirectional/forward_layer/cell/vars/1"][:],
            td["bidirectional/forward_layer/cell/vars/2"][:],
            td["bidirectional/backward_layer/cell/vars/0"][:],
            td["bidirectional/backward_layer/cell/vars/1"][:],
            td["bidirectional/backward_layer/cell/vars/2"][:],
        ])
        # word attention
        we.attention_layer.set_weights([
            td["attention_layer/vars/0"][:],
            td["attention_layer/vars/1"][:],
            td["attention_layer/vars/2"][:]
        ])

        # ── Sentence encoder weights ─────────────────────────────────────────
        # sentence BiGRU
        model.get_layer("bidirectional").set_weights([
            sent["bidirectional/forward_layer/cell/vars/0"][:],
            sent["bidirectional/forward_layer/cell/vars/1"][:],
            sent["bidirectional/forward_layer/cell/vars/2"][:],
            sent["bidirectional/backward_layer/cell/vars/0"][:],
            sent["bidirectional/backward_layer/cell/vars/1"][:],
            sent["bidirectional/backward_layer/cell/vars/2"][:],
        ])
        # sentence attention
        model.get_layer("attention_layer").set_weights([
            sent["attention_layer/vars/0"][:],
            sent["attention_layer/vars/1"][:],
            sent["attention_layer/vars/2"][:]
        ])
        # dense
        model.get_layer("dense").set_weights([
            sent["dense/vars/0"][:],
            sent["dense/vars/1"][:]
        ])

    log.info(f"Manually loaded weights from: {h5_path}")


def build_word_encoder(max_words, vocab_size, embed_dim, embedding_matrix, cfg=None):
    """Kept for API compatibility."""
    if cfg is None:
        cfg = CFG
    return WordEncoder(
        vocab_size=vocab_size, embed_dim=embed_dim,
        embedding_matrix=embedding_matrix,
        cnn_filters=cfg.model.cnn_filters,
        cnn_kernels=cfg.model.cnn_kernels,
        gru_units=cfg.model.word_gru_units,
        attention_dim=cfg.model.word_attention_dim,
    )


def load_han_model(weights_path, max_sentences, max_words, vocab_size,
                   embed_dim, embedding_matrix, cfg=None):
    model = build_han_model(
        max_sentences=max_sentences, max_words=max_words,
        vocab_size=vocab_size, embed_dim=embed_dim,
        embedding_matrix=embedding_matrix, cfg=cfg,
    )
    # Build model by running a dummy forward pass
    dummy = np.zeros((1, max_sentences, max_words), dtype="int32")
    model(dummy, training=False)

    # Load weights manually by h5 path
    _set_weights_from_h5(model, weights_path)
    return model
