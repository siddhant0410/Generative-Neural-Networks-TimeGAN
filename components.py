# components.py

import tensorflow as tf
from utils import rnn_cell

def build_embedder(X, T, module, hidden_dim, num_layers):
    with tf.compat.v1.variable_scope('embedder', reuse=tf.compat.v1.AUTO_REUSE):
        cells = [rnn_cell(module, hidden_dim) for _ in range(num_layers)]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, X, sequence_length=T, dtype=tf.float32)
        H = tf.compat.v1.layers.dense(outputs, hidden_dim, activation=tf.nn.sigmoid)
    return H

def build_recovery(H, T, module, hidden_dim, num_layers, output_dim):
    with tf.compat.v1.variable_scope('recovery', reuse=tf.compat.v1.AUTO_REUSE):
        cells = [rnn_cell(module, hidden_dim) for _ in range(num_layers)]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, H, sequence_length=T, dtype=tf.float32)
        X_tilde = tf.compat.v1.layers.dense(outputs, output_dim, activation=tf.nn.sigmoid)
    return X_tilde

def build_generator(Z, T, module, hidden_dim, num_layers):
    with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
        cells = [rnn_cell(module, hidden_dim) for _ in range(num_layers)]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, Z, sequence_length=T, dtype=tf.float32)
        E = tf.compat.v1.layers.dense(outputs, hidden_dim, activation=tf.nn.sigmoid)
    return E

def build_supervisor(H, T, module, hidden_dim, num_layers):
    with tf.compat.v1.variable_scope('supervisor', reuse=tf.compat.v1.AUTO_REUSE):
        cells = [rnn_cell(module, hidden_dim) for _ in range(num_layers - 1)]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, H, sequence_length=T, dtype=tf.float32)
        S = tf.compat.v1.layers.dense(outputs, hidden_dim, activation=tf.nn.sigmoid)
    return S

def build_discriminator(H, T, module, hidden_dim, num_layers):
    with tf.compat.v1.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):
        cells = [rnn_cell(module, hidden_dim) for _ in range(num_layers)]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, H, sequence_length=T, dtype=tf.float32)
        Y_hat = tf.compat.v1.layers.dense(outputs, 1, activation=None)
    return Y_hat
