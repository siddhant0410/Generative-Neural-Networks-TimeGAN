
import numpy as np
import tensorflow as tf

def extract_time(data):
    time = [len(seq) for seq in data]
    return time, max(time)

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = [np.random.uniform(0., 1., [batch_size, z_dim]) for _ in range(max_seq_len)]
    Z_mb = np.transpose(np.asarray(Z_mb), (1, 0, 2))
    for i in range(len(T_mb)):
        if T_mb[i] < max_seq_len:
            Z_mb[i, T_mb[i]:, :] = 0
    return Z_mb

def rnn_cell(module_name, hidden_dim):
    if module_name == 'gru':
        return tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim)
    elif module_name == 'lstm':
        return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_dim)
    else:
        raise ValueError("Unsupported RNN type")

def batch_generator(data, time, batch_size):
    idx = np.random.permutation(len(data))[:batch_size]
    seq_batch = [data[i] for i in idx]
    time_batch = [time[i] for i in idx]
    max_seq_len = max(time_batch)
    batch = np.zeros((batch_size, max_seq_len, data[0].shape[1]))
    for i, seq in enumerate(seq_batch):
        batch[i, :len(seq), :] = seq
    return batch, time_batch