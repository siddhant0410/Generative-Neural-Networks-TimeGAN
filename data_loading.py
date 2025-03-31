# data_loading.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(name, seq_len):
    if name == 'ARKF':
        path = './data/ARKF.csv'
    elif name == 'heart':
        path = './data/heart_rate.csv'
    else:
        raise ValueError('Dataset not supported.')

    data = pd.read_csv(path)
    data = data.dropna()

    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)

    sequences = []
    for i in range(0, len(norm_data) - seq_len):
        sequence = norm_data[i:i+seq_len]
        sequences.append(sequence)

    return [np.array(seq) for seq in sequences]