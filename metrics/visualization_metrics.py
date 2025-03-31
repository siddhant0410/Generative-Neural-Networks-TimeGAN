import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualization(ori_data, generated_data, data_name='synthetic'):
    ori_flat = [seq.flatten() for seq in ori_data]
    gen_flat = [seq.flatten() for seq in generated_data]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
    tsne_results = tsne.fit_transform(np.concatenate((ori_flat, gen_flat), axis=0))

    half = len(ori_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:half, 0], tsne_results[:half, 1], label='Original', alpha=0.6)
    plt.scatter(tsne_results[half:, 0], tsne_results[half:, 1], label='Synthetic', alpha=0.6)
    plt.title(f't-SNE Visualization: {data_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
