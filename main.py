
import argparse
import numpy as np
from data_loading import load_data
from timegan import timegan
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['ARKF', 'heart'], default='ARKF', help='Dataset to use')
    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length')
    parser.add_argument('--module', choices=['lstm', 'gru'], default='gru', help='RNN module')
    parser.add_argument('--hidden_dim', type=int, default=24, help='Hidden dimensions')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of layers')
    parser.add_argument('--iterations', type=int, default=5000, help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--metric_iteration', type=int, default=1, help='How many times to repeat metrics evaluation')
    args = parser.parse_args()

    ori_data = load_data(args.data_name, args.seq_len)

    
    parameters = {
        'module': args.module,
        'hidden_dim': args.hidden_dim,
        'num_layer': args.num_layer,
        'iterations': args.iterations,
        'batch_size': args.batch_size
    }

    
    generated_data = timegan(ori_data, parameters)

    print('Evaluating...')
    dis_scores = []
    pred_scores = []
    for i in range(args.metric_iteration):
        dis = discriminative_score_metrics(ori_data, generated_data)
        pred = predictive_score_metrics(ori_data, generated_data)
        dis_scores.append(dis)
        pred_scores.append(pred)
        print(f'Iteration {i+1}: Discriminative Score = {dis:.4f}, Predictive Score = {pred:.4f}')

    print(f'\nAverage Discriminative Score: {np.mean(dis_scores):.4f} ± {np.std(dis_scores):.4f}')
    print(f'Average Predictive Score: {np.mean(pred_scores):.4f} ± {np.std(pred_scores):.4f}')

    # Visualize
    visualization(ori_data, generated_data, args.data_name)
