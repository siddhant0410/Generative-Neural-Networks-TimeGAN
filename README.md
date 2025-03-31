## Overview

TimeGAN combines adversarial training with supervised sequence learning and autoencoding to generate synthetic time-series that retain both **feature-level fidelity** and **temporal consistency**.

### Key Components:
- **Embedder & Recovery Network**: Learns a latent representation and reconstructs sequences.
- **Generator & Discriminator**: Adversarially trains to produce indistinguishable latent sequences.
- **Supervisor**: Guides temporal dynamics by learning latent step transitions.

## Datasets

### 1. **ARKF.csv** (Real-world financial data)
- Daily prices for ARK Fintech ETF: Open, High, Low, Close, Adj Close, Volume
- ~293 observations

### 2. **heart_rate.csv** (Real-world physiological data)
- Multichannel ECG/PPG sequences

### 3. **Synthetic Sine Data**
- Multi-dimensional sine waves generated programmatically
- Used as a sanity check for temporal consistency

## 🚀 How to Run

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Command to Run
Once the dependencies have been installed run the below command
```bash
python main_timegan.py \
  --data_name ARKF \
  --seq_len 24 \
  --module gru \
  --hidden_dim 24 \
  --num_layer 3 \
  --iterations 5000 \
  --batch_size 128 \
  --metric_iteration 10
```

## Results :
### python main_timegan.py --data_name heart --seq_len 24 --module gru --hidden_dim 24 --num_layer 2 --iterations 1000 --batch_size 128 --metric_iteration 10
Iteration 1: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 2: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 3: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 4: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 5: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 6: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 7: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 8: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 9: Discriminative Score = 1.0000, Predictive Score = 0.2070
Iteration 10: Discriminative Score = 1.0000, Predictive Score = 0.2070

Average Discriminative Score: 1.0000 ± 0.0000
Average Predictive Score: 0.2070 ± 0.0000
![5)](https://github.com/user-attachments/assets/d3e7aba2-0e61-4e64-a8f7-248faeb3a6a6)

### python main_timegan.py --data_name ARKF --seq_len 24 --module gru --hidden_dim 24 --num_layer 3 --iterations 5000 --batch_size 128 --metric_iteration 10 

Evaluating...
Iteration 1: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 2: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 3: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 4: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 5: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 6: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 7: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 8: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 9: Discriminative Score = 1.0000, Predictive Score = 0.1635
Iteration 10: Discriminative Score = 1.0000, Predictive Score = 0.1635

Average Discriminative Score: 1.0000 ± 0.0000
Average Predictive Score: 0.1635 ± 0.0000
![Uploading 7.).png…]()

