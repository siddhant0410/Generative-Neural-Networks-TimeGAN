import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def predictive_score_metrics(ori_data, generated_data):
    ori_x = [seq[:-1].flatten() for seq in ori_data if len(seq) > 1]
    ori_y = [seq[1:].flatten() for seq in ori_data if len(seq) > 1]

    gen_x = [seq[:-1].flatten() for seq in generated_data if len(seq) > 1]
    gen_y = [seq[1:].flatten() for seq in generated_data if len(seq) > 1]

    model = Ridge()
    model.fit(gen_x, gen_y)

    pred_y = model.predict(ori_x)
    score = mean_absolute_error(ori_y, pred_y)
    return score