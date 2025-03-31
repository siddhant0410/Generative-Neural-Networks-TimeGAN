import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def discriminative_score_metrics(ori_data, generated_data):
    
    ori_labels = np.ones(len(ori_data))
    gen_labels = np.zeros(len(generated_data))

    ori_flat = [x.flatten() for x in ori_data]
    gen_flat = [x.flatten() for x in generated_data]

    X = np.concatenate([ori_flat, gen_flat], axis=0)
    y = np.concatenate([ori_labels, gen_labels], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    discriminative_score = np.abs(0.5 - acc) * 2
    return discriminative_score