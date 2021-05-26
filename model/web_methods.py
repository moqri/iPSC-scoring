from joblib import load
import numpy as np
import pandas as pd

def get_probability(expr_list, model_file):
    clf = load(model_file)
    expr_arr = np.reshape(expr_list, (1, -1))
    prob = clf.predict_proba(expr_arr)
    return prob[0][1]

def get_percentile(prob, probs_file):
    ipsc_probs = pd.read_csv(probs_file)
    new_row = pd.DataFrame([[0, prob]])
    ipsc_probs = ipsc_probs.append(new_row)
    ipsc_probs['percentile'] = ipsc_probs[1].rank(pct=True)
    percentile = ipsc_probs['percentile'].iloc[-1]
    return percentile