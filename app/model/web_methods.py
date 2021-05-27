from joblib import load
import numpy as np
import pandas as pd


def get_probability(expr_list, model_file):
    '''
    Takes a list of 20 expression values (floats, in the correct order)
    and returns the probability our classifier outputs.

    :param expr_list: list of 20 expression values, in order (list of floats)
    :param model_file: joblib file where classifer is stored on disk
    :return: probability value (float)
    '''
    clf = load(model_file)
    expr_arr = np.reshape(expr_list, (1, -1))
    prob = clf.predict_proba(expr_arr)
    return prob[0][1]


def get_percentile(prob, probs_file):
    '''
    Takes a probability and precomputed probabilities file, and returns the percentile of the single value.

    :param prob: Single probability output by get_probability (float)
    :param probs_file: Precomputed file with two columns, the second being probability values of other iPSCs
    :return: percentile rank of the given probability (float)
    '''
    ipsc_probs = pd.read_csv(probs_file)
    new_row = pd.DataFrame([[0, prob]])
    ipsc_probs = ipsc_probs.append(new_row)
    ipsc_probs['percentile'] = ipsc_probs[1].rank(pct=True)
    percentile = ipsc_probs['percentile'].iloc[-1]
    return percentile


def get_prob_and_percentile(expr_list, model_file, probs_file):
    '''
    Takes a list of 20 expression values (floats), the joblib file storing the model, and a precomputed
    probabilities file. Returns a tuple of (probability, percentile), both floats.

    :param expr_list: list of 20 expression values, in order (list of floats)
    :param model_file: joblib file where classifer is stored on disk
    :param probs_file:
    :return: probability value (float)
    '''
    prob = get_probability(expr_list, model_file)
    percentile = get_percentile(prob, probs_file)
    return prob, percentile