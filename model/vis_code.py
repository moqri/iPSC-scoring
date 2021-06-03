import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from joblib import dump
import model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from scipy import stats

def plot_1_2_roc():
    lr1 = model.LogisticModel('../data', 'train_set_v2.csv', 'test_set_v2.csv', feature_count=1)
    lr1.execute_all()
    lr1_disp = metrics.plot_roc_curve(lr1.clf, lr1.X_test, lr1.y_test, name="LinearRegression, 1 transcript")

    lr2 = model.LogisticModel('../data', 'train_set_v2.csv', 'test_set_v2.csv', feature_count=2)
    lr2.execute_all()
    metrics.plot_roc_curve(lr2.clf, lr2.X_test, lr2.y_test, ax=lr1_disp.ax_, name="LinearRegression, 2 transcripts")

    lr3 = model.LogisticModel('../data', 'train_set_v2.csv', 'test_set_v2.csv', feature_count=3)
    lr3.execute_all()
    metrics.plot_roc_curve(lr3.clf, lr3.X_test, lr3.y_test, ax=lr1_disp.ax_, name="LinearRegression, 3 transcripts")

    plt.title('ROC curves for 1 and 2 transcripts (features)')
    plt.savefig('../figures/roc_one_two.png')

def corr_1_20():
    lr19 = model.LogisticModel('../data', 'train_set_v3.csv', 'test_set_v3.csv', feature_count=2)
    lr19.execute_all()

    lr20 = model.LogisticModel('../data', 'train_set_v3.csv', 'test_set_v3.csv', feature_count=3)
    lr20.execute_all()

    spearman = stats.spearmanr(lr19.y_pred, lr20.y_pred)[0]
    print('spearman coefficient is', str(spearman))

def pluritest_spearman(feature_size_list):

    spear_vals = []

    for fs in feature_size_list:
        lr = model.LogisticModel('../data', 'train_set_v3.csv', 'test_set_v3.csv', feature_count=fs)
        lr.execute_all()

        pt_scores = pd.read_csv('../data/test_ipsc_names_pluritest.csv', index_col=0, header=None)
        index_list = pt_scores.index.tolist()
        X_test_ipsc = pd.read_csv('../data/test_ipsc.csv', index_col=0)
        X_test_ipsc = X_test_ipsc[X_test_ipsc.index.isin(index_list)].iloc[:,:lr.feature_count]
        y_ipsc_probs = lr.prediction_probs(lr.clf, X_test_ipsc)
        y_ipsc_positive = pd.DataFrame(y_ipsc_probs).iloc[:,1]

        pt_score_list = pt_scores[2].tolist()
        our_score_list = y_ipsc_positive.tolist()

        spearman = stats.spearmanr(pt_score_list, our_score_list)
        print('spearman coefficient for fs=' + str(fs) + ' is ' + str(spearman[0]))

        spear_vals += [spearman[0]]

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(feature_size_list, spear_vals, marker='d')
    plt.xlabel('number of transcript features in model')
    plt.ylabel('Spearman correlation coefficient with PluriTest')
    plt.title('Spearman correlation with PluriTest with increasing feature count')
    plt.savefig('../figures/spearman_corr_line_pluritest.png')


    '''
    # lr has m=20 features now
    plt.xlabel('PluriTest score')
    plt.ylabel('log-normalized StemDB score')
    plt.title("Model probabilities correlation with PluriTest")
    plt.scatter(pt_score_list, -1*np.log10([1-a for a in our_score_list]))
    plt.savefig('../figures/pluritest_correlation.png')
    '''

def feature_importance():
    lr = model.LogisticModel('../data', 'train_set_v3.csv', 'test_set_v3.csv')
    lr.execute_all()
    clf = lr.clf
    weights = clf.coef_[0]
    stds = lr.X_train.std(axis=0).to_numpy()
    std_weights = np.multiply(weights, stds)

    # Get gene ordering
    gene_names = pd.read_csv('../data/top20tf_ens.csv')['Gene name'].tolist()

    x = np.arange(len(gene_names))  # the label locations
    width = 0.35  # the width of the bars


    fig, ax = plt.subplots()
    plt.xticks(rotation=90)

    ax2 = ax.twinx()
    rects1 = ax.bar(x - width / 2, weights, width, color='orange', label='Coefficient Alone')
    rects2 = ax2.bar(x + width / 2, std_weights, width, color='red', label='Coefficient * StdDev')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Feature coefficient")
    ax2.set_ylabel("Feature coefficient * Feature StdDev")
    ax.set_title("Feature importance measures")
    ax.set_xticks(x)
    ax.set_xticklabels(gene_names)
    #plt.legend(['Coefficient', 'Coefficient x StdDev'])
    ax.legend(loc=[0.6,0.9])
    ax2.legend(loc=[0.6,0.8])

    fig.subplots_adjust(bottom=0.2)

    plt.savefig('../figures/feature_importances.png')


    '''
    fig, ax = plt.subplots()
    plt.bar([x for x in range(len(weights))], weights)
    x = np.arange(len(weights))
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    ax.set_xticklabels(gene_names)
    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    # plt.xlabel = "Gene"
    plt.title("Feature weights (coefficients)")
    plt.ylabel = "Feature weight in model"
    plt.savefig('../figures/feature_coefficients.png')
    '''


if __name__ == '__main__':
    plot_1_2_roc()
    pluritest_spearman(list(range(1,21)))
    feature_importance()
