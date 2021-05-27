# This script defines a function for finding co expression of genes of interest

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
import numpy as np

# Input: a string soi which includes the name of specified gene,
# an int topcount which returns the top __ correlated genes
def compute_genecor(soi, topcount):
    # Load dataset as pandas dataframe, edited to add "geneid" to A1
    df = pd.read_csv('esc_all_edit.csv')
    df = df.set_index('Cell Line')

    # Reset the index labels for the first col:
    # because numerical index helpful opposed to string labels
    df.reset_index(drop=True, inplace=True)

    # Get the whole column associated with gene of interest
    soi_col = df.loc[:, soi]

    # Use correlation with function to compare the column to all other genes in dataset
    all_cor = df.corrwith(soi_col)

    # Sort values so highest correlation is at the beginning of the returned series
    sorted = all_cor.sort_values(ascending=False)
    # print(sorted)

    # Initiatlize counters, titles, and values for the graphical plot
    counter = 0
    titles = []
    values = []

    string = sorted.index[0]

    for i in range(len(sorted)):
        if counter == topcount:  # Find top ten genes
            break
        string = sorted.index[i]
        if string[:2] == 'RP' or string[:2] == 'RL' or string == soi:
            continue
        titles += [string]
        value = sorted[i]
        values += [value]
        counter = counter + 1

    y_pos = np.arange(len(titles))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, titles, fontsize=6)
    plt.ylabel('Expression')
    plt.title('Gene Co Expression with ' + soi)

    plt.show()

def find_key(findme):
# FIND GENES BY STR FRAGMENT
    df = pd.read_csv('esc_all_edit.csv')
    df = df.set_index('Cell Line')

    keys = df.keys()
    pattern = findme
    matching = fnmatch.filter(keys, pattern)
    print('These keys exist in dataframe: ', matching)

# TEST FUNCTION CALL
compute_genecor('SOX2', 20)
#find_key('NANOG')

