
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset as pandas dataframe, edited to add "geneid" to A1
df = pd.read_csv('esc_all_edit.csv')

# Transpose col and row, keeping "geneid" labels intact:
# because visuz.stat.corr_mat() function requires
# cell lines in cols and geneid in rows
df = df.set_index('Cell Line').T

# Reset the index labels for the first col:
# because visuz.stat.corr_mat() function requires
# a reset numerical index opposed to string labels
df.reset_index(drop=True, inplace=True)

#print(df.head())

# Use df.corr to calculate correlation for each cell
corr = df.corr(method='pearson')
plt.figure(figsize=(20,10))

# Plot heatmap
sns.heatmap(corr, vmin=corr.values.min(), vmax=1, square=False, cmap="Reds", linewidths=0.1, annot=False, annot_kws={"fontsize":8})
plt.show()
