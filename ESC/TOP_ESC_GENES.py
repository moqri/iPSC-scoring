import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('esc_all_sig.csv')
num_col = df.shape[1]

exp_counts = df.mean(0)
sorted = exp_counts.sort_values(ascending=False)

counter = 0
titles = []
values = []

string = sorted.index[0]
print(string[:2])


for i in range(len(sorted)):
    if counter == 10:
        break
    string = sorted.index[i]
    if string[:2] == 'RP' or string[:2] == 'RL':
        continue
    titles += [string]
    value = sorted[i]
    values += [value]
    counter = counter + 1

y_pos = np.arange(len(titles))

plt.bar(y_pos, values, align='center', alpha=0.5)
plt.xticks(y_pos, titles, fontsize=5)
plt.ylabel('Expression')
plt.title('Gene Expression in ESCs')

plt.show()