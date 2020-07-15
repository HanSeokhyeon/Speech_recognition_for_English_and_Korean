"""
permutation feature importance
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open("pfi_mel40_mfcc40_5.pkl", "rb") as f:
    data = pickle.load(f)

max_cer, cers = data[0], data[1:]

df = pd.DataFrame(columns=['feature', 'pfi'])
for i, cer in enumerate(cers):
    pfi = []
    for j, c in enumerate(cer):
        now_pfi = (c-max_cer)/max_cer*100
        pfi.append(["f{}".format(i), now_pfi[0]])
    # print(pfi)
    pfi = pd.DataFrame(pfi, columns=['feature', 'pfi'])
    df = df.append(pfi, ignore_index=True)
    # print(df)

# print(df)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 6.4))

# plt.plot(pfi, color='black')
sns.barplot(x='feature', y='pfi', data=df, ci=None, color='gray')

# plt.xlim(0, 79)
plt.xticks([20, 60], ["$X_{0...39}$", "$MFCC_{0...39}$"])
plt.axvline(39.5, color='black', alpha=0.7)


plt.show()
