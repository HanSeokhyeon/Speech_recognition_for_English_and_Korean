"""
permutation feature importance
"""
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 8))

with open("pfi_mel40_mfcc40_5.pkl", "rb") as f:
    data = pickle.load(f)

max_cer, cers = data[0][0], data[1:]
cers = cers[40:] + cers[:40]
if type(max_cer) == tuple:
    max_cer = max_cer[0]

df = pd.DataFrame(columns=['Feature', 'PFI'])
for i, cer in enumerate(cers):
    pfi = []
    for j, c in enumerate(cer):
        now_pfi = (c-max_cer)/max_cer*100
        pfi.append(["f{}".format(i), now_pfi])
    pfi = pd.DataFrame(pfi, columns=['Feature', 'PFI'])
    df = df.append(pfi, ignore_index=True)

plt.subplot(2, 1, 1)

sns.barplot(x='Feature', y='PFI', data=df, ci=None, color='gray')

plt.xticks([20, 60], ["$Mel_{0...39}$", "$MFCC_{0...39}$"])
plt.axvline(39.5, color='black', alpha=0.7)

plt.ylim(-1.5, 4.5)

with open("pfi_mel40_spikegram40_5.pkl", "rb") as f:
    data = pickle.load(f)

max_cer, cers = data[0][0], data[1:]

df = pd.DataFrame(columns=['Feature', 'PFI'])
for i, cer in enumerate(cers):
    pfi = []
    for j, c in enumerate(cer):
        now_pfi = (c-max_cer)/max_cer*100
        pfi.append(["f{}".format(i), now_pfi])
    pfi = pd.DataFrame(pfi, columns=['Feature', 'PFI'])
    df = df.append(pfi, ignore_index=True)

plt.subplot(2, 1, 2)

sns.barplot(x='Feature', y='PFI', data=df, ci=None, color='gray')

plt.xticks([20, 56, 76], ["$Mel_{0...39}$", "$G_{0...31}$", "$T_{0...7}$"])
plt.axvline(39.5, color='black', alpha=0.7)
plt.axvline(71.5, color='black', alpha=0.7)

plt.ylim(-1.5, 4.5)

plt.show()
