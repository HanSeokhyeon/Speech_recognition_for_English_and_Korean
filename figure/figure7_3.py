from util.timit_dataset import load_dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 11))

# Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# X : Padding to shape [num of sample, max_timestep, feature_dim]
# Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(data_path='dataset/TIMIT/timit_mfcc_mel_240.pkl')

X = np.concatenate(X_test, axis=0)[:, :80]
X = np.concatenate([X[:, 40:], X[:, :40]], axis=1)
dfx = pd.DataFrame(X, columns=range(80))
corr = dfx.corr().abs()
corr = corr.iloc[::-1]

ticks_x = [0, 10, 20, 30,
         40, 48, 56, 64,
         72, 80]
ticks_y = [0, 8,
           16, 24, 32, 40,
           50, 60, 70, 80]
tickslabel = ["$X_0$", "$X_{10}$", "$X_{20}$", "$X_{30}$",
              "$G_0$", "$G_8$", "$G_{16}$", "$G_{24}$",
              "$T_0$", "$T_8$"]

ticks = [20, 60]
tickslabel = ["$X_{0...39}$", "$MFCC_{0...39}$"]

mask = np.zeros_like(corr)
for i in range(80):
    mask[i, :-(i+1)] = True

plt.subplot(2, 1, 1)

ax = sns.heatmap(corr, annot=False, mask=mask, cmap='binary', xticklabels=tickslabel, yticklabels=list(reversed(tickslabel)), vmin=0, vmax=1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

plt.xticks(rotation=0)

for h in [0, 40, 80]:
    plt.axhline(h, color='black', alpha=0.3)
    plt.axvline(h, color='black', alpha=0.3)

# Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# X : Padding to shape [num of sample, max_timestep, feature_dim]
# Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(data_path='dataset/TIMIT/timit_mel_spikegram_240.pkl')

X = np.concatenate(X_test, axis=0)[:, :80]
dfx = pd.DataFrame(X, columns=range(80))
corr = dfx.corr().abs()
corr = corr.iloc[::-1]

ticks_x = [0, 10, 20, 30,
          40, 48, 56, 64,
          72, 80]
ticks_y = [0, 8,
           16, 24, 32, 40,
           50, 60, 70, 80]
tickslabel = ["$X_0$", "$X_{10}$", "$X_{20}$", "$X_{30}$",
              "$G_0$", "$G_8$", "$G_{16}$", "$G_{24}$",
              "$T_0$", "$T_8$"]

ticks_x = [20, 56, 76]
ticks_y = [4, 24, 60]
tickslabel = ["$X_{0...39}$", "$G_{0...31}$", "$T_{0...7}$"]

mask = np.zeros_like(corr)
for i in range(80):
    mask[i, :-(i+1)] = True

plt.subplot(2, 1, 2)

ax = sns.heatmap(corr, annot=False, mask=mask, cmap='binary', xticklabels=tickslabel, yticklabels=list(reversed(tickslabel)), vmin=0, vmax=1)
ax.set_xticks(ticks_x)
ax.set_yticks(ticks_y)

plt.xticks(rotation=0)

for h in [0, 8, 40, 80]:
    plt.axhline(h, color='black', alpha=0.3)
for v in [0, 40, 72, 80]:
    plt.axvline(v, color='black', alpha=0.3)

plt.show()
