from util.timit_dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 8), edgecolor='k')

X_train, _, _, _, _, _ = load_dataset(data_path='dataset/TIMIT/timit_mel_spikegram_240.pkl')

X = np.concatenate(X_train, axis=0)[:, :72]
mel = X[:, :40].T
spike = X[:, 40:].T

mel_x = np.repeat(range(0, 40), mel.shape[1])
mel_y = np.reshape(mel, -1)


hist_mel, _, _, _ = plt.hist2d(mel_x, mel_y, bins=(40, 1000), range=[[0, 40], [-3, 3]], cmap='binary', vmax=4000)
hist_mel[np.where(hist_mel > 4000)] = 4000
hist_mel = (hist_mel / 4000).T

spike_x = np.repeat(range(0, 32), spike.shape[1])
spike_y = np.reshape(spike, -1)

hist_spike, _, _, _ = plt.hist2d(spike_x, spike_y, bins=(32, 1000), range=[[0, 32], [-3, 3]], cmap='binary', vmax=4000)
hist_spike[np.where(hist_spike > 4000)] = 4000
hist_spike = (hist_spike / 4000).T

plt.clf()

plt.subplot(2, 1, 1)

res = sns.heatmap(hist_mel, cmap='binary')
res.invert_yaxis()

for _, spine in res.spines.items():
    spine.set_visible(True)


plt.title("Mel-spectrogram")
plt.xlabel("Band")
plt.xticks([0, 10, 20, 30], [0, 10, 20, 30])
plt.ylabel("Value")
plt.yticks([1000//6, 1000//6*3, 1000//6*5], [-2, 0, 2])

plt.subplot(2, 1, 2)

res = sns.heatmap(hist_spike, cmap='binary')
res.invert_yaxis()

for _, spine in res.spines.items():
    spine.set_visible(True)


plt.title("Spikegram")
plt.xlabel("Band")
plt.xticks([0, 10, 20, 30], [0, 10, 20, 30])
plt.ylabel("Value")
plt.yticks([1000//6, 1000//6*3, 1000//6*5], [-2, 0, 2])

plt.show()
