from util.timit_dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 8))

X_train, _, _, _, _, _ = load_dataset(data_path='dataset/TIMIT/timit_mel_spikegram_240.pkl')

X = np.concatenate(X_train, axis=0)[:, :72]
mel = X[:, :40].T
spike = X[:, 40:].T

mel_x = np.repeat(range(0, 40), mel.shape[1])
mel_y = np.reshape(mel, -1)

plt.subplot(2, 1, 1)

h, _, _, im = plt.hist2d(mel_x, mel_y, bins=(40, 1000), range=[[0, 40], [-3, 3]], cmap='binary', vmax=4000)
fig.colorbar(im)
plt.title("Mel-spectrogram")
plt.xlabel("Band")
plt.ylabel("Value")

spike_x = np.repeat(range(0, 32), spike.shape[1])
spike_y = np.reshape(spike, -1)
plt.subplot(2, 1, 2)

h, _, _, im = plt.hist2d(spike_x, spike_y, bins=(32, 1000), range=[[0, 32], [-3, 3]], cmap='binary', vmax=4000)
fig.colorbar(im)

plt.title("Spikegram")
plt.xlabel("Band")
plt.ylabel("Value")

plt.show()
