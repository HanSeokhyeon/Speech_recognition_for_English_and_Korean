"""
그림 2. 40 밴드 멜 필터뱅크와 제안하는 방법에서 사용하는 32 밴드 감마톤 필터뱅크의 주파수 응답
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
mel_filter_freq = librosa.filters.mel(sr=16000, n_fft=400, n_mels=32).T
plt.plot(mel_filter_freq, color='gray')

plt.title("32-band Mel filterbank")

plt.xlim(0, 201)
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200], [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
plt.xlabel("Frequency [Hz]")

plt.ylim(0)
plt.yticks([])
plt.ylabel("Gain")


plt.subplot(2, 1, 2)
gammatone_filter = np.fromfile("fft.raw", dtype=np.float64)
gammatone_filter = gammatone_filter.reshape(32, 2048)
gammatone_filter_freq = np.abs(np.fft.fft(gammatone_filter, axis=1))[:1024].T
plt.plot(gammatone_filter_freq, color='gray')

plt.title("32-band gammatone filterbank")

plt.xlim(0, 1024)
plt.xticks([0, 128, 256, 384, 512, 640, 768, 896, 1024], [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
plt.xlabel("Frequency [Hz]")

plt.ylim(0)
plt.yticks([])
plt.ylabel("Gain")


plt.show()
