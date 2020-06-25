"""
그림 3. 음성 신호의 스펙트로그램, 멜-스펙트로그램과 스파이크그램의 예
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 10))

###################################################################

plt.subplot(3,1,1)
signal = np.fromfile("SI648.WAV", dtype=np.int16)[512:]
spectrogram, frequency, time, _ = plt.specgram(x=signal, Fs=16000, NFFT=400, noverlap=160)

plt.title("Spectrogram")

plt.xlabel("Time [s]")

plt.yticks([0, 2000, 4000, 6000, 8000])
plt.ylabel("Frequency [Hz]")

###################################################################

plt.subplot(3,1,2)
mel_spectrogram = librosa.feature.melspectrogram(y=signal/32767.5, sr=16000, n_fft=400, hop_length=160, n_mels=40)
display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', sr=16000, hop_length=160,
                         x_axis='time')

mel_filter_center_freq = np.argmax(librosa.filters.mel(sr=16000, n_fft=400, n_mels=40).T, axis=0) * 40

plt.title("Mel spectrogram")

plt.xticks([1, 2, 3])
plt.xlabel("Time [s]")

plt.yticks([mel_filter_center_freq[f] for f in [9, 19, 29, 39]], [10, 20, 30, 40])
plt.ylabel("Band")

###################################################################

plt.subplot(3,1,3)




plt.show()
