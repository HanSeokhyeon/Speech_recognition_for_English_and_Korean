"""
그림 1. 제안하는 방법에서 사용하는 감마톤 필터의 파형

기존 그림 파형 4개 -> 새로운 그림 파형 6개
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
ax2 = ax.twinx()
gammatone_filter = np.fromfile("../fft.raw", dtype=np.float64)
gammatone_filter = gammatone_filter.reshape(32, 2048)

freq_value = [20, 48, 80, 116, 156, 201, 250, 306, 367, 436, 513, 599, 695, 801, 921, 1053, 1202, 1367, 1551, 1757,
              1987, 2243, 2528, 2847, 3202, 3599, 4041, 4534, 5085, 5698, 6383, 7147]

gap = 0.3
filter = [1, 3, 6, 10, 15, 21]
for i, v in enumerate(filter):
    ax.plot(gammatone_filter[v] + gap * i)
    ax2.plot(gammatone_filter[v] + gap*i)

ax.set_xticks([0, 400, 800, 1200, 1600, 2000])
ax.set_xticklabels([0, 25, 50, 75, 100, 125])
ax.set_xlabel("Time [ms]")

ax.set_yticks(np.arange(0, 6)*gap)
ax.set_yticklabels([freq_value[v] for v in filter])
ax.set_ylabel("Frequency [Hz]")

ax2.set_yticks(np.arange(0, 6)*gap)
ax2.set_yticklabels(filter)
ax2.set_ylabel("Band")

plt.show()
