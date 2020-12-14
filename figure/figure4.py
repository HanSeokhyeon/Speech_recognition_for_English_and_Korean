"""
그림 4. 스파이크그램으로부터 K개의 주파수 기반 특성(위)과 L개의 시간 기반 특성(아래)을 구하는 과정
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(6, 10))

freq_value = [20, 48, 80, 116, 156, 201, 250, 306, 367, 436, 513, 599, 695, 801, 921, 1053, 1202, 1367, 1551, 1757,
              1987, 2243, 2528, 2847, 3202, 3599, 4041, 4534, 5085, 5698, 6383, 7147]

point = 10000
scale = 0.65

x = np.fromfile("SI648_spike.raw", dtype=np.float64)
x = x.reshape(-1, 4)

num = np.fromfile("SI648_num.raw", dtype=np.int32)
num_acc = [sum(num[:i+1]) for i in range(len(num))]
for i, v in enumerate(num_acc[:-1]):
    x[num_acc[i]:num_acc[i+1], 2] += 12288*(i+1)

x[:, 0] = np.vectorize(lambda a: freq_value[int(a)])(x[:, 0])

###################################################################

plt.subplot(2, 1, 1)

plt.scatter(x=x[:, 2], y=x[:, 0], s=x[:, 1]**scale, c='black')

plt.title("K spectral features")

start = point; end = start + 400

plt.xticks(np.arange(start, end+1, 80), np.arange(0, 26, 5))
plt.xlim(start, end)
plt.xlabel("Time [ms]")

plt.yticks([freq_value[i] for i in [10, 20, 30]], [10, 20, 30])
plt.ylim(0, 8000)
plt.ylabel("Band")

# 가로줄 8개
for freq in freq_value[::4]:
    plt.axhline(freq, color='black', linestyle='-.', linewidth=0.5)

###################################################################

plt.subplot(2, 1, 2)

plt.scatter(x=x[:, 2], y=x[:, 0], s=x[:, 1]**scale, c='black')

plt.title("L temporal features")

start = point; end = start + 400

plt.xticks(np.arange(start, end+1, 80), np.arange(0, 26, 5))
plt.xlim(start, end)
plt.xlabel("Time [ms]")

plt.yticks([freq_value[i] for i in [10, 20, 30]], [10, 20, 30])
plt.ylim(0, 8000)
plt.ylabel("Band")

# 세로줄
for time in np.arange(start, end+1, 40):
    plt.axvline(time, color='black', linestyle='-.', linewidth=0.5)

###################################################################

fig1 = plt.gcf()
plt.show()

fig1.savefig("figures/figure4.png")