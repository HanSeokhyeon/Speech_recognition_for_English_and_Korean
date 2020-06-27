"""
그림 4. 위상 조정 전(위)과 후(아래)의 스파이크그램
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 10))

freq_value = [20, 48, 80, 116, 156, 201, 250, 306, 367, 436, 513, 599, 695, 801, 921, 1053, 1202, 1367, 1551, 1757,
              1987, 2243, 2528, 2847, 3202, 3599, 4041, 4534, 5085, 5698, 6383, 7147]

delay_value = [371, 326, 297, 275, 307, 279, 255, 209, 174, 165, 140, 120,  92,  90,  78,
               68,  60,  47, 41, 41, 36, 32, 25, 28, 20, 20, 16, 16, 14, 14,
               10, 9]

point = 16200
length = 16000 * 0.03
scale = 0.65

delay_now = [point-v+max(delay_value)-91 for v in delay_value]

x = np.fromfile("SI648_spike.raw", dtype=np.float64)
x = x.reshape(-1, 4)

num = np.fromfile("SI648_num.raw", dtype=np.int32)
num_acc = [sum(num[:i+1]) for i in range(len(num))]
for i, v in enumerate(num_acc[:-1]):
    x[num_acc[i]:num_acc[i+1], 2] += 12288*(i+1)

x_phase = np.copy(x)
x_phase[:, 2] = np.vectorize(lambda a, b: b + delay_value[int(a)])(x_phase[:, 0], x_phase[:, 2])

x[:, 0] = np.vectorize(lambda a: freq_value[int(a)])(x[:, 0])
x_phase[:, 0] = np.vectorize(lambda a: freq_value[int(a)])(x_phase[:, 0])

###################################################################

plt.subplot(2, 1, 1)

plt.scatter(x=x[:, 2], y=x[:, 0], s=x[:, 1]**scale, c='black')
plt.plot(delay_now + [delay_now[-1]+delay_value[-1]], freq_value+[8000])

plt.title("Before phase alignment")

start = point; end = start + length

plt.xticks(np.arange(start, end+1, 80), np.arange(0, 31, 5))
plt.xlim(start, end)
plt.xlabel("Time [ms]")

plt.yticks([freq_value[i] for i in [10, 20, 30]], [10, 20, 30])
plt.ylim(0, 8000)
plt.ylabel("Band")

plt.grid(axis='x')

###################################################################

plt.subplot(2, 1, 2)

plt.scatter(x=x_phase[:, 2], y=x_phase[:, 0], s=x_phase[:, 1]**scale, c='black')
plt.plot([delay_now[-1]+delay_value[-1]]*33, freq_value+[8000])

plt.title("After phase alignment")

start = point; end = start + length

plt.xticks(np.arange(start, end+1, 80), np.arange(0, 31, 5))
plt.xlim(start, end)
plt.xlabel("Time [ms]")

plt.yticks([freq_value[i] for i in [10, 20, 30]], [10, 20, 30])
plt.ylim(0, 8000)
plt.ylabel("Band")

plt.grid(axis='x')

###################################################################

plt.show()
