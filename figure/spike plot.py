import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


n_band = 32
n_structure = 4

feature_frame = 16000   # 1600 for emotion 400 for phoneme

freq_value = [20, 48, 80, 116, 156, 201, 250, 306, 367, 436, 513, 599, 695, 801, 921, 1053, 1202, 1367, 1551, 1757,
              1987, 2243, 2528, 2847, 3202, 3599, 4041, 4534, 5085, 5698, 6383, 7147]
time_value = [0, 1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000] # for emotion
# time_value = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400] # for phoneme

def main():
    spike_filename = "./SPike_rock_0.raw"
    # spike_filename = "D:/Spike_DB/TIMIT/TIMIT.raw"

    tmp = 0
    x = np.fromfile(spike_filename, dtype=np.float64)
    x = np.reshape(x, (-1, n_structure))

    # x = x[18882:,:] # for phoneme because mute
    x=x[:,:]
    for i in range(len(x)):
        if x[i][3] > 20:
           tmp = i
           break
        elif i == len(x)-1:
            tmp = i

    x = x[:tmp,:]
    for i in range(len(x)):
        x[i][0] = freq_value[int(x[i][0])]

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')

    # plt.figure(figsize=(6.60, 4.00))
    plt.figure(figsize=(6.60, 3.80))

    plt.scatter(x=x[:,2], y=x[:,0], s=x[:,1]*0.0005, c='black')
    plt.xlim(0, feature_frame)
    plt.ylim(0, freq_value[-1])

    plt.xticks(ticks=[0, 3200, 6400, 9600, 12800, 16000], labels=[0, 0.2, 0.4, 0.6, 0.8, 1], fontproperties=font, fontsize=14)  # for emotion
    # plt.xticks(ticks=[0, 80, 160, 240, 320, 400], labels=[0, 5, 10, 15, 20, 25], fontproperties=font, fontsize=14)  # for phoneme
    plt.yticks(ticks=[436, 1757, 5698], labels=[10, 20, 30])

    # plt.title("Spikegram", fontproperties=font, fontsize=16)
    # plt.xlabel("Time [s]", fontproperties=font, fontsize=14)
    plt.ylabel("Band", fontproperties=font, fontsize=14)

    # for y in freq_value:
    #     plt.axhline(y=y, linestyle='--', linewidth=1.0, alpha=0.5, color='black')
    # for x in time_value:
    #     plt.axvline(x=x, linestyle='--', linewidth=1.0, alpha=0.5, color='black')

    fig = plt.gcf()
    plt.show()

    # fig.savefig("emo_freq.png")

    print('finish')
    return 0

if __name__ == '__main__':
    main()