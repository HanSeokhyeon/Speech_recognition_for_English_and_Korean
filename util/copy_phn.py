import numpy as np
from shutil import copy

list_names = ["timit_dataset_list/TRAIN_list.csv", "timit_dataset_list/TEST_developmentset_list.csv", "timit_dataset_list/TEST_coreset_list.csv"]

for list_name in list_names:
    dataset_list = np.loadtxt(list_name, delimiter=',', dtype=np.str)

    for filename in dataset_list:
        original_phn = "../dataset/TIMIT/{}.PHN".format(filename)
        destination_phn = "../dataset/TIMIT_spikegram/{}.PHN".format(filename)

        copy(original_phn, destination_phn)

        print(filename)


