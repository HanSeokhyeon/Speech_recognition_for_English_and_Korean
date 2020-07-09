import yaml
from util.timit_dataset import load_dataset
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load config file for experiment
parser = argparse.ArgumentParser(description='Training script for LAS on TIMIT .')
parser.add_argument('config_path', metavar='config_path', type=str, help='Path to config file for training.')
paras = parser.parse_args()
config_path = paras.config_path
conf = yaml.load(open(config_path, 'r'))

# Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# X : Padding to shape [num of sample, max_timestep, feature_dim]
# Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(**conf['meta_variable'])

X = np.concatenate(X_test, axis=0)[:, :80]
dfx = pd.DataFrame(X, columns=range(80))
corr = dfx.corr()

ticks = [0, 10, 20, 30,
         40, 48, 56, 64,
         72, 80]
tickslabel = ["$X_0$", "$X_{10}$", "$X_{20}$", "$X_{30}$",
              "$G_0$", "$G_8$", "$G_{16}$", "$G_{24}$",
              "$T_0$", "$T_8$"]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(8, 6.5))

ax = sns.heatmap(dfx.corr(), annot=False, cmap='gray', xticklabels=tickslabel, yticklabels=tickslabel)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

plt.show()

