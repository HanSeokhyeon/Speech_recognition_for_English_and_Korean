import os
import sys
import glob
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle
import numpy as np
import python_speech_features as features
import json

if len(sys.argv) != 3:
    print('Usage:python3 preprocess.py <kspon directory> <output_file>')


##### SCRIPT META VARIABLES #####
txt_file_postfix = '.txt'
pcm_file_postfix = '.pcm'

##### Validation split #####
# default using 5% of data as validation
test_split = 0.8
val_split = 0.1

data_type = 'float32'

paths = sys.argv[1] + '/*'
directories = glob.glob(paths)
directories.sort()
if '.DS_Store' in directories:
    os.remove("{}/{}".format(paths, '.DS_Store'))
    directories.remove('.DS_Store')
train_file_num = 3696
valid_file_num = 400
test_file_num = 192
train_path = glob.glob("{}/*".format(directories[0])) \
				+ glob.glob("{}/*".format(directories[1])) \
				+ glob.glob("{}/*".format(directories[2])) \
				+ glob.glob("{}/*".format(directories[3]))
train_path = sorted(set([fname[:-4] for fname in train_path]))
train_path = train_path[:train_file_num]
valid_test_path = glob.glob("{}/*".format(directories[4]))
valid_test_path = sorted(set([fname[:-4] for fname in valid_test_path]))
valid_path = valid_test_path[:valid_file_num]
test_path = valid_test_path[valid_file_num:valid_file_num+test_file_num]
parent = os.path.abspath(os.path.join(paths[:-2], os.pardir))
target_path = os.path.join(parent, sys.argv[2])

y_label = {}


def create_mfcc(filename):
	"""Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
	"""

	rate, sample = 16000, np.fromfile(filename, dtype=np.int16)

	mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
	preemph=0.97, appendEnergy=True)
	d_mfcc = features.delta(mfcc, 2)
	a_mfcc = features.delta(d_mfcc, 2)

	out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)

	return out, out.shape[0]


def calc_norm_param(X):
	"""Assumes X to be a list of arrays (of differing sizes)"""
	total_len = 0
	mean_val = np.zeros(X[0].shape[1]) # 39
	std_val = np.zeros(X[0].shape[1]) # 39
	for obs in X:
		obs_len = obs.shape[0]
		mean_val += np.mean(obs,axis=0) * obs_len
		std_val += np.std(obs, axis=0) * obs_len
		total_len += obs_len

	mean_val /= total_len
	std_val /= total_len

	return mean_val, std_val, total_len


def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X


def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X


def remove_bracket(s):
	result = []
	find = [False]*4
	for c in s:
		if c == '(' and not find[0]:
			find[0] = True
		elif c == ')' and not find[1]:
			find[1] = True
		elif c == '(' and not find[2]:
			find[2] = True
		elif c == ')' and not find[3]:
			find[3] = True
		elif find[0] and not find[1]:
			continue
		else:
			result.append(c)
	return s


def bracket_filter(sentence):
    new_sentence = str()
    flag = False

    for ch in sentence:
        if ch == '(' and flag == False:
            flag = True
            continue
        if ch == '(' and flag == True:
            flag = False
            continue
        if ch != ')' and flag == False:
            new_sentence += ch
    return new_sentence


def special_filter(sentence):
    SENTENCE_MARK = ['.', '?', ',', '!']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';']
    import re
    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            # o/, n/ 등 처리
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx+1] == '/':
                continue
        if ch == '%':
            new_sentence += '퍼센트'
        elif ch == '#':
            new_sentence += '샾'
        elif ch not in EXCEPT:
            new_sentence += ch
    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def add_y_label(y):
    for c in y:
        if c not in y_label:
            y_label[c] = [len(y_label), 1]
        else:
            y_label[c][1] += 1


def preprocess_dataset(file_list):
    i = 0
    X = []
    Y = []

    for fname in file_list[:]:
        txt_fname = "{}{}".format(fname, txt_file_postfix)
        pcm_fname = "{}{}".format(fname, pcm_file_postfix)

        X_val, total_frames = create_mfcc(pcm_fname)
        total_frames = int(total_frames)

        X.append(X_val)

        y = np.loadtxt(txt_fname, dtype=str, encoding='cp949')
        if y.size > 1:
            y_origin = " ".join(y)
        else:
            y_origin = str(y)
        y_remove = special_filter(bracket_filter(y_origin))

        add_y_label(y_remove)

        Y.append(y_remove)

        i += 1
        print('file No.', i, end='\r', flush=True)

    print('Done')
    return X, Y


def char2index(ys):
    new_ys = []
    for s in ys:
        new_y = []
        for c in s:
            new_y.append(y_label[c][0])
        new_ys.append(np.array(new_y))
    return new_ys


##### PREPROCESSING #####
print()

print('Preprocessing train data...')
X_train, y_train = preprocess_dataset(train_path)
print('Preprocessing valid data...')
X_valid, y_valid = preprocess_dataset(valid_path)
print('Preprocessing test data...')
X_test, y_test = preprocess_dataset(test_path)
print('Preprocessing completed.')

X_train_max = max(X_train, key=np.shape).shape[0]
X_valid_max = max(X_valid, key=np.shape).shape[0]
X_test_max = max(X_test, key=np.shape).shape[0]
X_max = max(X_train_max, X_valid_max, X_test_max)
print('Max timestep: {}'.format(X_max))

with open("korean_labels.json", 'w', encoding='UTF-8-sig') as f:
    json.dump(y_label, f, ensure_ascii=False)

y_train = char2index(y_train)
y_valid = char2index(y_valid)
y_test = char2index(y_test)

y_train_max = max(y_train, key=np.shape).shape[0]
y_valid_max = max(y_valid, key=np.shape).shape[0]
y_test_max = max(y_test, key=np.shape).shape[0]
y_max = max(y_train_max, y_valid_max, y_test_max)
print('Max label len: {}'.format(y_max))

print()
print('Collected {} training instances (should be 3696 in complete TIMIT )'.format(len(X_train)))
print('Collected {} validating instances (should be 400 in complete TIMIT )'.format(len(X_valid)))
print('Collected {} testing instances (should be 192 in complete TIMIT )'.format(len(X_test)))

print()
print('Normalizing data to let mean=0, sd=1 for each channel.')

mean_val, std_val, _ = calc_norm_param(X_train)

X_train = normalize(X_train, mean_val, std_val)
X_valid	= normalize(X_valid, mean_val, std_val)
X_test 	= normalize(X_test, mean_val, std_val)

X_train = set_type(X_train, data_type)
X_valid	= set_type(X_valid, data_type)
X_test 	= set_type(X_test, data_type)

print()
print('Saving data to ',target_path)
with open(target_path + '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train, y_train, X_valid, y_valid, X_test, y_test],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)

print()
print('Preprocessing completed in {:.3f} secs.'.format(timeit.default_timer() - program_start_time))
