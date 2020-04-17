# reference : https://github.com/Faur/TIMIT
# 			  https://github.com/jameslyons/python_speech_features/issues/53
import os
import sys
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle
import numpy as np
import librosa
# a python package for speech features at https://github.com/jameslyons/python_speech_features

if len(sys.argv) != 3:
	print('Usage: python3 preprocess.py <timit directory> <output_file>')

##### SCRIPT META VARIABLES #####
phn_file_postfix = '.PHN'
wav_file_postfix = '.WAV'
data_type = 'float32'

work_dir = os.getcwd()

paths = sys.argv[1]

# Train 3696 valid 400 test 192
train_path	= np.loadtxt("timit_dataset_list/TRAIN_list.csv", dtype=str)
valid_path	= np.loadtxt("timit_dataset_list/TEST_developmentset_list.csv", dtype=str)
test_path	= np.loadtxt("timit_dataset_list/TEST_coreset_list.csv", dtype=str)
target_path	= os.path.join(paths, sys.argv[2])


# 61 different phonemes
phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phonemes2index = {k:v for v,k in enumerate(phonemes)}


def get_total_duration(file):
	"""Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
	for line in reversed(list(open(file))):
		[_, val, _] = line.split()
		return int(val)


def get_delta(x, N):
    pad_x = np.pad(x, ((0, 0), (N, N)), 'edge')
    delta = np.zeros(np.shape(x))
    iterator = [i + 1 for i in range(N)]
    for t in range(np.shape(x)[1]):
        tmp1, tmp2 = 0, 0
        for n in iterator:
            tmp1 += n * (pad_x[:, (t + N) + n] - pad_x[:, (t + N) - n])
            tmp2 += 2 * n * n
        delta[:, t] = np.divide(tmp1, tmp2)

    return delta


def create_mfcc(filename):
	"""Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
	"""

	rate, sample = 16000, np.fromfile(filename, dtype=np.int16)[512:]
	sample = sample / 32767.5
	mfcc = librosa.feature.mfcc(sample,
								sr=rate,
								n_fft=400,
								hop_length=160,
								n_mfcc=40,
								center=False)
	d_mfcc = get_delta(mfcc, 2)
	a_mfcc = get_delta(d_mfcc, 2)

	out = np.concatenate([mfcc.T, d_mfcc.T, a_mfcc.T], axis=1)

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


def preprocess_dataset(file_list):
	"""Preprocess data, ignoring compressed files and files starting with 'SA'"""
	i = 0
	X = []
	Y = []

	for fname in file_list:
		phn_fname = "{}/{}{}".format(paths, fname, phn_file_postfix)
		wav_fname = "{}/{}{}".format(paths, fname, wav_file_postfix)

		total_duration = get_total_duration(phn_fname)
		fr = open(phn_fname)

		X_val, total_frames = create_mfcc(wav_fname)
		total_frames = int(total_frames)

		X.append(X_val)

		y_val = np.zeros(total_frames) - 1
		start_ind = 0
		for line in fr:
			[start_time, end_time, phoneme] = line.rstrip('\n').split()
			start_time = int(start_time)
			end_time = int(end_time)

			phoneme_num = phonemes2index[phoneme] if phoneme in phonemes2index else -1
			end_ind = int(np.round((end_time) / total_duration * total_frames))
			y_val[start_ind:end_ind] = phoneme_num

			start_ind = end_ind
		fr.close()

		if -1 in y_val:
			print('WARNING: -1 detected in TARGET')
			print(y_val)

		Y.append(y_val.astype('int32'))

		i += 1
		print('file No.', i, end='\r', flush=True)

	print('Done')
	return X, Y


##### PREPROCESSING #####
print()

print('Preprocessing train data...')
X_train, y_train = preprocess_dataset(train_path)
print('Preprocessing valid data...')
X_valid, y_valid = preprocess_dataset(valid_path)
print('Preprocessing test data...')
X_test, y_test = preprocess_dataset(test_path)
print('Preprocessing completed.')

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
