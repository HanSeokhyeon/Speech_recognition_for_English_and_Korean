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
last_paths = paths.split('/')[-1]

# Train 3696 valid 400 test 192
train_path	= np.loadtxt("timit_dataset_list/TRAIN_list.csv", dtype=str)
valid_path	= np.loadtxt("timit_dataset_list/TEST_developmentset_list.csv", dtype=str)
test_path	= np.loadtxt("timit_dataset_list/TEST_coreset_list.csv", dtype=str)
target_path	= os.path.join(paths, sys.argv[2])

spike_frame = 2048 * 6
n_band = 32
n_band_sum = 8
n_time = 8
n_structure = 4

# 61 different phonemes
phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
    "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
    "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phonemes2index = {k:v for v,k in enumerate(phonemes)}

i_max = 0


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


def create_spikegram(filename):
    """Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
    """
    rate, sample = 16000, np.fromfile(filename, dtype=np.int16)[512:]
    sample = sample / 32767.5
    mel = librosa.core.power_to_db(librosa.feature.melspectrogram(sample,
                                                                  sr=rate,
                                                                  n_fft=400,
                                                                  hop_length=160,
                                                                  n_mels=40,
                                                                  center=False))

    filename_spikegram = filename.replace('TIMIT', 'TIMIT_spikegram')
    rate, spikegram = 16000, get_data(filename_spikegram[:-4], sample.shape[0])

    feature = make_feature(y=spikegram,
                           frame=400,
                           hop_length=160)
    feature = np.concatenate((mel, feature), axis=0)
    d_feature = get_delta(feature, 2)
    a_feature = get_delta(d_feature, 2)

    out = np.concatenate([feature.T, d_feature.T, a_feature.T], axis=1)

    return out, out.shape[0]


def get_data(filename, wav_length):
    raw_filename = filename + "_spike.raw"
    num_filename = filename + "_num.raw"

    x = np.fromfile(raw_filename, dtype=np.float64)
    x = np.reshape(x, (-1, n_structure))
    num = np.fromfile(num_filename, dtype=np.int32)

    n_data = np.shape(num)[0]
    acc_num = [sum(num[:i]) for i in range(n_data + 1)]

    for k in range(n_data):
        x[acc_num[k]:acc_num[k + 1], 2] += k * spike_frame

    spikegram = get_spikegram(x=x, num=num, acc_num=acc_num, n_data=n_data)
    spikegram = spikegram[:, :wav_length]

    return spikegram


def get_delay():
    gammatone_filter = np.fromfile("../timit_dataset_list/Gammatone_Filter_Order4.raw", dtype=np.float64)
    gammatone_filter = np.reshape(gammatone_filter, (n_band, -1))
    gammatone_filter = gammatone_filter[:, 1:-1]

    max_point = np.argmax(np.abs(gammatone_filter), axis=1)

    return max_point


max_point = get_delay()


def get_spikegram(x, num, acc_num, n_data):
    # get spikegram_old by SNR
    spikegram = np.zeros((n_band, spike_frame * n_data))
    for k in range(n_data):
        for n in range(num[k]):
            spikegram[int(x[acc_num[k] + n, 0])][int(x[acc_num[k] + n, 2])] \
                += np.abs(x[acc_num[k] + n, 1])

    for idx, point in enumerate(max_point):
        spikegram[idx, point:] = spikegram[idx, :-point]

    return spikegram


def make_feature(y, frame, hop_length):
    feature = []
    feature_tmp = np.zeros(n_band+n_time)
    num_of_frame = int((y.shape[1] - frame) / hop_length + 1)
    start, end = 0, frame

    if y.shape[1] % frame != 0:
        y = np.pad(y, ((0, 0), (0, frame - y.shape[1] % frame)), 'constant', constant_values=0)

    for i in range(num_of_frame):
        feature_tmp[:n_band] = librosa.power_to_db(np.sum(y[:, start:end], axis=1)+1)
        tmp_sum = np.reshape(np.sum(y[:, start:end], axis=0), (n_time, -1))
        feature_tmp[n_band:] = librosa.power_to_db(np.sum(tmp_sum, axis=1)+1)
        start += hop_length
        end += hop_length
        feature.append(np.copy(feature_tmp.reshape(1, -1)))

    feature = np.concatenate(feature, axis=0)
    spectral = feature[:, :n_band]
    # temporal = feature[:, n_band:]
    spectral = spectral.reshape((-1, n_band_sum, n_band//n_band_sum))
    spectral = np.sum(spectral, axis=2)
    # feature = np.concatenate((spectral, temporal), axis=1)

    return spectral.T


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

        X_val, total_frames = create_spikegram(wav_fname)
        total_frames = int(total_frames)

        X.append(X_val)

        y_val = np.zeros(total_frames) - 1
        start_ind = 0
        for j, line in enumerate(fr):
            [start_time, end_time, phoneme] = line.rstrip('\n').split()
            start_time = int(start_time)
            end_time = int(end_time)

            phoneme_num = phonemes2index[phoneme] if phoneme in phonemes2index else -1
            end_ind = int(np.round((end_time) / total_duration * total_frames))
            y_val[start_ind:end_ind] = phoneme_num

            start_ind = end_ind
        fr.close()

        global i_max
        i_max = max(j, i_max)

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
max_length1 = np.shape(max(X_train, key=lambda x: np.shape(x)))[0]
print('Preprocessing valid data...')
X_valid, y_valid = preprocess_dataset(valid_path)
max_length2 = np.shape(max(X_valid, key=lambda x: np.shape(x)))[0]
print('Preprocessing test data...')
X_test, y_test = preprocess_dataset(test_path)
max_length3 = np.shape(max(X_test, key=lambda x: np.shape(x)))[0]
print('Preprocessing completed.')
max_length = max(max_length1, max_length2, max_length3)
print("{} {} {} {}".format(max_length1, max_length2, max_length3, max_length))
print(i_max)

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
