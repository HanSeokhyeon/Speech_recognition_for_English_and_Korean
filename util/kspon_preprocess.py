import os
import sys
import random

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

paths = sys.argv[1]
directories = os.listdir(paths)
# random.shuffle(directories)
# dir_num = len(directories)
# train_num = int(dir_num * test_split)
# test_num = dir_num - train_num
# valid_num = int(train_num * val_split)
# train_num = train_num - valid_num
# train_source_path = directories[:train_num]
# valid_source_path = directories[train_num:train_num+valid_num]
# test_source_path = directories[train_num+valid_num:]

directories.sort()
if '.DS_Store' in directories:
    os.remove("{}/{}".format(paths, '.DS_Store'))
    directories.remove('.DS_Store')
train_file_num = 3696
valid_file_num = 400
test_file_num = 192
train_source_path = os.listdir("{}/{}".format(paths, directories[0])) \
                    + os.listdir("{}/{}".format(paths, directories[1])) \
                    + os.listdir("{}/{}".format(paths, directories[2])) \
                    + os.listdir("{}/{}".format(paths, directories[3]))
train_source_path = sorted(set([fname[:-4] for fname in train_source_path]))
train_source_path = train_source_path[:train_file_num]
valid_test_source_path = os.listdir("{}/{}".format(paths, directories[4]))
valid_test_source_path = sorted(set([fname[:-4] for fname in valid_test_source_path]))
valid_source_path = valid_test_source_path[:valid_file_num]
test_source_path = valid_test_source_path[valid_file_num:valid_file_num+test_file_num]


pass