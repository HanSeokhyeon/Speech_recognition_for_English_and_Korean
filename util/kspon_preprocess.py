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
random.shuffle(directories)
dir_num = len(directories)
train_num = int(dir_num * test_split)
test_num = dir_num - train_num
valid_num = int(train_num * val_split)
train_num = train_num - valid_num
train_source_path = directories[:train_num]
valid_source_path = directories[train_num:train_num+valid_num]
test_source_path = directories[train_num+valid_num:]
pass