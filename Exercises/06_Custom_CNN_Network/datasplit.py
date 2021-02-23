from glob import glob
from shutil import copy
import os
from random import shuffle


dataset_dir = 'dataset'
training_dir = os.path.join(dataset_dir, 'Training')
train_neg_dir = os.path.join(training_dir, '0')
train_pos_dir = os.path.join(training_dir, '1')
testing_dir = os.path.join(dataset_dir, 'Testing')
test_neg_dir = os.path.join(testing_dir, '0')
test_pos_dir = os.path.join(testing_dir, '1')


if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(testing_dir):
    os.mkdir(testing_dir)
if not os.path.isdir(train_neg_dir):
    os.mkdir(train_neg_dir)
if not os.path.isdir(train_pos_dir):
    os.mkdir(train_pos_dir)
if not os.path.isdir(test_neg_dir):
    os.mkdir(test_neg_dir)
if not os.path.isdir(test_pos_dir):
    os.mkdir(test_pos_dir)

pos_sample_dir = 'classes/0'
neg_sample_dir = 'classes/1'


VAL_SPLIT = 0.2
pos_samples = glob(pos_sample_dir + '/*.jpg')
neg_samples = glob(neg_sample_dir + '/*.jpg')
shuffle(pos_samples)
shuffle(neg_samples)

train_pos_size = int(len(pos_samples) * (1 - VAL_SPLIT / 2))
train_neg_size = int(len(neg_samples) * (1 - VAL_SPLIT / 2))

for i, file in enumerate(pos_samples):
    if i < train_pos_size:
        dest = train_pos_dir
    else:
        dest = test_pos_dir
    copy(file, dest)

for i, file in enumerate(neg_samples):
    if i < train_neg_size:
        dest = train_neg_dir
    else:
        dest = test_neg_dir
    copy(file, dest)
