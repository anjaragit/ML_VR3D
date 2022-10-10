import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from matplotlib import pyplot as plt

import argparse
import math
import h5py
import provider
from pointnet_seg import get_model

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99

ALL_FILES = provider.getDataFiles('indoor3d_sem_seg/all_files.txt')
room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg/room_filelist.txt')]

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

test_area = 'Area_2'
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)



def train_bauman():
    #model = get_model(2)
    model = get_model((None, 9), NUM_CLASSES)
    model.summary()

    #learning_rate = get_learning_rate_schedule()
    #optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5)

    #PointCloudProvider.initialize_dataset()
    #generator_training = PointCloudProvider('train', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)
    #generator_validation = PointCloudProvider('test', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)

    #callbacks = [
    #    tf.keras.callbacks.ModelCheckpoint(CKPT_DIR, save_weights_only=False, save_best_only=True),
    #    tf.keras.callbacks.TensorBoard(LOG_DIR)
    #]

    #TRANSFER LEARNING : loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=['accuracy'])
    #loss='sparse_categorical_crossentropy',
    #model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    model.fit(x=train_data, y=train_label, epochs=10) #validation_data=test_data)

    '''model.fit_generator(
        generator=generator_training,
        validation_data=generator_validation,
        steps_per_epoch=len(generator_training),
        validation_steps=len(generator_validation),
        epochs=MAX_EPOCH,
        callbacks=callbacks,
        use_multiprocessing=False
    )'''

    model.save("trained_model.pb")
    print('\ntrain ok... \nmodel saved...')

print('\nLet train it now...')
#train_MAS()
train_bauman()
