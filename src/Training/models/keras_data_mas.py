import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

import argparse
import math
import h5py
import provider
from pointnet_seg import get_model
from mon_model import model_bertin, rotate_point_cloud, jitter_point_cloud

def log_string(out_str):
    print(out_str)

NUM_CLASSES = 2

ALL_FILES = provider.getDataFiles('h5_data/all_files.txt')
room_filelist = [line.rstrip() for line in open('h5_data/road_file_list.txt')]

# Shuffle train files
train_file_idxs = np.arange(0, len(ALL_FILES))
print('nb de tous les fichiers est : '+str(len(ALL_FILES)))


# Load ALL data
data_batch_list = []
label_batch_list = []
for fn in range(len(ALL_FILES)):
    log_string('----' + str(fn) + '-----')
    data_batch, label_batch = provider.loadDataFile(ALL_FILES[train_file_idxs[fn]])
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)

print('Total :')
print(data_batches.shape)
print(label_batches.shape)

test_area = 'projet_3'

train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
print('\ntrain_data = ', str(train_data.shape))
print('train_label = ', str(train_label.shape))

test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print('test_data = ', str(test_data.shape))
print('test_label = ', str(test_label.shape))

log_string ('\nFichier .h5 loaded....\n')


def train_MAS():
    # epoch number
    epo = 10

    model = model_bertin()
    #model.summary()
    # compile classification model
    model.compile(optimizer='adam',
                  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # train model
    for i in range(epo):
        # rotate and jitter point cloud every epoch
        #train_points_rotate = rotate_point_cloud(train_data)
        #train_points_jitter = jitter_point_cloud(train_points_rotate)
        #print('Rotation et Jitter... OK')
        #model.fit(train_points_jitter, train_label, batch_size=32, epochs=1, shuffle=True, verbose=1)
        model.fit(train_data, train_label, batch_size=32, epochs=1, shuffle=True, verbose=1)

        # evaluate model
        if i % 5 == 0:
            score = model.evaluate(test_data, test_label, verbose=1)
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])

def train_bauman():
    #model = get_model(2)
    model = get_model((1024, 7), NUM_CLASSES)
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

    #TRANSFER LEARNING : loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) metrics=['categorical_accuracy']
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=['accuracy'])
    #loss='sparse_categorical_crossentropy',
    #model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    model.fit(x=train_data, y=train_label, epochs=1, shuffle=True, verbose=1 ) #validation_data=test_data)

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
