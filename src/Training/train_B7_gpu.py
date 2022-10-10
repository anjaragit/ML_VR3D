import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import socket
import importlib
import os
import sys
import datetime
import time 

#from pytictoc import TicToc

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/kaggle/input/vr3d-etech'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'logs'))
from provider2 import PointCloudProvider
import tf_util

"""parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet_seg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='logs/fit/', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
FLAGS = parser.parse_args()
"""


MODEL = importlib.import_module('pointnet_seg') # import network module
MODEL_FILE = '/kaggle/input/vr3d-etech/models/pointnet_seg.py'
#MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')



MAX_EPOCH = 1
NUM_POINT = 4096  # help='Point Number [256/512/1024/2048/4096] [default: 1024]
MAX_NUM_POINT = 4096
BATCH_SIZE = 32
NUM_CLASSES = 2 #13 #40

BASE_LEARNING_RATE = 0.001
DECAY_STEP = 2000
DECAY_RATE = 0.1


def get_learning_rate_schedule():
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,  # Initial learning rate
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    return learning_rate


def train():
    with tf.device("gpu:0"):
        model = MODEL.get_model((None, 7), NUM_CLASSES)
        model.summary()

        learning_rate = get_learning_rate_schedule()
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # initialize Dataset
        PointCloudProvider.initialize_dataset()

        print('\nput in generator...')
        generator_training = PointCloudProvider('train', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)
        generator_validation = PointCloudProvider('test', BATCH_SIZE, n_classes=NUM_CLASSES, sample_size=MAX_NUM_POINT)
        print('\ntraining...')
        print(len(generator_training))
    
    
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(generator_training, validation_data=generator_validation,
                        steps_per_epoch=20,#len(generator_training)
                        validation_steps=5,#len(generator_validation)
                        epochs=MAX_EPOCH,use_multiprocessing=False) #100 taille de donne a traite
    
        print('\nsave model...')
        model.save_weights("/kaggle/working/trained_ep_save2.h5")
        #model.save("/kaggle/working/trained_ep_save.h5")#error multiprocessing

        print('\nFinished...')
        #model.load_weights("../input/weights-output/trained_ep_save1.h5")
        #pred_model = model.predict(generator_validation)
        #print('pred',pred_model)
if __name__ == "__main__":
    train()