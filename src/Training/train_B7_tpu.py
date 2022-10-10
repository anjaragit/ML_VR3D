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
from pytictoc import TicToc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'logs'))
from provider2 import PointCloudProvider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet_seg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='logs/', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
FLAGS = parser.parse_args()

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')


MAX_EPOCH = 50
NUM_POINT = 4096  # help='Point Number [256/512/1024/2048/4096] [default: 1024]
MAX_NUM_POINT = 4096
BATCH_SIZE = 32
NUM_CLASSES = 2 #13 #40

BASE_LEARNING_RATE = 0.001
DECAY_STEP = 2000
DECAY_RATE = 0.9


def get_learning_rate_schedule():
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,  # Initial learning rate
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    return learning_rate

def train():
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
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(generator_training, validation_data=generator_validation,
    steps_per_epoch=len(generator_training),
    validation_steps=len(generator_validation),
    epochs=MAX_EPOCH, use_multiprocessing=True)

  print('\nsave model...')
  model.save_weights("/kaggle/output/trained_ep19_gpu_Kgl_modelB7_30_06_param2.h5")
  print('\nFinished...')

def train_strateg():
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        # Select appropriate distribution strategy
        if tpu:
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)  #, steps_per_run=128)
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
            with strategy.scope():
              train()
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy that works on CPU and single GPU
            print('Running on CPU instead')
            with strategy.scope():
              train()
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    except ValueError:  # If TPU not found
        tpu = None
        print('there is no TPU device')


if __name__ == "__main__":
  t = TicToc()
  t.tic()
  train_strateg()
  t.toc()