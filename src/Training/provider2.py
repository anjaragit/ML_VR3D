import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data/")



class PointCloudProvider(Sequence):
    """
    Lazily load point clouds and annotations from filesystem and prepare it for model training.
    """

    def __init__(self, mode='train', batch_size=32, n_classes=2, sample_size=4096, task="seg"): #task="classification"):
        """
        Instantiate a data provider instance for point cloud data.
        Args:
            dataset: pandas DataFrame containing
            n_classes: The number of different cthe index to the files (train or test set)
            batch_size: the desired batch sizelasses (needed for one-hot encoding of labels)
            sample_size: the amount of points to sample per instance.
            task: string denoting the tasks for which the data is to be loaded. Either "classification" (default) or "segmentaion".
        """
        self.datasets = {
            'train': 'train_VDS_7.h5',
            'test': 'test_VDS_7.h5'
        }
        self.mode = mode
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.sample_size = sample_size
        self.task = task

        self.indices = np.arange(0,len(h5py.File(self.datasets[self.mode], 'r')['data']), 1)
        print( 'Indice = ', self.indices )

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil((len(self.indices)/self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        #print('Generated batch_indices = ', batch_indices.shape)
        mask = np.zeros_like(self.indices).astype(np.bool)
        mask[batch_indices] = True
        X = h5py.File(self.datasets[self.mode], 'r')['data'][mask, ...]
        #print('X',X.shape)
        y = h5py.File(self.datasets[self.mode], 'r')['label'][mask, ...]
        #print('y', y.shape, '\n')

        #return np.array(X), to_categorical(np.array(y), num_classes=self.n_classes)
        #self.rotate_point_clouds(np.array(X))
        return np.array(X), np.array(y)

    def sample_random_points(self, pc):
        r_idx = np.random.randint(pc.shape[1], size=self.sample_size)
        return np.take(pc, r_idx, axis=1)

    def on_epoch_end(self):
        """Shuffle training data, so batches are in different order"""
        np.random.shuffle(self.indices)

    def rotate_point_clouds(self, batch, rotation_angle_range=(-np.pi / 8, np.pi / 8)):
        """Rotate point cloud around y-axis (=up) by random angle"""
        for b in range(batch.shape[0]):
            phi = np.random.uniform(*rotation_angle_range)
            c, s = np.cos(phi), np.sin(phi)
            R = np.asarray([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]])
            shape_pc = batch[b, ...]
            batch[b, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return batch

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def load_data_file(self, filename):
        if self.task == "classification":
            return self.load_h5(filename)
        else:
            return self.load_h5_data_label_seg(filename)

    def load_h5_data_label_seg(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:]
        label = f['label'][:]
        #seg = f['pid'][:]
        return data, label  #, seg

    @staticmethod
    def initialize_dataset():
        """
        Loads an index to all files and structures them.
        :param data_directory: directory containing the data files
        :param file_extension: extension of the data files
        :return: pandas dataframe containing an index to all files and a label index,
            mapping numerical label representations to label names.
        """

        print("[Provider]: Creating Virtual Dataset")

        train_index = os.path.join(BASE_DIR, 'data/h5_data/train_files.txt')
        test_index = os.path.join(BASE_DIR, 'data/h5_data/test_files.txt')

        train_files = [line.rstrip() for line in open(train_index)]
        print('train_files = ', train_files)
        test_files = [line.rstrip() for line in open(test_index)]
        print('test_files = ', test_files)

        def create_vds(files, prefix='train'):

            print('\nVDS creation...')
            out_size = 0
            for f in files:
                out_size += h5py.File(DATA_DIR+f, 'r')['data'].shape[0]

            print('Total h5 size : ', out_size) #(23585, 4096, 9)

            # Assemble virtual dataset
            point_layout = h5py.VirtualLayout(shape=(out_size, 4096, 7), dtype='uint8')
            label_layout = h5py.VirtualLayout(shape=(out_size, 4096, 1), dtype='uint8')
            print('\nPoint VirtualLayout = ', point_layout.shape)
            print('Label VirtualLayout = ', label_layout.shape)

            for i, f in enumerate(files):
                h5_path = DATA_DIR+f
                size = len(h5py.File(h5_path, 'r')['data'])
                start_idx = i*1000  # only the last chunk is smaller than 2048
                end_idx = start_idx + size

                print('\nput in VirtualSource...')
                vsource_points = h5py.VirtualSource(h5_path, 'data', shape=(size, 4096, 7), maxshape=(out_size, 4096, 7))
                vsource_label = h5py.VirtualSource(h5_path, 'label', shape=(size, 4096, 1), maxshape=(out_size, 4096, 1))
                print('vsource_points = ',vsource_points.shape,
                      'vsource_label = ',vsource_label.shape)

                point_layout[start_idx:end_idx, ...] = vsource_points
                label_layout[start_idx:end_idx, ...] = vsource_label
                print('point_layout = ', point_layout.shape,
                      'label_layout = ', label_layout.shape)
                print('In : layout[', start_idx,':',end_idx,'...]')

            # Add virtual dataset to output file$

            with h5py.File("{}_VDS_7.h5".format(prefix), 'w', libver='latest') as f:
                print('\n\tcreate_virtual_dataset for DATA...')
                f.create_virtual_dataset('data', point_layout)
                print('\n\t', point_layout.shape)

                print('\n\tcreate_virtual_dataset for LABEL...')
                f.create_virtual_dataset('label', label_layout)
                print('\n\t', label_layout.shape)

        create_vds(train_files, 'train')
        create_vds(test_files, 'test')

        print("[Provider] Created Virtual Dataset.")


#init = PointCloudProvider()
#init.initialize_dataset()
