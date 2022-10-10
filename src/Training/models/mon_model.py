import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Lambda, concatenate

#tf.config.experimental_run_functions_eagerly(True)
import h5py

#@tf.function
def mat_mul(A, B):
    return tf.matmul(A, B)


def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 7)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


'''
global variable
'''
# number of points in each sample
num_points = 4096 #1024
# number of categories
k = 2

# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

'''
Pointnet Architecture
'''
def model_bertin() :
    # input_Transformation_net
    input_points = Input(shape=(num_points, 7))
    print('\nMy input = ', input_points.shape)

    print('----------------input_Transformation_net---------------------')
    x = Convolution1D(64, 1, activation='relu', input_shape=(num_points, 7))(input_points)
    print('++ Conved_1 (64) = ', x.shape)

    x = BatchNormalization()(x)
    print('Batched_1 = ', x.shape)

    x = Convolution1D(128, 1, activation='relu')(x)
    print('++ Conved_2 (128) = ', x.shape)

    x = BatchNormalization()(x)
    print('Batched_2 = ', x.shape)

    x = Convolution1D(1024, 1, activation='relu')(x)
    print('++ Conved_3 (1024) = ', x.shape)

    x = BatchNormalization()(x)
    print('Batched_3 = ', x.shape)

    x = MaxPooling1D(pool_size=num_points)(x)
    print('----MAX pooled = ', x.shape)

    x = Dense(512, activation='relu')(x)
    print('     Densed_1 (512) = ', x.shape)

    x = BatchNormalization()(x)
    print('Batched_4 = ', x.shape)

    x = Dense(256, activation='relu')(x)
    print('     Densed_2 (256) = ', x.shape)

    x = BatchNormalization()(x)
    print('Batched_5 = ', x.shape)

    #x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    #print('     Densed_3 = ', x.shape)
    #input_T = Reshape((3, 3))(x)

    x = Dense(7 * 7, weights=[np.zeros([256, 7 * 7]), np.eye(7).flatten().astype(np.float32)])(x)
    print('     Densed_3 7*7 = ', x.shape)
    input_T = Reshape((7, 7))(x)

    print('___Reshaped = ', input_T.shape)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    print('.....TRANSFORM_1 = ', g.shape)
    print('\n--------------------Forward net-------------------------------')

    g = Convolution1D(64, 1, input_shape=(num_points, 7), activation='relu')(g)
    print('++ Conved_4 (64) = ', g.shape)

    g = BatchNormalization()(g)
    print('Batched_6 = ', g.shape)

    g = Convolution1D(64, 1, input_shape=(num_points, 7), activation='relu')(g)
    print('++ Conved_5 (64) = ', g.shape)

    g = BatchNormalization()(g)
    print('Batched_7 = ', g.shape)

    # feature transformation net
    print('\n--------------------feature transformation-------------------------------')
    f = Convolution1D(64, 1, activation='relu')(g)
    print('++ Conved_6 (64) = ', f.shape)

    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    print('++ Conved_7 (128) = ', f.shape)

    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    print('++ Conved_8 (1024) = ', f.shape)

    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    print('----MAX pooled = ', f.shape)

    f = Dense(512, activation='relu')(f)
    print('     Densed_4 512 = ', f.shape)

    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    print('     Densed_5 256 = ', f.shape)

    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    print('     Densed_5 64*64 = ', f.shape)

    feature_T = Reshape((64, 64))(f)
    print('___Reshaped = ', feature_T.shape)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    print('.....TRANSFORM_2 = ', g.shape)
    print('\n--------------------Forward net-------------------------------')

    seg_part1 = g
    print('save : \n\t\t', seg_part1.shape, ' as a Specific Feature... ')

    g = Convolution1D(64, 1, activation='relu')(g)
    print('++ Conved_9 (64) = ', g.shape)

    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    print('++ Conved_10 (128) = ', g.shape)

    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    print('++ Conved_11 (1024) = ', g.shape)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)
    print('----MAX pooled = ', global_feature.shape)

    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    print('.....TRANSFORM_3 for Global and LAST Feature = ', global_feature.shape)
    print('\n--------------------POINT NET SEGMENTATION-------------------------------')

    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    print('\t\t Fusion de Local Feature avec Global Feature = ', c.shape)
    c = Convolution1D(512, 1, activation='relu')(c)
    print('++ Conved_12 (512) = ', c.shape)

    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    print('++ Conved_13 (256) = ', c.shape)

    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    print('++ Conved_14 (128) = ', c.shape)

    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    print('++ Conved_15 (128) = ', c.shape)

    c = BatchNormalization()(c)
    prediction = Convolution1D(k, 1, activation='softmax')(c)
    print('++ Conved_16 Pour prediction (CLASS) = ', prediction.shape)

    print('--------------------SEGMENTATION RESEAU DE NEURONE FINI-------------------------------')
    print('input model = ', input_points)
    print('output model = ', prediction)
    # define model
    model = Model(inputs=input_points, outputs=prediction)

    model.summary()
    return model

model_bertin()