from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import Input, Lambda, concatenate
import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from tf_util import dense_bn, conv1d_bn
import pointnet_base

def get_model(input_shape, classes):
#def get_model(classes):
    """
    PointNet model for segmentation
    :param input_shape: shape of the input point clouds (NxK)
    :param classes: number of classes in the segmentation problem
    :param activation: activation of the last layer
    :return: Keras model of the classification network
    """

    assert K.image_data_format() == 'channels_last'

    inputs = Input(input_shape, name='Input_cloud')
    #inputs = Input(shape=(4096, 7), dtype=tf.float32, name='Input_cloud')
    net, local_features = pointnet_base.get_model(inputs)
    print('--> base du modele depuis CLASSIF.....')
    print('RES NET for input = ', net.shape)
    print('    LOCAL Feature = ', local_features.shape)

    global_feature_expanded = Lambda(K.expand_dims, arguments={'axis': 1})(net)
    print('\n    Global feature = ', global_feature_expanded.shape)
    global_feature_tiled = Lambda(K.tile, arguments={'n': [1, K.shape(local_features)[1], 1]})(global_feature_expanded)
    print('    Global feature tiled with LOCAL Feature = ', global_feature_tiled.shape)

    net = Lambda(concatenate)([local_features, global_feature_tiled])
    print('_NET Concat local by global = ', net.shape)

    net = conv1d_bn(net, num_filters=512, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv1')
    print('_SEG_conv1 filtered by 512 = ', net.shape)

    net = conv1d_bn(net, num_filters=256, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv2')
    print('_SEG_conv2 filtered by 256 = ', net.shape)

    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='seg_conv3')
    print('_SEG_conv3 filtered by 128 = ', net.shape)

    point_features = net
    net = conv1d_bn(point_features, num_filters=128, kernel_size=1, padding='valid',
                    scope='seg_conv4', activation='softmax')
    print('_SEG_conv4 by 128 = ', net.shape)

    net = conv1d_bn(net, num_filters=classes, kernel_size=1, padding='valid',
                    scope='seg_conv5',activation='softmax')
    #    num_filters=classes ,num_filters=num_filters=len(classes)
    print('_LAST NET by Class = ', net.shape)

    model = Model(inputs=inputs, outputs=net, name='pointnet_seg')

    return model

#model = get_model(2)
NUM_CLASSES = 2
#model = get_model((None, 7), NUM_CLASSES)
#model.summary()