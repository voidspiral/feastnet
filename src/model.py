from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
import h5py

from src.my_batch_norm import bn_layer_top
from src.train import random_seed


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
    return tf.Variable(initial)


def assignment_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
    return tf.Variable(initial)


def tile_repeat(n, repTime):
    '''
    create something like 111..122..2333..33 ..... n..nn
    one particular number appears repTime consecutively.
    This is for flattening the indices.
    '''
    idx = tf.range(n)
    idx = tf.reshape(idx, [-1, 1])  # Convert to a n x 1 matrix.
    idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime
    y = tf.reshape(idx, [-1])
    return y




def get_weight_assigments_translation_invariance(x, adj, u, c):
    batch_size, num_points, in_channels = x.get_shape().as_list()
    batch_size, num_points, K = adj.get_shape().as_list()
    M, in_channels = u.get_shape().as_list()
    # [batch, N, K, ch]
    patches = get_patches(x, adj)
    # [batch, N, ch, 1]
    x = tf.reshape(x, [-1, num_points, in_channels, 1])
    # [batch, N, ch, K]
    patches = tf.transpose(patches, [0, 1, 3, 2])
    # [batch, N, ch, K]
    patches = tf.subtract(x, patches)
    # [batch, ch, N, K]
    patches = tf.transpose(patches, [0, 2, 1, 3])
    # [batch, ch, N*K]
    x_patches = tf.reshape(patches, [-1, in_channels, num_points * K])
    # batch, M, N*K
    patches = tf.map_fn(lambda x: tf.matmul(u, x), x_patches)
    # batch, M, N, K
    patches = tf.reshape(patches, [-1, M, num_points, K])
    # [batch, K, N, M]
    patches = tf.transpose(patches, [0, 3, 2, 1])
    # [batch, K, N, M]
    patches = tf.add(patches, c)
    # batch, N, K, M
    patches = tf.transpose(patches, [0, 2, 1, 3])
    patches = tf.nn.softmax(patches)
    return patches

def get_patches_1(x, adj):
    '''
    获得 x 的adj patch
    :param x:  N, C
    :param adj:  N, K
    :return: N,K,C
    '''
    num_points, in_channels = x.get_shape().as_list()
    zeros = tf.zeros([1, in_channels], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    x = tf.concat([zeros, x], 0)  # [N+1, C]
    patches = tf.gather(x, adj)  # [N,K,C]
    return patches

def get_slices(x, adj):
    '''
    
    :param x: 需要 pad
    :param adj:需要 pad
    :return:
    '''
    batch_size, num_points, in_channels = x.get_shape().as_list()
    batch_size, input_size, K = adj.get_shape().as_list()
    zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
    x = tf.concat([zeros, x], 1)
    x = tf.reshape(x, [batch_size * (num_points + 1), in_channels])
    adj = tf.reshape(adj, [batch_size * num_points * K])
    adj_flat = tile_repeat(batch_size, num_points * K)
    adj_flat = adj_flat * (num_points + 1)
    adj_flat = adj_flat + adj
    adj_flat = tf.reshape(adj_flat, [batch_size * num_points, K])
    slices = tf.gather(x, adj_flat)
    slices = tf.reshape(slices, [batch_size, num_points, K, in_channels])
    return slices


def get_patches(x, adj):
    batch_size, num_points, in_channels = x.get_shape().as_list()
    batch_size, num_points, K = adj.get_shape().as_list()
    patches = get_slices(x, adj)
    return patches


def custom_conv2d(x, adj, out_channels,x_size, M ,need_BN,is_training,scope):
    print("Translation-invariant\n")
    batch_size, input_size, in_channels = x.get_shape().as_list()
    W = weight_variable([M, out_channels, in_channels])
    if need_BN:
        b = bias_variable([out_channels])
    u = assignment_variable([M, in_channels])
    c = assignment_variable([M])
    batch_size, input_size, K = adj.get_shape().as_list()
    # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
    adj_size = tf.count_nonzero(adj, 2)
    # deal with unconnected points: replace NaN with 0
    non_zeros = tf.not_equal(adj_size, 0)
    adj_size = tf.cast(adj_size, tf.float32)
    adj_size = tf.where(non_zeros, tf.reciprocal(adj_size), tf.zeros_like(adj_size))
    # [batch_size, input_size, 1, 1]
    adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
    # [batch_size, input_size, K, M]
    q = get_weight_assigments_translation_invariance(x, adj, u, c)
    # [batch_size, in_channels, input_size]
    x = tf.transpose(x, [0, 2, 1])
    W = tf.reshape(W, [M * out_channels, in_channels])
    # Multiple w and x -> [batch_size, M*out_channels, input_size]
    wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
    # Reshape and transpose wx into [batch_size, input_size, M*out_channels]
    wx = tf.transpose(wx, [0, 2, 1])
    # Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
    patches = get_patches(wx, adj)
    # [batch_size, input_size, K, M]
    # q = get_weight_assigments_translation_invariance(x, adj, u, c)
    # Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
    patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
    # [out, batch_size, input_size, K, M]
    patches = tf.transpose(patches, [4, 0, 1, 2, 3])
    patches = tf.multiply(q, patches)
    patches = tf.transpose(patches, [1, 2, 3, 4, 0])
    # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
    patches = tf.reduce_sum(patches, axis=2)
    patches = tf.multiply(adj_size, patches)
    # Add add elements for all m
    patches = tf.reduce_sum(patches, axis=2)
    if need_BN:
        # [batch_size, input_size, out]
        patches = bn_layer_top(patches,out_channels,x_size,is_training)
    else:
        patches = patches + b
    return patches


def custom_lin(input, out_channels):
    batch_size, input_size, in_channels = input.get_shape().as_list()
    W = weight_variable([in_channels, out_channels])
    b = bias_variable([out_channels])
    return tf.map_fn(lambda x: tf.matmul(x, W), input) + b


def custom_max_pool(input, kernel_size, stride=[2, 2], padding='VALID'):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(input, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1],
                             padding=padding)
    return outputs


def get_model_fill(x, adj,x_size,is_training):
    """
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
    adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])

    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    # batch_size, input_size, out_channels
    # x = tf.nn.relu(custom_lin(x, 16, scope='lin1'))
    h_dims = [16, 16, 32, 64, 64, 128]
    
    for i,dim in enumerate(h_dims):
        # [batch_size, input_size, out]
        x = tf.nn.relu(custom_conv2d(x, adj, dim,x_size=x_size, M=9,
                                     need_BN=True,is_training=is_training, scope='conv%d'%i))
    y = tf.nn.relu(custom_conv2d(x, adj, 3, x_size=x_size, M=9,
                                 need_BN=False, is_training=is_training, scope='output'))
    # [batch_size, input_size, 3]
    return y

