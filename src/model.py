from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
import h5py

# from src.train import  COARSEN_LEVEL

random_seed=0
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


def get_weight_assigments(x, adj, u, v, c):
    batch_size, in_channels, num_points = x.get_shape().as_list()
    batch_size, num_points, K = adj.get_shape().as_list()
    M, in_channels = u.get_shape().as_list()
    # [batch_size, M, N]
    ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
    vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
    # [batch_size, N, M]
    vx = tf.transpose(vx, [0, 2, 1])
    # [batch_size, N, K, M]
    patches = get_patches(vx, adj)
    # [K, batch_size, M, N]
    patches = tf.transpose(patches, [2, 0, 3, 1])
    # [K, batch_size, M, N]
    patches = tf.add(ux, patches)
    # [K, batch_size, N, M]
    patches = tf.transpose(patches, [0, 1, 3, 2])
    patches = tf.add(patches, c)
    # [batch_size, N, K, M]
    patches = tf.transpose(patches, [1, 2, 0, 3])
    patches = tf.nn.softmax(patches)
    return patches


def get_weight_assigments_translation_invariance(x, adj, u, c):
    # u.shape=[M, in_channels]
    
    K = adj.get_shape()[1]
    M, in_channels = u.get_shape().as_list()
    # [N, K, ch]
    patches = get_patches(x, adj)
    # [ N, ch, 1]
    x = tf.reshape(x, [-1, in_channels, 1])
    # [ N, ch, K]
    patches = tf.transpose(patches, [0, 2,1])
    # [N, ch, K]
    patches = tf.subtract(x, patches)  # invariance
    # [ch, N, K]
    patches = tf.transpose(patches, [1,0,2])
    # [ch, N*K]
    x_patches = tf.reshape(patches, [in_channels, -1])
    #  M, N*K  = [M, ch]  x  [ch, N*K]
    patches = tf.matmul(u, x_patches)
    # M, N, K
    patches = tf.reshape(patches, [M, -1, K])
    # [K, N, M]
    patches = tf.transpose(patches, [2, 1,0])
    # [K, N, M]
    patches = tf.add(patches, c)
    # N, K, M
    patches = tf.transpose(patches, [1,0,2])
    patches = tf.nn.softmax(patches)
    return patches


def get_patches(x, adj):
    '''
    获得 x 的adj patch
    :param x:  N, C
    :param adj:  N, K
    :return: N,K,C
    '''
    num_points, in_channels = x.get_shape().as_list()
    zeros = tf.zeros([1, in_channels], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    x = tf.concat([zeros, x], 0) #[N+1, C]
    patches = tf.gather(x, adj) #[N,K,C]
    return patches


def get_patches_2(x, adj):
    '''
    获得 x 的adj patch
    :param x: num_points, in_channels
    :param adj: num_points, K
    :return:
    '''
    num_points, in_channels = x.get_shape().as_list()
    input_size, K = adj.get_shape().as_list()
    zeros = tf.zeros([ 1, in_channels], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    x = tf.concat([zeros, x], 0) #[num_points+1, in_channels]
    
    ############### 2-ring
    zeros_adj = tf.zeros([ 1, K], dtype=tf.int32)
    _adj = tf.concat([zeros_adj, adj], 0)  # [num_points+1 , K]
    
    patches_adj = tf.gather(_adj, adj)  # [num_points,K,K]
    K2=K*K
    patches_adj = tf.reshape(patches_adj, [num_points, K2])
    
    def cut_nodes(x):
        y, idx = tf.unique(x)
        paddings = tf.constant([0, K2 - tf.shape(y)])
        y = tf.pad(y, paddings)
        return y
    
    # [num_points,K2] adj_2 need to save in numpy
    adj_2 = tf.map_fn(lambda x: cut_nodes(x), patches_adj)
    patches = tf.gather(x, adj_2)  # [num_points,K2,in_channels]

    return patches


def custom_conv2d(x, adj, out_channels, M, translation_invariance=True,scope=''):
    if translation_invariance == True:
        with tf.variable_scope(scope):
            print("Translation-invariant\n")
            in_channels = x.get_shape().as_list()[1]
            W = weight_variable([M, out_channels, in_channels])  # 卷积核参数
            b = bias_variable([out_channels])
            u = assignment_variable([M, in_channels])
            c = assignment_variable([M])
            K = adj.get_shape()[1]
            # Calculate neighbourhood size for each input - [N, neighbours]
            adj_size = tf.count_nonzero(adj, 1)  # [N] 每个元素 是该点的邻接点数量
            # deal with unconnected points: replace NaN with 0
            non_zeros = tf.not_equal(adj_size, 0)  # [N] bool  是否有孤立点
            adj_size = tf.cast(adj_size, tf.float32)
            adj_size = tf.where(non_zeros, tf.reciprocal(adj_size), tf.zeros_like(adj_size))  # 非孤立点 删选出来
            # [N, 1, 1]
            adj_size = tf.reshape(adj_size, [-1, 1, 1])
            # [N, K, M] 当K index 到 0 时， M 维相等
            q = get_weight_assigments_translation_invariance(x, adj, u, c)
            # [C, N]
            x = tf.transpose(x, [1,0])
            # [M*O,C]
            W = tf.reshape(W, [M * out_channels, in_channels])  # 卷积核参数
            # [M*O, N]
            wx = tf.matmul(W, x)  # 卷积
            # [N, M*O]
            wx = tf.transpose(wx, [1,0])
            # adj中k为0的索引取到的 wx 也为0
            # [N, K, M*O]
            patches = get_patches(wx, adj)
            # [ N, K, M, O]
            patches = tf.reshape(patches, [-1, K, M, out_channels])
            # [O, N, K, M]
            patches = tf.transpose(patches, [3, 0, 1, 2])
            # [N, K, M]*[O, N, K, M]=[O, N, K, M] element-wise
            patches = tf.multiply(q, patches)
            # [N, K, M, O]
            patches = tf.transpose(patches, [1, 2, 3, 0])
            # Add all the elements for all neighbours for a particular m
            # sum data in index K is zero, so need 归一化
            patches = tf.reduce_sum(patches, axis=1)  # [ N, M, O]
            # [N, 1, 1]*[ N, M, O]=[ N, M, O]
            patches = tf.multiply(adj_size, patches)  # /Ni
            # Add add elements for all m， 因为不是 reduce_mean 所以支持M中的0
            # [ N, O]
            patches = tf.reduce_sum(patches, axis=1)
            # [N, O]
            patches = patches + b

            return patches

    else:
        batch_size, input_size, in_channels = x.get_shape().as_list()
        W = weight_variable([M, out_channels, in_channels])
        b = bias_variable([out_channels])
        u = assignment_variable([M, in_channels])
        v = assignment_variable([M, in_channels])
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
        q = get_weight_assigments(x, adj, u, v, c)
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
        # [batch_size, input_size, out]
        patches = patches + b
        return patches



def custom_lin(input, out_channels,scope):
    # 可以理解为升降维 1x1 Conv, 只对input最后一维进行 全连接
    with tf.variable_scope(scope):

        input_size, in_channels = input.get_shape().as_list()
        W = weight_variable([in_channels, out_channels])
        b = bias_variable([out_channels])
        return tf.matmul(input, W)+b


def custom_max_pool(input, kernel_size, stride=[2, 2], padding='VALID'):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(input, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1],
                             padding=padding)
    return outputs


def perm_data(input, indices):
    """
    排列成待卷积的序列
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    new x can be used for pooling
    """
    
    M, channel = input.shape
    Mnew = tf.shape(indices)[0]
    
    xnew = tf.zeros((Mnew, channel))
    
    def loop_body(i):
        if i < M:
            xnew[i] = input[indices[i]]
    
    xnew = tf.while_loop(lambda i, *args: i < Mnew, loop_body, [xnew])
    return xnew

    
def get_model_fill(x, adj):
    """
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
    adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])
    
    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    # batch_size, input_size, out_channels
    x = tf.nn.relu(custom_lin(x,  16,scope='lin1'))
    h_dims=[32,32,64,64,128,128]
    for dim in h_dims:
        x=tf.nn.relu(custom_conv2d(x,  adj, dim, 9,scope='conv1'))
    y_conv = custom_conv2d(x, adj, 3, 9,scope='conv4')
    return y_conv



def get_model_original(x, adj, num_classes):
    """
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
    adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])

    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    out_channels_fc0 = 16
    # batch_size, input_size, out_channels
    h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
    # Conv1
    M_conv1 = 9
    out_channels_conv1 = 32
    # [batch_size, input_size, out]
    h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
    # Conv2
    M_conv2 = 9
    out_channels_conv2 = 64
    h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
    # Conv3
    M_conv3 = 9
    out_channels_conv3 = 128
    h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
    # Lin(1024)
    out_channels_fc1 = 1024
    h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
    # Lin(num_classes)
    y_conv = custom_lin(h_fc1, num_classes)
    return y_conv

COARSEN_LEVEL=2


def get_model(x, adj, perms, num_classes, architecture):
    """
    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    out_channels_fc0 = 16
    fc0 = tf.nn.relu(custom_lin(x[0], out_channels_fc0))
    # Conv1
    M_conv1 = 9
    out_channels_conv1 = 32
    conv1 = tf.nn.relu(custom_conv2d(fc0, adj[0], out_channels_conv1, M_conv1))
    # Conv2
    M_conv2 = 9
    out_channels_conv2 = 64
    # [batch_size, input_size, out]
    conv2 = tf.nn.relu(custom_conv2d(conv1, adj[0], out_channels_conv2, M_conv2))
    conv2 = perm_data(conv2, perms[0])
    
    # Conv3
    M_conv3 = 9
    out_channels_conv3 = 128
    conv3 = tf.nn.relu(custom_conv2d(conv2, adj[1], out_channels_conv3, M_conv3))
    # Lin(1024)
    out_channels_fc1 = 1024
    fc1 = tf.nn.relu(custom_lin(conv3, out_channels_fc1))
    # Lin(num_classes)
    y = custom_lin(fc1, num_classes)
    return y
