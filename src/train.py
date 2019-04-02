from __future__ import division

import os

import tensorflow as tf
import numpy as np
import math
import time
import h5py
import argparse

from src.coarsening import adj_to_A, coarsen, A_to_adj
from src.data_process import get_training_data
from src.model import *
from src.utils import *

random_seed = 0
np.random.seed(random_seed)

sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=int, default=0)
parser.add_argument('--dataset_path')
parser.add_argument('--results_path')
parser.add_argument('--num_iterations', type=int, default=50000)
parser.add_argument('--num_input_channels', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_points', type=int)
parser.add_argument('--num_classes', type=int)

FLAGS = parser.parse_args()

ARCHITECTURE = FLAGS.architecture
DATASET_PATH = FLAGS.dataset_path
RESULTS_PATH = FLAGS.results_path
NUM_ITERATIONS = FLAGS.num_iterations
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
LEARNING_RATE = FLAGS.learning_rate
NUM_POINTS = FLAGS.num_points
NUM_CLASSES = FLAGS.num_classes
BATCH_SIZE = 16
IN_CHANNELS = 3
K = 4
COARSEN_LEVEL=2

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
Load dataset 
x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
										  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
										  e.g. [16,10,4] 16 batch, 10 vertice with 4 neib for each
"""

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
adj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])
adj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])
perm0 = tf.placeholder(tf.int32, shape=[None])

y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, NUM_CLASSES])
# conv2.shape:  [batch_size, input_size, out_channel]
# output = get_model(x, [adj0, adj1], [perm0], NUM_CLASSES)
output = get_model(x, adj0, NUM_CLASSES)

batch = tf.Variable(0, trainable=False)

# Standard classification loss
# cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv), axis=1))
cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output), axis=1))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=batch)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Checkpoint restored\n")

# Train for the dataset

# []
x_train, adj_train, y_train = get_training_data('')
x_valid, adj_valid, y_valid = get_training_data('')
case_num = x_train.shape[0]
perm = np.arange(case_num)

for iter in range(NUM_ITERATIONS):
    for i in range(case_num):
        A_0 = adj_to_A(adj_train[i])
        perm_0,A_1=coarsen(A_0, x_train[i], COARSEN_LEVEL)
        adj_1=A_to_adj(NUM_POINTS,K,A_1)
        _,loss_train = sess.run([train_step,cross_entropy], feed_dict={
            x: x_train[i],
            adj0: adj_train[i],
            adj1:adj_1,
            perm0:perm_0,
            y: y_train[i]})
        
    np.random.shuffle(perm)  # 打乱
    x_train = x_train[perm]
    adj_train = adj_train[perm]
    y_train = y_train[perm]

    if iter % 1000 == 0:
        A_0 = adj_to_A(adj_valid[0])
        perm_0,A_1=coarsen(A_0, x_valid[0], COARSEN_LEVEL)
        adj_1=A_to_adj(NUM_POINTS,K,A_1)

        loss_train = sess.run([cross_entropy], feed_dict={
            x: x_valid[0],
            adj0: adj_valid[0],
            adj1:adj_1,
            perm0:perm_0,
            y: y_valid[0]})
