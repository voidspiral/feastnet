import tensorflow as tf

from src.model import get_model

BATCH_SIZE = 1


x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3])
adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 10])
x_hole
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3])
is_training = tf.placeholder(tf.bool)

#[1,input_size,3]
output = get_model(x, adj,is_training)



