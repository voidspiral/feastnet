import tensorflow as tf

from src.data_process import get_training_data
from src.loss import mesh_loss, laplace_loss
from src.model import get_model_fill

BATCH_SIZE = 1

# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
# [ coarse_total_size,10]
X_adj = tf.placeholder(tf.int32, shape=[None, 10])
X_add_adj = tf.placeholder(tf.int32, shape=[None, 10])
# 软件产生
# [ coarse_fill_size]
X_add_idx= tf.placeholder(tf.int32, shape=[None])
# [ coarse_fill_edge_size,2]
X_add_edge=tf.placeholder(tf.int32, shape=[None, 2])

# [true_hole_size,3]
Y= tf.placeholder(tf.float32, shape=[None, 3])
# [true_hole_size,3]
Y_nm=tf.placeholder(tf.float32, shape=[None, 3])


#[input_size,3]
output = get_model_fill(X,  X_adj)

zeros = tf.zeros([1, 3], dtype=tf.float32)
# 索引为0的邻接点，会索引到 0,0
output = tf.concat([zeros, output], 0)  # [N+1, C]

# [coarse_fill_size, 3]
pred_add=tf.gather(output, X_add_idx)
# loss= mesh_loss(pred_add, X_add_edge, Y_nm, Y) + laplace_loss(pred_add, X_add_adj)
loss_total,Chamfer_loss,edge_loss,normal_loss= mesh_loss(pred_add, X_add_edge, Y_nm, Y)
loss=normal_loss
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)


data_path='F:/ProjectData/surface/Aitest 22'
annotation_path={
    'x':data_path+'/x.txt',
    'adj':data_path+'/x_adj.txt',
    'add_index':data_path+'/x_add_idx.txt',
    'y_normal':data_path+'/y_normal.txt'
    
}
x, x_adj, x_add_idx, x_add_adj, x_add_edge, y, y_nm=get_training_data(annotation_path)

with tf.Session()as sess:
    epochs=10000
    for epoch in range(epochs):
        feed_in = {X: x,
                   X_adj:x_adj,
                   X_add_idx:x_add_idx,
                   X_add_adj:x_add_adj,
                   X_add_edge:x_add_edge,
                   Y:y,
                   Y_nm:y_nm,
                   }
    
        sess.run(tf.global_variables_initializer())
        feed_dict={}
        _,loss_train=sess.run([optimizer, loss], feed_dict=feed_in)
        print('loss_train = %.5f'%(loss_train))