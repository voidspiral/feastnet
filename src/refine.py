import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from src.data_process import get_training_data
from src.loss import mesh_loss, laplace_loss, test_loss, laplace_loss_cascade, mask_output
from src.model import get_model_fill, get_model_res

BATCH_SIZE = 1

# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
Mask = tf.placeholder(tf.bool, shape=[None])
# [ coarse_total_size,10]
Adj = tf.placeholder(tf.int32, shape=[None, 10])

# [ coarse_fill_size]
Pidx = tf.placeholder(tf.int32, shape=[None])
# [ coarse_fill_edge_size,2]
Pedge = tf.placeholder(tf.int32, shape=[None, 2])

# [true_hole_size,3]
Y = tf.placeholder(tf.float32, shape=[None, 3])
# [true_hole_size,3]
Ynm = tf.placeholder(tf.float32, shape=[None, 3])

# [input_size,3]
output = get_model_fill(X, Adj)

# loss=tf.reduce_mean(tf.square(X-output))
size = tf.shape(output)
output = tf.where(Mask, output, X)

mesh_loss, Chamfer_loss, edge_loss, normal_loss, dbg_list = mesh_loss(output, Pidx, Pedge, Ynm, Y)
lap_loss = laplace_loss(output, Adj, Pidx)
# lap_loss_c=laplace_loss_cascade(X, output, X_adj, X_add_idx)


total_loss = mesh_loss +  lap_loss
# total_loss=mesh_loss+lap_loss

optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss)

train_data_path = 'F:/ProjectData/surface/rabbit/2aitest'
ckpt_path = 'F:/tf_projects/3D/FeaStNet-master/ckpt'
x, adj, pidx, pedge, y, ynm, mask = \
    get_training_data(train_data_path, load_previous=True)
# x, adj, pidx,pedge, y, ynm,mask=get_training_data(annotation_path)

dir_load = '20190418-1138'  # where to restore the model
# dir_load =None
model_name = 'model.ckpt-430'

saver = tf.train.Saver()

dir_save = datetime.now().strftime("%Y%m%d-%H%M")
save_checkpoints_dir = ckpt_path + '/' + dir_save
os.makedirs(save_checkpoints_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    if dir_load is not None:
        load_checkpoints_dir = os.path.join(ckpt_path, dir_load)
        var_file = os.path.join(load_checkpoints_dir, model_name)
        saver.restore(sess, var_file)  # 从模型中恢复最新变量
    
    epochs = 100000
    min_loss = 1000000
    
    for epoch in range(epochs):
        for x_, adj_, pidx_, pedge_, y_, ynm_, mask_ in zip(x, adj, pidx, pedge, y, ynm, mask):
            feed_in = {X: x_,
                       Adj: adj_,
                       Pidx: pidx_,
                       Pedge: pedge_,
                       Y: y_,
                       Ynm: ynm_,
                       Mask: mask_,
                       }
            
            _, loss, Chamfer_loss1, edge_loss1, normal_loss1, lap_loss1, \
                = sess.run([optimizer,
                            total_loss, Chamfer_loss, edge_loss, normal_loss, lap_loss,
                
                            ],
                           feed_dict=feed_in)
        if epoch % 1 == 0:
            if loss < min_loss:
                min_loss = loss
                print('save ckpt\n')
                saver.save(sess, save_checkpoints_dir + "/model.ckpt", global_step=int(epoch))
            
            print('epoch = %d \nloss=%.4f\nChamfer_loss=%.4f\nedge_loss=%.4f\nnormal_loss=%.4f\nlap_loss=%.4f\n'
                  % (epoch, loss, Chamfer_loss1, edge_loss1, normal_loss1, lap_loss1))
            
            print('===============')

