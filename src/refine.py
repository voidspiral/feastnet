import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from src.data_process import get_training_data
from src.loss import mesh_loss, laplace_loss, test_loss, laplace_loss_cascade
from src.model import get_model_fill

BATCH_SIZE = 1

# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
# [ coarse_total_size,10]
X_adj = tf.placeholder(tf.int32, shape=[None, 10])

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

# loss=tf.reduce_mean(tf.square(X-output))
size=tf.shape(output)

mesh_loss,Chamfer_loss,edge_loss,normal_loss= mesh_loss(output, X_add_idx,X_add_edge, Y_nm, Y)
lap_loss=laplace_loss(output, X_adj,X_add_idx)
lap_loss_c=laplace_loss_cascade(X, output, X_adj, X_add_idx)


total_loss= mesh_loss + 200 * lap_loss + 100 * lap_loss_c
# total_loss=mesh_loss+lap_loss

optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss)

data_path = 'F:/tf_projects/3D/FeaStNet-master/data'
ckpt_path='F:/tf_projects/3D/FeaStNet-master/ckpt'
annotation_path={
    'x':data_path+'/x.txt',
    'adj':data_path+'/x_adj.txt',
    'add_index':data_path+'/x_add_idx.txt',
    'y_normal':data_path+'/y_normal.txt'
    
}
x, x_adj, x_add_idx,x_add_edge, y, y_nm=get_training_data(annotation_path)

# dir_load = '/20190312-1528'  # where to restore the model
dir_load =None
model_name = 'model.ckpt-59'

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

    epochs=100000
    min_loss=10000
    feed_in = {X: x,
               X_adj:x_adj,
               X_add_idx:x_add_idx,
               X_add_edge:x_add_edge,
               Y:y,
               Y_nm:y_nm,
               }
    
    for epoch in range(epochs):
    
        _,loss,Chamfer_loss1,edge_loss1,normal_loss1,lap_loss1,lap_loss2\
            =sess.run([optimizer,
                        total_loss,Chamfer_loss,edge_loss,normal_loss,lap_loss,
                       lap_loss_c],
                       feed_dict=feed_in)
        if epoch%100==0:
            if  Chamfer_loss1< min_loss:
                min_loss=Chamfer_loss1
                print('save ckpt\n')
                saver.save(sess, save_checkpoints_dir+"/model.ckpt", global_step=int(epoch/100))

            print('epoch = %d \nChamfer_loss=%.2f\nedge_loss=%.2f\nnormal_loss=%.2f\nlap_loss1=%.2f\nlap_loss2=%.2f\n'
                  %(epoch,Chamfer_loss1,edge_loss1,normal_loss1,lap_loss1,lap_loss2))

            print('===============')
    
    
    output_array= sess.run(output,feed_dict=feed_in)
    np.savetxt(data_path+'/data.txt',output_array,fmt='%.5f')

    idx_array=np.concatenate([np.expand_dims(x_add_idx,1),output_array[x_add_idx-1]],axis=1)
    np.savetxt(data_path+'/p_output.txt',idx_array,fmt='%.5f')
    exit()
