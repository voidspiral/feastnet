import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from src.data_process import get_training_data
from src.loss import mesh_loss, laplace_loss, test_loss, laplace_loss_cascade, mask_output
from src.model import get_model_fill

BATCH_SIZE = 1

# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
Mask = tf.placeholder(tf.bool, shape=[None])
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
output=tf.where(Mask, output, X)

mesh_loss,Chamfer_loss,edge_loss,normal_loss,dbg_list= mesh_loss(output, X_add_idx,X_add_edge, Y_nm, Y)
lap_loss=laplace_loss(output, X_adj,X_add_idx)
# lap_loss_c=laplace_loss_cascade(X, output, X_adj, X_add_idx)


total_loss= mesh_loss+10*lap_loss
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
x, x_adj, x_add_idx,x_add_edge, y, y_nm,mask=get_training_data(annotation_path)

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
               Mask:mask,
               Y:y,
               Y_nm:y_nm,
               }
    
    for epoch in range(epochs):
    
        _,loss,Chamfer_loss1,edge_loss1,normal_loss1,lap_loss1,\
        nod1,nod2,e_idx,e_len,var\
            =sess.run([optimizer,
                        total_loss,Chamfer_loss,edge_loss,normal_loss,lap_loss,
                       dbg_list[0],dbg_list[1],dbg_list[2],dbg_list[3],dbg_list[4]
                       ],
                       feed_dict=feed_in)
        edge_len=np.reshape(e_len,[-1,1])
        if epoch%50==0:
            if  loss< min_loss:
                min_loss=loss
                print('save ckpt\n')
                saver.save(sess, save_checkpoints_dir+"/model.ckpt", global_step=int(epoch/50))

            print('epoch = %d \nloss=%.4f\nChamfer_loss=%.4f\nedge_loss=%.4f\nnormal_loss=%.4f\nlap_loss=%.4f\n'
                  %(epoch,loss,Chamfer_loss1,edge_loss1,normal_loss1,lap_loss1))

            print('===============')
    
    
    output_array= sess.run(output,feed_dict=feed_in)
    np.savetxt(data_path+'/data.txt',output_array,fmt='%.5f')

    idx_array=np.concatenate([np.expand_dims(x_add_idx,1),output_array[x_add_idx-1]],axis=1)
    np.savetxt(data_path+'/p_output.txt',idx_array,fmt='%.5f')
    exit()
