import tensorflow as tf
import numpy as np
from src.data_process import get_training_data
from src.model import get_model_fill

data_path = 'F:/tf_projects/3D/FeaStNet-master/data'

# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
# [ coarse_total_size,10]
X_adj = tf.placeholder(tf.int32, shape=[None, 10])

# [ coarse_fill_size]
X_add_idx= tf.placeholder(tf.int32, shape=[None])
# [ coarse_fill_edge_size,2]



#[input_size,3]
output = get_model_fill(X,  X_adj)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_path = 'F:/tf_projects/3D/FeaStNet-master/ckpt'
    dir_save='/20190413-1931'
    save_checkpoints_dir = ckpt_path + '/' + dir_save

    var_file = save_checkpoints_dir+'/model.ckpt-4'
    saver.restore(sess, var_file)  # 从模型中恢复最新变量
    annotation_path = {
        'x': data_path + '/x.txt',
        'adj': data_path + '/x_adj.txt',
        'add_index': data_path + '/x_add_idx.txt',
        'y_normal': data_path + '/y_normal.txt'
    }
    x, x_adj, x_add_idx, x_add_edge, y, y_nm,mask = get_training_data(annotation_path)

    feed_in = {X: x,
               X_adj:x_adj,
               X_add_idx:x_add_idx
               }
    output_array = sess.run(output, feed_dict=feed_in)
    np.savetxt(data_path+'/data.txt',output_array,fmt='%.5f')

    idx_array=np.concatenate([np.expand_dims(x_add_idx,1),output_array[x_add_idx-1]],axis=1)
    np.savetxt(data_path+'/p_output.txt',idx_array,fmt='%.5f')
