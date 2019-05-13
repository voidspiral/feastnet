import os
import time

import tensorflow as tf
import numpy as np
from src.data_process import get_training_data
from src.model import get_model_fill

data_path = 'F:/ProjectData/surface/leg/valid'
# [ coarse_total_size,3]
X = tf.placeholder(tf.float32, shape=[None, 3])
# [ coarse_total_size,10]
X_adj = tf.placeholder(tf.int32, shape=[None, 10])

#[input_size,3]
output = get_model_fill(X,  X_adj)
saver = tf.train.Saver()

NUM_PARALLEL_EXEC_UNITS=4
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_path = 'F:/tf_projects/3D/FeaStNet-master/ckpt'
    dir_save='/20190427-0057'
    save_checkpoints_dir = ckpt_path + '/' + dir_save

    var_file = save_checkpoints_dir+'/model.ckpt-360'
    saver.restore(sess, var_file)  # 从模型中恢复最新变量
    x, x_adj, x_add_idx, x_add_edge, y, y_nm,mask = get_training_data(data_path,load_previous=False)

    feed_in = {X: x[0],
               X_adj:x_adj[0],
               }
    
    for i in range(10):
        time_start = time.time()
        output_array = sess.run(output, feed_dict=feed_in)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
    
    np.savetxt(data_path+'/data.txt',output_array,fmt='%.5f')
    idx_array=np.concatenate([np.expand_dims(x_add_idx[0],1),output_array[x_add_idx[0]-1]],axis=1)
    np.savetxt(data_path+'/p_output.txt',idx_array,fmt='%.5f')
