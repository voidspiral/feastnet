# Preparing a TF model for usage in Android
# By Omid Alemi - Jan 2017
# Works with TF r1.0
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
from src.data_process import get_training_data
from src.model import get_model_fill

MODEL_PATH='F:/tf_projects/3D/FeaStNet-master/ckpt'
# Freeze the graph
input_saver_def_path = ""
input_binary = False
output_node_names = "output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = True


X = tf.placeholder(tf.float32, shape=[None, 3])
# [ coarse_total_size,10]
X_adj = tf.placeholder(tf.int32, shape=[None, 10])
output = get_model_fill(X,  X_adj)
output=tf.identity(output,name='output')




ckpt_path = os.path.join(MODEL_PATH, '20190427-0057')
ckpt_file=os.path.join(ckpt_path, 'model.ckpt-360')

input_graph_path = os.path.join(ckpt_path, 'structure.pb')
output_frozen_graph_name = os.path.join(ckpt_path, 'output_graph.pb')
output_optimized_graph_name = os.path.join(ckpt_path, 'opt_graph.pb')

with tf.Session()as sess:
    gd = sess.graph.as_graph_def()
    
    tf.train.write_graph(gd, ckpt_path, input_graph_path)

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, ckpt_file, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ["Placeholder","Placeholder_1"], # an array of the input node(s)
    ["output"], # an array of output nodes
    # tf.float32.as_datatype_enum
    [tf.float32.as_datatype_enum,tf.int32.as_datatype_enum]
)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())



with tf.gfile.GFile(output_optimized_graph_name, 'rb') as f:
   graph_def_optimized = tf.GraphDef()
   graph_def_optimized.ParseFromString(f.read())

# NUM_PARALLEL_EXEC_UNITS=4
# config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
#                         inter_op_parallelism_threads=2,
#                         allow_soft_placement=True,
#                         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
#
# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["KMP_BLOCKTIME"] = "0"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

G = tf.Graph()
with tf.Session(graph=G) as sess:
    output = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])
    # print('Operations in Optimized Graph:')
    # print([op.name for op in G.get_operations()])
    data_path = 'F:/ProjectData/surface/leg/valid'

    x, x_adj, x_add_idx, x_add_edge, y, y_nm,mask = get_training_data(data_path,load_previous=False)

    X = G.get_tensor_by_name('import/Placeholder:0')
    Adj = G.get_tensor_by_name('import/Placeholder_1:0')
    for i in range(10):
        time_start = time.time()

        output_list = sess.run(output, feed_dict={X: x[0],Adj:x_adj[0]})
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

    output_array=output_list[0]
    np.savetxt(data_path+'/data.txt',output_array,fmt='%.5f')
    idx_array=np.concatenate([np.expand_dims(x_add_idx[0],1),output_array[x_add_idx[0]-1]],axis=1)
    np.savetxt(data_path+'/p_output.txt',idx_array,fmt='%.5f')

    
