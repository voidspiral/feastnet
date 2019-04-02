import tensorflow as tf


def get_generate(whole_output,hole_x):
    
    hole_output=tf.gather(whole_output,hole_x)
    return hole_output

def Chamfer_loss(fill_output,ground_truth):
    '''
    
    :param hole_output: [coarse_fill_size,3]
    :param ground_truth: [true_hole_size,3]
    :return:
    '''
    fill_output=tf.expand_dims(fill_output,axis=0)
    ground_truth=tf.expand_dims(ground_truth,axis=1)
    # [coarse_fill_size,true_hole_size,3]
    distance_matrix=tf.square(fill_output-ground_truth)
    p_distance=tf.reduce_sum(tf.reduce_min(distance_matrix,axis=0))
    q_distance=tf.reduce_sum(tf.reduce_min(distance_matrix,axis=1))
    loss=p_distance+q_distance
    return loss
    
    