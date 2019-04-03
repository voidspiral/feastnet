import tensorflow as tf

from src.model import get_patches


def get_generate(whole_output, hole_x):
    hole_output = tf.gather(whole_output, hole_x)
    return hole_output


def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)


def mesh_loss(fill_output, gt_nm, edge_index, ground_truth):
    '''
    
    :param fill_output:  [batch_size,coarse_fill_point,3]
    :param gt_nm:  [batch_size,true_hole_size,3]
    :param edge_index:  [batch_size,coarse_fill_edge,2]
    :param adj:
    :param ground_truth:
    :param min_q_index: [batch_size,coarse_fill_size]
    :return:
    '''
    
    # Chamfer_loss
    # [batch_size, coarse_fill_size,1, 3]
    fill_output = tf.expand_dims(fill_output, axis=2)
    # [batch_size,1,    true_hole_size,3]
    ground_truth = tf.expand_dims(ground_truth, axis=1)
    # [batch_size,coarse_fill_size,true_hole_size]
    distance_matrix = tf.reduce_sum(tf.square(fill_output - ground_truth), axis=-1)
    # [batch_size,coarse_fill_size]
    min_q_index = tf.arg_min(distance_matrix, axis=2)
    p_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=2))
    # [batch_size,true_hole_size]
    q_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=1))
    Chamfer_loss = p_distance + 0.55 * q_distance
    
    # edge in graph
    nod1 = tf.map_fn(lambda x: tf.gather(x[0], x[1], axis=0), [fill_output, edge_index[:, 0]])
    # [batch_size,coarse_fill_edge,3]
    nod2 = tf.map_fn(lambda x: tf.gather(x[0], x[1], axis=0), [fill_output, edge_index[:, 1]])
    edge = tf.subtract(nod1, nod2)
    
    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300
    
    # params.shape[:axis] + indices.shape +params.shape[axis + 1:]
    # [batch_size,coarse_fill_point,3] 每个 coarse_fill_point 对应的最近 ground_truth的Normal
    p_q_Normal = tf.map_fn(lambda x: tf.gather(x[0], x[1], axis=0), [gt_nm, min_q_index])
    # [batch_size,coarse_fill_edge,3]
    p_q_Normal = tf.map_fn(lambda x: tf.gather(x[0], x[1], axis=0), [p_q_Normal, edge_index[:, 0]])
    
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(p_q_Normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5
    
    total_loss = Chamfer_loss * 3000 + edge_loss * 300 + normal_loss * 0.5
    return total_loss


def laplace_coord(pred, placeholders, block_id):
    '''
    
    :param pred:  [coarse_fill_point,3]
    :param placeholders:
    :param block_id:
    :return:
    '''
    
    vertex = tf.concat([pred, tf.zeros([1, 3])], 0)
    indices = placeholders['lape_idx'][block_id - 1][:, :8]
    weights = tf.cast(placeholders['lape_idx'][block_id - 1][:, -1], tf.float32)
    
    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1, 1]), [1, 3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace


def laplace_loss(pred1, pred2, placeholders, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, placeholders, block_id)
    lap2 = laplace_coord(pred2, placeholders, block_id)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500
    
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    move_loss = tf.cond(tf.equal(block_id, 1), lambda: 0., lambda: move_loss)
    return laplace_loss + move_loss
