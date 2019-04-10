import tensorflow as tf

from src.model import get_patches


def get_generate(whole_output, hole_x):
    hole_output = tf.gather(whole_output, hole_x)
    return hole_output


def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)


def mesh_loss(pred, edge_index, gt_nm, ground_truth):
    '''
    
    :param fill_output:  [coarse_fill_size,3]
    :param gt_nm:  [true_hole_size,3]
    :param edge_index:  [coarse_fill_edge,2]
    :param ground_truth: [true_hole_size,3]
    :param min_q_index: [coarse_fill_size]
    :return:
    '''
    
    # Chamfer_loss
    # [ coarse_fill_size,1, 3]
    fill_output = tf.expand_dims(pred, axis=1)
    # [1,    true_hole_size,3]
    ground_truth = tf.expand_dims(ground_truth, axis=0)
    # [coarse_fill_size,true_hole_size]
    distance_matrix = tf.reduce_sum(tf.square(fill_output - ground_truth), axis=-1)
    # [coarse_fill_size]
    min_q_index = tf.arg_min(distance_matrix, axis=1)
    
    p_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=1))
    q_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=0))
    Chamfer_loss = p_distance + 0.55 * q_distance
    
    # [coarse_fill_edge,3]
    nod1 = tf.gather(pred, edge_index[:, 0])
    # [coarse_fill_edge,3]
    nod2 = tf.gather(pred, edge_index[:, 1])
    # [coarse_fill_edge,3]
    edge = tf.subtract(nod1, nod2)
    
    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300
    
    # params.shape[:axis] + indices.shape +params.shape[axis + 1:]
    # [coarse_fill_point,3] 每个 coarse_fill_point 对应的最近 ground_truth的Normal

    p_q_Normal = tf.gather(gt_nm, min_q_index)
    # coarse_fill_edge 个起点对应的q的 normal
    # [coarse_fill_edge,3]
    p_q_Normal = tf.gather(p_q_Normal, edge_index[:, 0])
    # coarse_fill_edge个cross_product求和
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(p_q_Normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5
    
    total_loss = Chamfer_loss * 3000 + edge_loss * 300 + normal_loss * 0.5
    return total_loss


def laplace_coord(pred, adj):
    '''
    
    :param pred:  [coarse_fill_point,3]
    :param placeholders:
    :param block_id:
    :return:
    '''
    adj_size = tf.count_nonzero(adj, 2)  # [coarse_fill_point] 每个元素 是该点的邻接点数量
    # deal with unconnected points: replace NaN with 0
    non_zeros = tf.not_equal(adj_size, 0)  # [coarse_fill_point] bool  是否有孤立点
    adj_size = tf.cast(adj_size, tf.float32)
    # [coarse_fill_point]
    adj_weights = tf.where(non_zeros, tf.reciprocal(adj_size), tf.zeros_like(adj_size))  # 非孤立点 删选出来
    # [coarse_fill_point,3]
    adj_weights=tf.tile(tf.expand_dims(adj_weights,-1),[1,3])
    # [coarse_fill_point, 3]
    neigbor_sum = tf.reduce_sum(get_patches(pred, adj), 1)
    laplace = tf.subtract(pred, tf.multiply(neigbor_sum, adj_weights))
    return laplace


def laplace_loss(pred1, adj):
    '''
    仅仅是居中
    :param pred1:
    :param adj:
    :return:
    '''
    # laplace term
    lap1 = laplace_coord(pred1, adj)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap1))) * 1500
    return laplace_loss

def laplace_loss_casade(pred1, pred2, adj):
    # laplace term
    lap1 = laplace_coord(pred1, adj)
    lap2 = laplace_coord(pred2, adj)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500
    
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    move_loss = tf.cond(tf.equal(block_id, 1), lambda: 0., lambda: move_loss)
    return laplace_loss + move_loss
