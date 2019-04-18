import tensorflow as tf

from src.model import get_patches_1


def get_generate(whole_output, hole_x):
    hole_output = tf.gather(whole_output, hole_x)
    return hole_output


def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)

def test_loss(pred,ground_truth):
    return  tf.reduce_mean(tf.reduce_sum(tf.square(pred-ground_truth),-1))

# def gather_from_one(x,idx):
#
#     x=tf.concat([tf.expand_dims(tf.zeros_like(x[0],x.dtype),axis=0),x],axis=0)
#
#     x = tf.gather(x, idx)
#     return x

def mask_output(pred, X, mask):
    tf.where(mask, pred, X)
    
    def then_expression_fn():
        return pred
    
    def else_expression_fn():
        return X
    
    pred = tf.cond(mask, then_expression_fn, else_expression_fn)
    return pred

def build_mask(x_size, p_idx):
    mask = tf.zeros([x_size])
    mask[p_idx - 1] = 1
    mask=tf.cast(mask,tf.bool)
    return mask

def batch_loss(pred, exist, adj, p_idx, p_edge_idx, gt_nm, ground_truth, x_size):
    total_loss, Chamfer_loss, edge_loss, lap_loss = tf.map_fn(
        lambda inputs: sample_loss(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7]),
        [pred, exist, adj, p_idx, p_edge_idx, gt_nm, ground_truth, x_size],
        dtype=(tf.float32, tf.float32,tf.float32, tf.float32))
    total_loss=tf.reduce_mean(total_loss)
    Chamfer_loss=tf.reduce_mean(Chamfer_loss)
    edge_loss=tf.reduce_mean(edge_loss)
    lap_loss=tf.reduce_mean(lap_loss)
    return total_loss, Chamfer_loss, edge_loss, lap_loss


def sample_loss(pred, exist, adj, p_idx, p_edge_idx, gt_nm, ground_truth, x_size):
    mask=build_mask(x_size,p_idx)
    pred = tf.where(mask, pred, exist)

    loss, Chamfer_loss, edge_loss, normal_loss = mesh_loss(pred, p_idx, p_edge_idx, gt_nm, ground_truth)
    
    lap_loss = laplace_loss(pred, adj, p_idx)
    total_loss=loss+10*lap_loss
    return  total_loss,Chamfer_loss, edge_loss,lap_loss


def mesh_loss(pred, p_idx, p_edge_idx, gt_nm, ground_truth):
    '''

    :param pred:  [x_size,3]
    :param gt_nm:  [true_hole_size,3]
    :param p_edge_idx:  [add_edge,2]
    :param ground_truth: [true_hole_size,3]
    :param min_q_idx: [add_size]
    :return:
    '''

    
    # Chamfer_loss
    # [x_size, 1, 3]
    pred1 = tf.expand_dims(pred, axis=1)
    # [1, y_size,  3]
    ground_truth = tf.expand_dims(ground_truth, axis=0)
    # [x_size,y_size]
    distance_matrix = tf.reduce_sum(tf.square(pred1 - ground_truth), axis=-1)
    # [x_size] from 0
    min_q_idx = tf.argmin(distance_matrix, axis=1)
    
    p_distance = tf.reduce_min(distance_matrix, axis=1)  # [x_size]
    p_distance = tf.gather(p_distance, p_idx - 1)  # [p_size]
    p_distance = tf.reduce_sum(p_distance)  # []
    
    distance_matrix = tf.gather(distance_matrix, p_idx - 1)  # [p_size,y_size]
    q_distance = tf.reduce_min(distance_matrix, axis=0)  # [y_size]
    q_distance = tf.reduce_sum(q_distance)
    
    # 如果目标点非常多，那么 p_distance可以占比较大，否则会聚拢
    Chamfer_loss = 0.5 * p_distance + q_distance
    
    
    # [add_edge,3]
    p_nod1 = tf.gather(pred, p_edge_idx[:, 0] - 1)
    # [add_edge,3]
    
    p_nod2 = tf.gather(pred, p_edge_idx[:, 1] - 1)
    # [add_edge,3]
    p_edge = tf.subtract(p_nod1, p_nod2)
    # edge length loss [add_edge]
    edge_length = tf.reduce_sum(tf.square(p_edge), 1)  # [add_edge]
    mean, var = tf.nn.moments(edge_length, axes=[0])
    edge_loss = mean + 10 * var
    # edge_loss=mean
    
    # params.shape[:axis] + indices.shape +params.shape[axis + 1:]
    # [x_size,3] 每个 coarse_fill_point 对应的最近 ground_truth的Normal
    x_q_Normal = tf.gather(gt_nm, min_q_idx)  # [x_size,3]
    
    # add_edge 个起点对应的q的 normal
    # [add_edge,3]
    p_q_Normal = tf.gather(x_q_Normal, p_edge_idx[:, 0] - 1)
    # add_edge个cross_product求和
    # reduce_sum([add_edge,3] x [add_edge,3])=[add_edge]
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(p_q_Normal), unit(p_edge)), 1))
    normal_loss = tf.reduce_mean(cosine)
    
    total_loss = Chamfer_loss + 100 * edge_loss + 10 * normal_loss
    return total_loss, Chamfer_loss, edge_loss, normal_loss


def laplace_coord(pred, adj, p_idx):
    '''

    :param pred:  [x_size,3]
    :param placeholders:
    :param block_id:
    :return:
    '''
    adj_size = tf.count_nonzero(adj, 1)  # [x_size] 每个元素 是该点的邻接点数量
    
    adj_size = tf.cast(adj_size, tf.float32)
    # [x_size]
    adj_weights = tf.reciprocal(adj_size)
    # [x_size,3]
    adj_weights = tf.tile(tf.expand_dims(adj_weights, -1), [1, 3])
    
    patch_avg = tf.reduce_sum(get_patches_1(pred, adj), axis=1)  # [x_size,3]
    
    laplace = tf.subtract(pred, tf.multiply(patch_avg, adj_weights))
    
    p_laplace = tf.gather(laplace, p_idx - 1)  # [add_size,3]
    
    return p_laplace


def laplace_loss(pred, adj, p_idx):
    '''
    仅仅是居中
    :param pred:
    :param adj:
    :return:
    '''
    # laplace term
    lap = laplace_coord(pred, adj, p_idx)  # [add_size,3]
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap), 1))
    return laplace_loss


def laplace_loss_cascade(pred1, pred2, adj, add_idx):
    # laplace term
    lap1 = laplace_coord(pred1, adj, add_idx)
    lap2 = laplace_coord(pred2, adj, add_idx)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1))
    
    # move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    # move_loss = tf.cond(tf.equal(block_id, 1), lambda: 0., lambda: move_loss)
    return laplace_loss
