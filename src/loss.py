import tensorflow as tf

from src.model import get_patches


def get_generate(whole_output, hole_x):
    hole_output = tf.gather(whole_output, hole_x)
    return hole_output


def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)

def test_loss(pred,ground_truth):
    return  tf.reduce_mean(tf.reduce_sum(tf.square(pred-ground_truth),-1))

def gather_from_one(x,idx):
    
    x=tf.concat([tf.expand_dims(tf.zeros_like(x[0],x.dtype),0),x])

    x = tf.gather(x, idx)
    return x

    
def mesh_loss(pred, edge_idx, gt_nm, ground_truth):
    '''
    
    :param pred:  [x_size,3]
    :param gt_nm:  [true_hole_size,3]
    :param edge_idx:  [add_edge,2]
    :param ground_truth: [true_hole_size,3]
    :param min_q_idx: [add_size]
    :return:
    '''
    debug_log=[]
    # Chamfer_loss

    # [ x_size,1, 3]
    pred_e = tf.expand_dims(pred, axis=1)
    # [1,    true_hole_size,3]
    ground_truth = tf.expand_dims(ground_truth, axis=0)
    # [x_size,true_hole_size]
    distance_matrix = tf.reduce_sum(tf.square(pred_e - ground_truth), axis=-1)
    # [x_size] from 0
    min_q_idx = tf.argmin(distance_matrix, axis=1)
    debug_log.append(min_q_idx)
    tf.reduce_min(distance_matrix, axis=1)
    p_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=1))
    p_distance = tf.reduce_min(distance_matrix, axis=1) #[x_size]
    
    
    
    
    q_distance = tf.reduce_sum(tf.reduce_min(distance_matrix, axis=0))
    
    
    
    Chamfer_loss = p_distance + 0.55 * q_distance
    
    zeros = tf.zeros([1, 3], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    pred_1 = tf.concat([zeros, pred], 0) #[N+1, C]
    # [add_edge,3]
    nod1 = tf.gather(pred_1, edge_idx[:, 0])
    # [add_edge,3]
    nod2 = tf.gather(pred_1, edge_idx[:, 1])
    # [add_edge,3]
    edge = tf.subtract(nod1, nod2)
    
    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length)
    
    # params.shape[:axis] + indices.shape +params.shape[axis + 1:]
    # [x_size,3] 每个 coarse_fill_point 对应的最近 ground_truth的Normal
    p_q_Normal = tf.gather(gt_nm, min_q_idx)
    zeros = tf.zeros([1, 3], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    p_q_Normal = tf.concat([zeros, p_q_Normal], 0) #[x_size+1, 3]

    
    # add_edge 个起点对应的q的 normal
    # [add_edge,3]
    p_q_Normal = tf.gather(p_q_Normal, edge_idx[:, 0])
    # add_edge个cross_product求和
    # reduce_sum([add_edge,3] x [add_edge,3])=[add_edge]
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(p_q_Normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine)
    
    # total_loss = Chamfer_loss * 3000 + edge_loss * 300 + normal_loss * 0.5
    total_loss = Chamfer_loss
    return total_loss,Chamfer_loss,edge_loss,normal_loss


def laplace_coord(pred, adj ,add_idx):
    '''
    
    :param pred:  [x_size,3]
    :param placeholders:
    :param block_id:
    :return:
    '''
    adj_size = tf.count_nonzero(adj, 1)  # [coarse_fill_point] 每个元素 是该点的邻接点数量
    
    adj_size = tf.cast(adj_size, tf.float32)
    # [x_size]
    adj_weights = tf.reciprocal(adj_size)
    # [x_size,3]
    adj_weights=tf.tile(tf.expand_dims(adj_weights,-1),[1,3])
    
    patch_avg=tf.reduce_sum(get_patches(pred, adj),axis=1) #[x_size,3]
    
    laplace = tf.subtract(pred, tf.multiply(patch_avg, adj_weights))
    
    zeros = tf.zeros([1, 3], dtype=tf.float32)
    laplace = tf.concat([zeros, laplace], 0) #[x_size+1, 3]
    add_laplace = tf.gather(laplace, add_idx) #[add_size,3]

    return add_laplace


def laplace_loss(pred1, adj,add_idx):
    '''
    仅仅是居中
    :param pred1:
    :param adj:
    :return:
    '''
    # laplace term
    lap = laplace_coord(pred1, adj,add_idx) #[add_size,3]
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap),1)) * 1500
    return laplace_loss

def laplace_loss_casade(pred1, pred2, adj):
    # laplace term
    lap1 = laplace_coord(pred1, adj)
    lap2 = laplace_coord(pred2, adj)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500
    
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    move_loss = tf.cond(tf.equal(block_id, 1), lambda: 0., lambda: move_loss)
    return laplace_loss + move_loss
