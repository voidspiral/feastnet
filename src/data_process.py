import os

import numpy as np
from trimesh import grouping, remesh


def get_training_data(root_path, load_previous=True):
    load_path = root_path + '/train.npz'
    if load_previous == True and os.path.isfile(load_path):
        data = np.load(load_path)
        print('Loading training data from ' + load_path)
        return data['X'], data['Adj'], data['Pidx'], data['Pedge'], \
               data['Y'], data['Ynm'], data['Mask']
    
    sample_dir = os.listdir(root_path)
    x_list = []
    adj_list = []
    p_idx_list = []
    p_edge_list = []
    y_list = []
    y_nm_list = []
    mask_list = []
    
    for sample in sample_dir:
        sample_path = os.path.join(root_path, sample)
        if os.path.isdir(sample_path):
            x = np.loadtxt(os.path.join(sample_path, 'x.txt'))[:, 1:].astype(np.float32)
            adj = np.loadtxt(os.path.join(sample_path, 'x_ad.txt'))[:, 1:11].astype(np.int32)
            p_idx = np.loadtxt(os.path.join(sample_path, 'x_add_idx.txt')).astype(np.int32)
            
            p_adj = adj[p_idx - 1]
            p_edge = extract_edge(p_idx, p_adj)
            mask = build_mask(x.shape[0], p_idx)
            
            y_nm = np.loadtxt(os.path.join(sample_path, 'y_normal.txt')).astype(np.float32)
            y = y_nm[:, :3]
            y_nm = y_nm[:, 3:]
            
            x_list.append(x)
            adj_list.append(adj)
            p_idx_list.append(p_idx)
            p_edge_list.append(p_edge)
            y_list.append(y)
            y_nm_list.append(y_nm)
            mask_list.append(mask)
    
    X = np.array(x_list)
    Adj = np.array(adj_list)
    Pidx = np.array(p_idx_list)
    Pedge = np.array(p_edge_list)
    Y = np.array(y_list)
    Ynm = np.array(y_nm_list)
    Mask = np.array(mask_list)
    if not load_previous:
        np.savez(load_path, X=X, Adj=Adj, Pidx=Pidx, Pedge=Pedge,
                 Y=Y, Ynm=Ynm, Mask=Mask)
    
    return X, Adj, Pidx, Pedge, Y, Ynm, Mask


def build_mask(num, x_add_idx):
    mask = np.zeros([num])
    mask[x_add_idx - 1] = 1
    mask = mask.astype(np.bool)
    return mask


def extract_edge(x_add_idx, add_adj):
    '''
    the range is corresponding to adj
    :param add_adj:
    :return:
    '''
    pt_num, K = add_adj.shape
    first_pt = np.reshape(x_add_idx, [-1, 1, 1])
    first_pt = np.tile(first_pt, [1, K, 1])
    
    add_adj = np.reshape(add_adj, [pt_num, K, 1])
    pairs = np.concatenate([first_pt, add_adj], axis=-1)
    pairs = np.reshape(pairs, [pt_num * K, 2])
    pairs = pairs[np.where(pairs[:, 1] > 0)]
    return pairs


def gather_from_one(x, idx):
    x = np.concatenate([np.expand_dims(np.zeros_like(x[0], x.dtype), axis=0), x], axis=0)
    x = x[idx]
    return x


def extract_faces(faces):
    edges = np.stack([faces[:, g]
                      for g in [[0, 1], [1, 0],
                                [1, 2], [2, 1],
                                [2, 0], [0, 2]]])
    unique, inverse = grouping.unique_rows(edges)
    edges = edges[unique]
    edges = edges[np.argsort(edges[:, 0])]
    
    unique, inverse = grouping.unique_rows(edges[:, 0])
    adj = [np.concatenate([edges[:, 1][st:ed], np.zeros([10 - (ed - st)])])
           for st, ed in zip(unique[:-1], unique[1:])]
    adj = np.stack(adj)
    return edges, adj


# def get_data(vertices,faces,fill_faces_id,cascade_num):
#
#     for i in range(cascade_num):
#         new_vertices, new_faces=remesh.subdivide(vertices,faces,fill_faces_id)
#         extract_faces
#
#
# def extract_edges(faces):
#
# def insert_vertices(init_edge,cascade_num,x):
#     '''
#     update x adj add_adj add_edge  triangle
#     :param x:
#     :param adj:
#     :return: new edge index
#     '''
#     insert_num=init_edge.shape[0]
#
#     insert_pt=x(x[init_edge[:0]]+x[init_edge[:,1]])/2.
#     x=np.concatenate([x,insert_pt],axis=1)
#     adj=
#
#
#
#

def output(annotation_path):
    output = np.loadtxt(annotation_path['x'])[:, 1:].astype(np.float32)
    x_add_idx = np.loadtxt(annotation_path['add_index']).astype(np.int32)
    add = output[x_add_idx - 1]
    add = np.concatenate([np.expand_dims(x_add_idx, 1), add], axis=1)
    np.savetxt(annotation_path['p_output'], add, fmt='%.5f')


if __name__ == '__main__':
    train_data_path = 'F:/ProjectData/surface/rabbit/2aitest'
    
    x, adj, pidx, pedge, y, ynm, mask = get_training_data(train_data_path,
                                                          data_path=train_data_path, load_previous=False)
    
    # output(annotation_path)

