import os

import numpy as np
from trimesh import grouping,remesh

def get_training_data(annotation_path, data_path=None,load_previous=False):
    if load_previous == True and os.path.isfile(data_path):
        data = np.load(data_path)
        print('Loading training data from ' + data_path)
        return data['x'], data['adj'], data['y']
    
    x = np.loadtxt(annotation_path['x'])[:,1:].astype(np.float32)
    x_adj = np.loadtxt(annotation_path['adj'])[:,1:11].astype(np.int32)
    
    x_add_idx = np.loadtxt(annotation_path['add_index']).astype(np.int32)
    x_add_adj=x_adj[x_add_idx-1]
    x_add_edge=extract_edge(x_add_idx,x_add_adj)
    
    
    
    y_nm = np.loadtxt(annotation_path['y_normal'])
    y=y_nm[:,:3]
    y_nm=y_nm[:,3:]
    
    return x,x_adj,x_add_idx,x_add_edge,y,y_nm
    

def extract_edge(x_add_idx, add_adj):
    '''
    the range is corresponding to adj
    :param add_adj:
    :return:
    '''
    pt_num,K=add_adj.shape
    first_pt=np.reshape(x_add_idx,[-1,1,1])
    first_pt = np.tile(first_pt, [1, K,1])
    
    
    
    
    add_adj=np.reshape(add_adj, [pt_num, K, 1])
    pairs=np.concatenate([first_pt, add_adj], axis=-1)
    pairs=np.reshape(pairs,[pt_num*K,2])
    pairs=pairs[np.where(pairs[:,1]>0)]
    return pairs


def gather_from_one(x, idx):
    x = np.concatenate([np.expand_dims(np.zeros_like(x[0], x.dtype), axis=0), x], axis=0)
    x = x[idx]
    return x


def extract_faces(faces):
    edges = np.stack([faces[:, g]
                     for g in [[0, 1],[1,0],
                               [1, 2],[2,1],
                               [2, 0],[0,2]]])
    unique, inverse = grouping.unique_rows(edges)
    edges=edges[unique]
    edges=edges[np.argsort(edges[:,0])]
    
    unique, inverse = grouping.unique_rows(edges[:,0])
    adj=[np.concatenate([edges[:,1][st:ed],np.zeros([10-(ed-st)])])
         for st,ed in zip(unique[:-1],unique[1:])]
    adj=np.stack(adj)
    return edges,adj
    
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
    
    output = np.loadtxt(annotation_path['x'])[:,1:].astype(np.float32)
    x_add_idx = np.loadtxt(annotation_path['add_index']).astype(np.int32)
    add=output[x_add_idx-1]
    add=np.concatenate([np.expand_dims(x_add_idx,1),add],axis=1)
    np.savetxt(annotation_path['p_output'],add,fmt='%.5f')

  
    
    
if __name__ == '__main__':
    root_data_path = 'F:/tf_projects/3D/FeaStNet-master/data'
    annotation_path = {
        'x': root_data_path + '/x.txt',
        'adj': root_data_path + '/x_adj.txt',
        'add_index': root_data_path + '/x_add_idx.txt',
        'y_normal': root_data_path + '/y_normal.txt',
        'output': root_data_path + '/output.txt',
        'p_output': root_data_path + '/p_output.txt'
        
    }


    output(annotation_path)

