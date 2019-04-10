import os

import numpy as np
from trimesh import grouping,remesh

def get_training_data(annotation_path, data_path=None,load_previous=False):
    if load_previous == True and os.path.isfile(data_path):
        data = np.load(data_path)
        print('Loading training data from ' + data_path)
        return data['x'], data['adj'], data['y']
    
    x = np.loadtxt(annotation_path['x'])[:,1:]
    x_adj = np.loadtxt(annotation_path['adj'])[:,1:]
    
    x_add_index = np.loadtxt(annotation_path['add_index']).astype(np.int32)
    x_adj_1=np.concatenate([np.expand_dims(np.zeros_like(x_adj[0]),0),x_adj])
    x_add_adj=x_adj_1[x_add_index]
    x_add_edge=extract_edge(x_add_adj)
    
    
    
    y_nm = np.loadtxt(annotation_path['y_normal'])
    y=y_nm[:,:3]
    y_nm=y_nm[:,3:]
    
    return x,x_adj,x_add_index,x_add_edge,y,y_nm
    

def extract_edge(adj):
    '''
    the range is corresponding to adj
    :param adj:
    :return:
    '''
    pt_num,K=adj.shape
    first_pt=np.arange(1,pt_num+1)
    first_pt=np.reshape(first_pt,[pt_num,1,1])
    first_pt = np.tile(first_pt, [1, K,1])
    adj=np.reshape(adj,[pt_num,K,1])
    pairs=np.concatenate([first_pt,adj],axis=-1)
    pairs=np.reshape(pairs,[pt_num*K,2])
    pairs=pairs[np.where(pairs[:,1]>0)]
    return pairs

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