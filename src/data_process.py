import os

import numpy as np
from trimesh import grouping, remesh
from collections import Counter
margin_opposite_pt=[]
def get_training_data(root_path, load_previous=True):
    load_path = root_path + '/train.npz'
    if load_previous == True and os.path.isfile(load_path):
        data = np.load(load_path)
        print('Loading training data from ' + load_path)
        return data['X'], data['Adj'], data['Pidx'], data['Pedge'], \
               data['Y'], data['Ynm'], data['Mask']
    
    sample_dir = os.listdir(root_path)
    x_list = []
    face_list=[]
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
            face = np.loadtxt(os.path.join(sample_path, 'face.txt'))[:, 1:].astype(np.float32)
            face_idx = np.loadtxt(os.path.join(sample_path, 'face_idx.txt')).astype(np.int32)
            p_idx = np.loadtxt(os.path.join(sample_path, 'x_add_idx.txt')).astype(np.int32)

            p_edge, adj = extract_face(face, p_idx)

            mask = build_mask(x.shape[0], p_idx)

            new_faces, new_face_index, new_pt,mar_pt_id, mar_edges\
                =subdivide(x.shape[0],face,face_idx)
            
            
            
            
            
            y_nm = np.loadtxt(os.path.join(sample_path, 'y_normal.txt')).astype(np.float32)
            y = y_nm[:, :3]
            y_nm = y_nm[:, 3:]
            
            x_list.append(x)
            face_list.append(face)
            adj_list.append(adj)
            p_idx_list.append(p_idx)
            p_edge_list.append(p_edge)
            y_list.append(y)
            y_nm_list.append(y_nm)
            mask_list.append(mask)
    
    X = np.array(x_list)
    Face = np.array(face_list)
    Adj = np.array(adj_list)
    Pidx = np.array(p_idx_list)
    Pedge = np.array(p_edge_list)
    Y = np.array(y_list)
    Ynm = np.array(y_nm_list)
    Mask = np.array(mask_list)
    ################
    
    
    
    ################
    
    if not load_previous:
        np.savez(load_path, X=X, Adj=Adj, Pidx=Pidx, Pedge=Pedge,
                 Y=Y, Ynm=Ynm, Mask=Mask)
    
    return X, Adj, Pidx, Pedge, Y, Ynm, Mask

    p_edges, adj=extract_face(faces, p_idx)
    

def build_mask(num, x_add_idx):
    mask = np.zeros([num])
    mask[x_add_idx - 1] = 1
    mask = mask.astype(np.bool)
    return mask




def gather_from_one(x, idx):
    x = np.concatenate([np.expand_dims(np.zeros_like(x[0], x.dtype), axis=0), x], axis=0)
    x = x[idx]
    return x

def subdivide(vertice_num,
              faces,
              face_index=None):

    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # f =len(face_index)
    # [3*f,2]  0-f: [01] f-2f[12]  2f-3f[20]
    mid= np.stack([faces[:,e]for e in [[0, 1],
                               [1, 2],
                               [2, 0]]])


    # 新增点为 len(unique)=num_edge 个。  mid[unique] 就是所有新增点,
    # 新增点的id: vertice_num ~ vertice_num+len(unique)
    # [num_edge] [3*f]. unique 的每个值代表一个不重复的新中点在所有中点中的索引
    unique, inverse = grouping.unique_rows(mid)
    idx=np.array([id for id, v in Counter(inverse).items() if v==1])
    new_pt=mid[unique]
    mar_pt=idx+vertice_num # 边缘点的 id [mar_num]
    mar_neigb=mid[unique[idx]] #[mar_num,2]
    mar_in_f =unique[idx]//len(face_index)  #[margin_num] kind of edge
    mar_f_id=unique[idx] %  len(face_index) #[margin_num] f id
    other_pt_mar={0:2,1:0,2:1}
    other_class=[other_pt_mar[i] for i in list(mar_in_f)]
    other_id=faces[mar_f_id,other_class] #[margin_num]
    # [f,3]  3 column means 3 kinds of middle points [01] [12] [20]
    # which is duplicated in two adjacency triangles
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T


    # [f,3] 新face关于 新点的索引
    mid_idx = inverse[mid_idx] + vertice_num

    # the new faces with correct winding
    # [4*f,3]
    small_faces = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    # stack [f,3] [3*f,3] -> [4*f,3]
    new_faces = np.vstack((faces, small_faces[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = small_faces[:len(face_index)]
    new_face_index=np.concatenate([face_index,np.arange(len(face_index),4*len(face_index))])
    return  new_faces,new_face_index,new_pt,mar_pt,mar_neigb,other_id

# def extract_edge(p_idx, p_adj):
#     '''
#     the range is corresponding to adj
#     :param p_adj:
#     :return:
#     '''
#     pt_num, K = p_adj.shape
#     first_pt = np.reshape(p_idx, [-1, 1, 1])
#     first_pt = np.tile(first_pt, [1, K, 1])
#
#     p_adj = np.reshape(p_adj, [pt_num, K, 1])
#     pairs = np.concatenate([first_pt, p_adj], axis=-1)
#     pairs = np.reshape(pairs, [pt_num * K, 2])
#     pairs = pairs[np.where(pairs[:, 1] > 0)]
#     return pairs


def extract_face(faces, p_idx, mar_pt, mar_nb):
    '''
    
    :param faces: f,3
    :param p_idx: n
    :param mar_pt:
    :param mar_nb: m,2
    :param mar_f_id: m_f
    :return:
    '''
    #edge-->adj
    #[f,3,2]
    edges = np.stack([faces[:, g]
                      for g in [[0, 1], [1, 0],
                                [1, 2]]],axis=1)

    #[f(x) if condition else g(x) for x in sequence]
    mar_nb= mar_nb[:, [1, 0]]
    mar_num=mar_nb.shape[0]
    # faces=np.expand_dims(faces,axis=2)#[f,3,1,2]
    # (faces == mar_nb).all(axis=-1)#[f,3,m]
    
    mar_nb=np.reshape(mar_nb,[-1,1,1,2]) #[m,1,1,2]
    mar_e=(edges == mar_nb).all(axis=-1)  # [m,f,3] bool 只有m个值为 True
    mar_f=mar_e.any(axis=-1)  # [m,f] bool 只有m个值为 True
    mar_f_ne=np.expand_dims(mar_f,-1)*~mar_e # [m,f,3] bool  有2m个值为 True
    
    #[2m] [2m] [2m]
    n_idx,f_idx,e_idx=np.where(mar_f_ne)
    mar_ne=edges[f_idx,e_idx,:].reshape([mar_num,2,2]) #[m,2,2]
    mar_ne1=mar_ne[:,0,:] #[m,2]
    mar_ne2=mar_ne[:,1,:] #[m,2]
    mar_ne2=mar_ne2[:,:,[1,0]] #[m,2]
    mask=np.where((mar_ne1-mar_ne2)==0)[0] #[m]
    mar_op_pt=mar_ne1[np.arange(mar_num),mask] #[m]
    
    
    # # 只剩下一个face 和一个 margin对应，即洞外face
    # # 分割出 两个 新平面
    
    split_faces1=np.stack([mar_nb[:,0],mar_pt,mar_op_pt],axis=-1) #[m,3]
    split_faces2=np.stack([mar_nb[:,1],mar_op_pt,mar_pt],axis=-1) #[m,3]
    split_faces=np.concatenate([split_faces1,split_faces2]) #[2*m,3]
    faces=np.concatenate([faces,split_faces]) #[f+2m,3]
    
    # [2*m,3,2]
    split_edges = np.stack([split_faces[:, g]
                      for g in [[0, 1], [1, 0],
                                [1, 2]]],axis=1)
    edges=np.concatenate([edges,split_edges]) #[f+2m,3,2]
    
    unique, inverse = grouping.unique_rows(edges)
    edges = edges[unique]
    edges_sym=edges[:,[1,0]]
    edges=np.concatenate([edges,edges_sym])
    edges = edges[np.argsort(edges[:, 0])]
    
    unique, inverse = grouping.unique_rows(edges[:, 0])
    
    p_unique=unique[p_idx-1]
    p_edges=np.stack([edges[st:ed]
           for st, ed in zip(p_unique[:-1], p_unique[1:])])
    adj = [np.concatenate([edges[:, 1][st:ed], np.zeros([10 - (ed - st)])])
           for st, ed in zip(unique[:-1], unique[1:])]
    adj = np.stack(adj)
    

    return p_edges, adj,faces



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

