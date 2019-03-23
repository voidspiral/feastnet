import numpy as np
import scipy.sparse


def coarsen(A, levels, self_connections=False):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    """
    # graph: weights between nodes of coarser(parent) layer
    # parent:finer_layer_id -> coarser_layer_id
    graphs, parents = metis(A, levels)
    # 根据最顶层id升序，返回自底向上的id二叉树，二叉树的结构定义了层间连接关系
    # perm 和 graph 数量相等，比parent多一个
    perms = compute_perm(parents)

    for i, A in enumerate(graphs):# 细到粗
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

    return graphs, perms[0] if levels > 0 else None


def metis(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)
    #supernode_size = np.ones(N)
    #nd_sz = [supernode_size]
    #count = 0

    #while N > maxsize:
    for _ in range(levels):

        #count += 1

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights [N]
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        # 两个顺序：1.生成顺序，即本层id； 2.基于本层id 的 degree 顺序。二者共同决定下一层 id
        # 按照本层 id 排序。在这个基准上使用上一轮得到的degree increased rid 进行索引，
        # 先聚合 degree 小的点
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        # 为每个不孤立的点分配cluster, 一共有 len(rid) 个点，len(cluster_id)=len(rid)
        # 每个 adjoint 的起点对应的 cluster id , 由该 adjoint 被 cluster 的优先级
        # 决定，没啥意义，单纯的id
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
        parents.append(cluster_id)

        # TO DO
        # COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE 
        #supernode_size = full(   sparse(cluster_id,  ones(N,1) , supernode_size )     )
        #print(cluster_id)
        #print(supernode_size)
        #nd_sz{count+1}=supernode_size;

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        # 每个 adjoint 的起点所属的 cluster id , 由该 adjoint 被 cluster 的优先级决定
        nrr = cluster_id[rr] # [num of adjoint]
        # 求每个 adjoint 的终点对应的 cluster id , 由该 adjoint 被 cluster 的优先级决定
        ncc = cluster_id[cc] #[num of adjoint]
        nvv = vv  # 每个 adjoint 的 weight
        Nnew = cluster_id.max() + 1  # 一共多少 cluster
        # CSR is more appropriate: row,val pairs appear multiple times
        # cluster 的两两组合 作为新的 pair , 创建新的 邻接矩阵
        # 有层间父子id映射 claster_id，和父层的拓扑关系 W, W中 weight越大表示节点连接关系的越强，能够
        # 指导下一次cluster
        # scipy adds the values of the duplicate entries:  merge weights of cluster
        # W 的 index 代表new cluster 的 id
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros() # 稀疏
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS) 忽略自环，weight变小，该点更容易被团结。
        # 但是如果该点已经是团结过好几次的了，那么应该减小它被继续团结的可能，否则会产生吸收黑洞，
        # 所以不能忽略自环
        degree = W.sum(axis=0)
        # degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        # 根据 degree accend 将cluster id 排序。 下一层从degree小的开始 cluster 开始后续 cluster
        # 目的在于先解决孤立点
        rid = np.argsort(ss) #
    # graphs 比 parents 多一层
    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):
    # 只有起点rr排好序(本层cluster的生成顺序)，终点cc是随机的

    nnz = rr.shape[0] # 所有pair的数量
    N = rr[nnz-1] + 1 # 点的数量

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid
            # tid 第一层是随机的，以后每层是 degree 升序
            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id

def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    产生重新排列的索引向量
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last))) # rank the cluster id of the last layer
        #只有最后一层需要排序 indices=[0,1,2]

    for parent in parents[::-1]:
        # from the coarsest level
        #print('parent: {}'.format(parent))

        # Fake nodes go after real ones. len(parent) is the number of real node in this layer
        # add new id for fake nodes of this layer
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            # 每个父节点对应一个 indices_node, 索引上一层中的子节点
            # index of where condition is true
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            #print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singelton.
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
                #print('new singelton: {}'.format(indices_node))
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
                #print('singelton childrens: {}'.format(indices_node))

            indices_layer.extend(indices_node) # 每次加两个元素：上一层的两个索引
        indices.append(indices_layer)

    # Sanity checks.
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices_layer) == M
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1] # finest to coarsest

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
        == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])

def perm_data(x, indices):
    """
    排列成待卷积的序列
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    new x can be used for pooling
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i] = x[:,j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:,i] = np.zeros(N)
    return xnew

def perm_adjacency(A, indices):
    """
    indices 是为了组成使最顶层id为升序的二叉树，本层的id的顺序
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    new A can be used to following convolution and pooling
    """
    if indices is None:
        return A

    M, M = A.shape
    # 45 03 21    if 345 is fake, M=3
    # 0  1  2
    # indices is one of the above two lines
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    # e.g. 254|301
    #      012
    perm = np.argsort(indices)
    # row,col= M.nonzero() 两个array表示非零值的索引，按照row由小到大排序，
    # 相当于 sparse.find(W)+排序
    # 卷积要用到并行，需要将原id按 450321 排列，如下操作
    # e.g. 把id 为0的点从矩阵的 index 0 处移动到 index 2处,这是对adj的操作，
    # 还应该有对x的对应交换操作。交换完成后可以进行 两两 max 的 pooling 操作
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A


def adj_to_A(adj):
    '''
    in np ,used at beginging for once
    :param adj: num_points, K
    :return:
    '''
    num_points, K=adj.shape
    idx=np.arange(num_points)
    idx = np.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
    idx = np.tile(idx, [1, K])  # Create multiple columns, each column has one number repeats repTime
    x = np.reshape(idx, [-1]) # 0000  1111 2222 3333 4444 从0开始
    y =np.reshape(adj, [-1]) # 3200  1300 1200 ...  e.g. 一个环  从1开始
    v= (y!=0).astype(np.float32) # 1100 1100 1100
    y=y-1  # 1200  0200 0100 ...  e.g. 一个环  从0开始
    A = scipy.sparse.coo_matrix((v, (x, y)), shape=(num_points, num_points))
    # A=A.tocsr() # TODO needed or ?
    A.setdiag(0)
    return A


def A_to_adj(num_points,K,A):
    '''
    in np, used after each time of coarsening when new A is created
    in coarsen, A id is begin from 0, while in conv, adj id is begin from 1
    :return: num_points, K
    '''
    idx_row, idx_col, val = scipy.sparse.find(A)
    pair_num=idx_row.shape[0]
    perm = np.argsort(idx_row)
    rr = idx_row[perm]
    cc = idx_col[perm]
    adj = np.zeros([num_points,K], np.int32)
    cur_row = rr[0]
    cur_col=0
    col_count=0
    for i in range(pair_num):
        row=rr[i]
        col=cc[i]
        if row>cur_row:
            adj[cur_row,cur_col:]=0
            col_count+=cur_col
            cur_row=row
            cur_col=0
        
        adj[row,cur_col]=col+1
        cur_col+=1
    return adj


    

    
