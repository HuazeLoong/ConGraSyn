# cython: language_level=3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Cython 并行模块（如果需要并行）
from cython.parallel cimport prange, parallel

# NumPy：同时引入 Python 模块和 Cython 类型模块
import numpy as np
cimport numpy as cnp

# 初始化 NumPy 的 C-API（必须）
ctypedef cnp.int64_t INT64_t
cnp.import_array()


def floyd_warshall(cnp.ndarray[cnp.int64_t, ndim=2] adjacency_matrix):
    cdef unsigned int nrows = adjacency_matrix.shape[0]
    cdef unsigned int ncols = adjacency_matrix.shape[1]
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = np.asarray(adjacency_matrix, dtype=np.int64, order='C')
    assert adj_mat_copy.flags['C_CONTIGUOUS']

    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] M = adj_mat_copy
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] path = np.zeros((n, n), dtype=np.int64)

    cdef unsigned int i, j, k
    cdef cnp.int64_t M_ij, M_ik, cost_ikkj
    cdef cnp.int64_t* M_ptr = &M[0, 0]
    cdef cnp.int64_t* M_i_ptr
    cdef cnp.int64_t* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyd algo
    for k in range(n):
        M_k_ptr = M_ptr + n * k
        for i in range(n):
            M_i_ptr = M_ptr + n * i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(unsigned int max_dist, cnp.ndarray[cnp.int64_t, ndim=2] path, cnp.ndarray[cnp.int64_t, ndim=3] edge_feat):
    cdef unsigned int nrows = path.shape[0]
    cdef unsigned int ncols = path.shape[1]
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = np.asarray(path, dtype=np.int64, order='C')
    edge_feat_copy = np.asarray(edge_feat, dtype=np.int64, order='C')
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef cnp.ndarray[cnp.int64_t, ndim=4, mode='c'] edge_fea_all
    cdef int edge_dim = edge_feat.shape[2]  # shape[-1] 会出错，提前取出并转为 C int
    edge_fea_all = -1 * np.ones((n, n, max_dist_copy, edge_dim), dtype=np.int64)


    cdef unsigned int i, j, k, num_path
    cdef list path_list

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path_list = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path_list) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path_list[k], path_list[k + 1], :]

    return edge_fea_all
