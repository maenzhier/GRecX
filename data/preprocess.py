# coding = utf8

import tensorflow as tf
from diffnet.data.load_data import user_users_dict, user_user_edges, num_users, num_items, train_user_items_dict
from diffnet.data.load_data import user_item_edges, train_user_item_edges
import scipy.sparse as sp
from scipy.sparse import *
import numpy as np

from tqdm import tqdm
from diffnet.data.utils import save_graph


def norm_adj_rows(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    # return sparse_to_tuple(adj)
    return adj


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def surf(a, b, epoches):
    P0 = sp.eye(a.shape[0])
    # alpha = 0.15
    alpha = 0.2
    # M = np.zeros([a.shape[0], a.shape[0]])
    P = P0
    if epoches > 1:
        P = P @ a @ b * (1-alpha) + P0 * alpha
        M = P
        for _ in tqdm(range(epoches-1)):
            P = P @ a @ b * (1-alpha) + P0 * alpha
            M += P
        return M
    else:
        P = P @ a @ b * (1-alpha) + P0 * alpha
        return P


def social_surf(a, epoches):
    P0 = sp.eye(a.shape[0])
    # alpha = 0.15
    alpha = 0.2
    # M = np.zeros([a.shape[0], a.shape[0]])
    P = P0
    if epoches > 1:
        P = P @ a * (1-alpha) + P0 * alpha
        M = P
        for _ in tqdm(range(epoches-1)):
            P = P @ a * (1-alpha) + P0 * alpha
            M += P
        return M
    else:
        P = P @ a * (1-alpha) + P0 * alpha
        return P


def build_user_item_graph():
    row_col = list(zip(*train_user_item_edges))
    data = np.asarray([1.] * len(train_user_item_edges)).astype(np.float32)
    user_item_adj = csr_matrix((data, (row_col[0], row_col[1])), shape=(num_users, num_items))
    item_user_adj = csr_matrix((data, (row_col[1], row_col[0])), shape=(num_items, num_users))

    user_user_social_adj = csr_matrix((np.asarray([1.] * len(user_user_edges)).astype(np.float32),
                                       list(zip(*user_user_edges))), shape=(num_users, num_users))

    norm_user_item_adj = norm_adj_rows(user_item_adj)
    norm_item_user_adj = norm_adj_rows(item_user_adj)
    norm_user_user_social_adj = norm_adj_rows(user_user_social_adj)

    #0_ = 322(0.005,0.001,0.001),
    #00_ = 322(0.005,0.001,0.001), alpha = 0.2
    #01_ = 211(0.005,0.001,0.001), alpha = 0.2
    #02_ = 434(0.005,0.002,0.0005), alpha = 0.2
    #03_ = 545(0.005,0.003,0.0001), alpha = 0.2
    #04_ = 656(0.005,0.002,0.0005), alpha = 0.2
    #f_0_ = 322(0.005,0.001,0.001), alpha = 0.2, flickr dataset
    user_user_graph = surf(norm_user_item_adj, norm_item_user_adj, 3)
    item_item_graph = surf(norm_item_user_adj, norm_user_item_adj, 2)
    user_user_social_graph = social_surf(norm_user_user_social_adj, 2)


    u_ind = list(range(num_users))
    user_user_graph[[u_ind, u_ind]] = 0.
    i_ind = list(range(num_items))
    item_item_graph[[i_ind, i_ind]] = 0.
    user_user_social_graph[[u_ind, u_ind]] = 0.

    def get_thr_from_rate(T, rate, val=None):
        node_num = T.shape[0]
        interact_num = node_num ** 2
        use_interact_num = int(interact_num * rate)

        exist_interacts = np.asarray(T[T>0.])[0]
        if(use_interact_num >= len(exist_interacts)):
            return 0.
        else:
            sorted_exist_interacts = sorted(exist_interacts, reverse=True)
            return sorted_exist_interacts[use_interact_num]

    def convert(T, rate, th=None, shape=None):
        # threshold = 0.02
        if th is None:
            threshold = get_thr_from_rate(T, rate)
        else:
            threshold = th
        # T[T>threshold] = 1.0
        ft = T > threshold
        n_T = csr_matrix((np.asarray([1.] * len(ft.data)), ft.indices, ft.indptr), shape=shape)
        # T[T<threshold] = 0.0
        return n_T

    user_user_graph = convert(user_user_graph, 0.007, shape=(num_users, num_users))
    item_item_graph = convert(item_item_graph, 0.003, shape=(num_items, num_items))
    user_user_social_graph = convert(user_user_social_graph, 0.001, shape=(num_users, num_users))

    user_user_graph.eliminate_zeros()
    item_item_graph.eliminate_zeros()
    user_user_social_graph.eliminate_zeros()

    save_graph(user_user_graph, "./tmp/5_user_user_graph.pkl")
    save_graph(item_item_graph, "./tmp/5_item_item_graph.pkl")
    save_graph(user_user_social_graph, "./tmp/5_user_user_social_graph.pkl")


# build_user_item_graph()

def convert_sp_to_one_hot_vect(sgraph):
    u_degrees = tf.sparse.reduce_sum(sgraph, axis=1) / 2
    u_degrees = tf.cast(u_degrees, tf.int64)
    u_degrees = tf.where(u_degrees <= 3, u_degrees, 4)

    degrees_one_hot_vec = tf.one_hot(u_degrees, depth=5, dtype=tf.int32)

    return degrees_one_hot_vec



def get_user_item_raw_adj():
    user_item_indics = []
    user_item_values = []
    uiv = []
    user_list = sorted(list(train_user_items_dict.keys()))
    for user in user_list:
        items = sorted(train_user_items_dict[user])
        l = len(items)
        for item in items:
            user_item_indics.append([user, item])
            user_item_values.append(1./(l + 1))
            uiv.append(1)

    uisg = tf.SparseTensor(
        user_item_indics,
        uiv,
        [num_users, num_items],
    )

    user_degrees_one_hot_vec = convert_sp_to_one_hot_vect(uisg)
    item_degrees_one_hot_vec = convert_sp_to_one_hot_vect(tf.sparse.transpose(uisg))

    user_item_sparse_graph = tf.SparseTensor(
        user_item_indics,
        user_item_values,
        [num_users, num_items],
    )

    return user_item_sparse_graph, tf.sparse.transpose(user_item_sparse_graph), user_degrees_one_hot_vec, item_degrees_one_hot_vec

# get_user_item_raw_adj()


def build_mixed_user_item_graph():
    row_col = list(zip(*train_user_item_edges))
    data = np.asarray([1.] * len(train_user_item_edges)).astype(np.float32)
    user_item_adj = csr_matrix((data, (row_col[0], row_col[1])), shape=(num_users, num_items))
    index = np.arange(num_users)
    np.random.shuffle(index)


    item_user_adj = csr_matrix((data, (row_col[1], row_col[0])), shape=(num_items, num_users))

    user_user_social_adj = csr_matrix((np.asarray([1.] * len(user_user_edges)).astype(np.float32),
                                       list(zip(*user_user_edges))), shape=(num_users, num_users))

    norm_user_item_adj = norm_adj_rows(user_item_adj)
    norm_item_user_adj = norm_adj_rows(item_user_adj)
    norm_user_user_social_adj = norm_adj_rows(user_user_social_adj)

    #05_ = 322(0.005,0.001,0.001), alpha = 0.2
    user_user_graph = surf(norm_user_item_adj, norm_item_user_adj, 3)
    item_item_graph = surf(norm_item_user_adj, norm_user_item_adj, 2)
    user_user_social_graph = social_surf(norm_user_user_social_adj, 2)


    u_ind = list(range(num_users))
    user_user_graph[[u_ind, u_ind]] = 0.
    i_ind = list(range(num_items))
    item_item_graph[[i_ind, i_ind]] = 0.
    user_user_social_graph[[u_ind, u_ind]] = 0.

    def get_thr_from_rate(T, rate):
        node_num = T.shape[0]
        interact_num = node_num ** 2
        use_interact_num = int(interact_num * rate)

        exist_interacts = np.asarray(T[T>0.])[0]
        if(use_interact_num >= len(exist_interacts)):
            return 0.
        else:
            sorted_exist_interacts = sorted(exist_interacts, reverse=True)
            return sorted_exist_interacts[use_interact_num]

    def convert(T, rate):
        # threshold = 0.02
        threshold = get_thr_from_rate(T, rate)
        T[T > threshold] = 1.0
        T[T < threshold] = 0.0
        return T

    user_user_graph = convert(user_user_graph, 0.001)
    item_item_graph = convert(item_item_graph, 0.0008)
    user_user_social_graph = convert(user_user_social_graph, 0.001)

    user_user_graph.eliminate_zeros()
    item_item_graph.eliminate_zeros()
    user_user_social_graph.eliminate_zeros()

    save_graph(user_user_graph, "./tmp/04_user_user_graph.pkl")
    save_graph(item_item_graph, "./tmp/04_item_item_graph.pkl")
    save_graph(user_user_social_graph, "./tmp/04_user_user_social_graph.pkl")

# build_mixed_user_item_graph()

# 合并
# def surf_u_metapaths(a, b, s, epoches):
#     P0 = sp.eye(a.shape[0])
#     alpha = 0.15
#     P = P0
#     PS = P0
#     if epoches > 1:
#         P = P @ a @ b * (1-alpha) + P0 * alpha
#         M = P
#         PS = PS @ s * (1-alpha) + P0 * alpha
#         M += PS
#         for _ in tqdm(range(epoches-1)):
#             P = P @ a @ b * (1-alpha) + P0 * alpha
#             M += P
#             PS = PS @ s * (1-alpha) + P0 * alpha
#             M += PS
#         return M
#     else:
#         P = P @ a @ b * (1-alpha) + P0 * alpha
#         M = P
#         PS = PS @ s * (1-alpha) + P0 * alpha
#         M += PS
#         return M
#
#
# def surf_i_metapaths(a, b, s, epoches):
#     P0 = sp.eye(a.shape[0])
#     alpha = 0.15
#     P = P0
#     PS = P0
#     if epoches > 1:
#         P = P @ a @ b * (1-alpha) + P0 * alpha
#         M = P
#         PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
#         M += PS
#         for _ in tqdm(range(epoches-1)):
#             P = P @ a @ b * (1-alpha) + P0 * alpha
#             M += P
#             PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
#             M += PS
#         return M
#     else:
#         P = P @ a @ b * (1-alpha) + P0 * alpha
#         M = P
#         PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
#         M += PS
#         return M
#
#
# def build_metapaths_user_item_graph():
#     row_col = list(zip(*train_user_item_edges))
#     data = np.asarray([1.] * len(train_user_item_edges)).astype(np.float32)
#     user_item_adj = csr_matrix((data, (row_col[0], row_col[1])), shape=(num_users, num_items))
#     item_user_adj = csr_matrix((data, (row_col[1], row_col[0])), shape=(num_items, num_users))
#
#     user_user_social_adj = csr_matrix((np.asarray([1.] * len(user_user_edges)).astype(np.float32),
#                                        list(zip(*user_user_edges))), shape=(num_users, num_users))
#
#     norm_user_item_adj = norm_adj_rows(user_item_adj)
#     norm_item_user_adj = norm_adj_rows(item_user_adj)
#     norm_user_user_social_adj = norm_adj_rows(user_user_social_adj)
#
#     user_user_graph = surf_u_metapaths(norm_user_item_adj, norm_item_user_adj, norm_user_user_social_adj, 3)
#     item_item_graph = surf_i_metapaths(norm_item_user_adj, norm_user_item_adj, norm_user_user_social_adj, 2)
#     user_user_social_graph = social_surf(norm_user_user_social_adj, 1)
#
#     u_ind = list(range(num_users))
#     user_user_graph[[u_ind, u_ind]] = 0.
#     i_ind = list(range(num_items))
#     item_item_graph[[i_ind, i_ind]] = 0.
#     user_user_social_graph[[u_ind, u_ind]] = 0.
#
#     def get_thr_from_rate(T, rate):
#         node_num = T.shape[0]
#         interact_num = node_num ** 2
#         use_interact_num = int(interact_num * rate)
#
#         exist_interacts = np.asarray(T[T>0.])[0]
#         if(use_interact_num >= len(exist_interacts)):
#             return 0.
#         else:
#             sorted_exist_interacts = sorted(exist_interacts, reverse=True)
#             return sorted_exist_interacts[use_interact_num]
#
#     def convert(T, rate):
#         # threshold = 0.02
#         threshold = get_thr_from_rate(T, rate)
#         T[T>threshold] = 1.0
#         T[T<threshold] = 0.0
#         return T
#
#     user_user_graph = convert(user_user_graph, 0.005)
#     item_item_graph = convert(item_item_graph, 0.001)
#     user_user_social_graph = convert(user_user_social_graph, 0.001)
#
#     user_user_graph.eliminate_zeros()
#     item_item_graph.eliminate_zeros()
#     user_user_social_graph.eliminate_zeros()
#
#     save_graph(user_user_graph, "./tmp/usu_user_user_graph.pkl")
#     save_graph(item_item_graph, "./tmp/iusui_item_item_graph.pkl")
#     save_graph(user_user_social_graph, "./tmp/user_user_social_graph.pkl")

# 分开-组合
def surf_u_metapaths(a, b, s, epoches):
    P0 = sp.eye(a.shape[0])
    alpha = 0.15
    P = P0
    PS = P0
    if epoches > 1:
        # P = P @ a @ b * (1-alpha) + P0 * alpha
        # M = P
        PS = PS @ s * (1-alpha) + P0 * alpha
        MS = PS
        for _ in tqdm(range(epoches-1)):
            # P = P @ a @ b * (1-alpha) + P0 * alpha
            # M += P
            PS = PS @ s * (1-alpha) + P0 * alpha
            MS += PS
        return MS
    else:
        # P = P @ a @ b * (1-alpha) + P0 * alpha
        # M = P
        PS = PS @ s * (1-alpha) + P0 * alpha
        MS = PS
        return MS


def surf_i_metapaths(a, b, s, epoches):
    P0 = sp.eye(a.shape[0])
    alpha = 0.15
    P = P0
    PS = P0
    if epoches > 1:
        # P = P @ a @ b * (1-alpha) + P0 * alpha
        # M = P
        PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
        MS = PS
        for _ in tqdm(range(epoches-1)):
            # P = P @ a @ b * (1-alpha) + P0 * alpha
            # M += P
            PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
            MS += PS
        return MS
    else:
        # P = P @ a @ b * (1-alpha) + P0 * alpha
        # M = P
        PS = PS @ a @ s @ b * (1-alpha) + P0 * alpha
        MS = PS
        return MS


#metapaths = ["uis", "issu"]
def build_metapath_graphs(u, i ,s, metapaths):
    metapath_graph_dict = {}

    for mp in metapaths:
        mps = list(mp)
        p = eval(mps[0])
        for j in range(len(mps)-1):
            p = p @ eval(mps[j + 1])
        metapath_graph_dict[mp] = p

    return metapath_graph_dict


def surf_metapaths(num, metapath_graph_dict, epoches):
    P0 = sp.eye(num)
    alpha = 0.15
    # P = P0
    tmp = {}
    surf_metapath_graph_dict = {}
    for mp in metapath_graph_dict.keys():
        print(mp)
        P = P0 @ metapath_graph_dict[mp] * (1-alpha) + P0 * alpha
        tmp[mp] = P
        surf_metapath_graph_dict[mp] = P

    if epoches > 1:
        for _ in tqdm(range(epoches-1)):
            for mp in metapath_graph_dict.keys():
                print(mp)
                P = tmp[mp] @ metapath_graph_dict[mp] * (1-alpha) + P0 * alpha
                tmp[mp] = P
                surf_metapath_graph_dict[mp] += P

    return surf_metapath_graph_dict

def get_thr_from_rate(T, rate):
    node_num = T.shape[0]
    interact_num = node_num ** 2
    use_interact_num = int(interact_num * rate)

    exist_interacts = np.asarray(T[T>0.])[0]
    if(use_interact_num >= len(exist_interacts)):
        return 0.
    else:
        sorted_exist_interacts = sorted(exist_interacts, reverse=True)
        return sorted_exist_interacts[use_interact_num]


def convert(T, rate):
    # threshold = 0.02
    threshold = get_thr_from_rate(T, rate)
    T[T>threshold] = 1.0
    T[T<threshold] = 0.0
    return T


def build_metapaths_user_item_graph():
    row_col = list(zip(*train_user_item_edges))
    data = np.asarray([1.] * len(train_user_item_edges)).astype(np.float32)
    user_item_adj = csr_matrix((data, (row_col[0], row_col[1])), shape=(num_users, num_items))
    item_user_adj = csr_matrix((data, (row_col[1], row_col[0])), shape=(num_items, num_users))

    user_user_social_adj = csr_matrix((np.asarray([1.] * len(user_user_edges)).astype(np.float32),
                                       list(zip(*user_user_edges))), shape=(num_users, num_users))

    u = norm_user_item_adj = norm_adj_rows(user_item_adj)
    i = norm_item_user_adj = norm_adj_rows(item_user_adj)
    s = norm_user_user_social_adj = norm_adj_rows(user_user_social_adj)

    # ionly(131(0.005,0.001,0.001)) uonly(411(0.005,0.001,0.001))

    uus = build_metapath_graphs(u, i, s, ["s"])
    uus_surf_metapath_graph_dict = surf_metapaths(num_users, uus, 4)
    u_ind = list(range(num_users))
    for mp in uus_surf_metapath_graph_dict.keys():
        uus_surf_metapath_graph_dict[mp][[u_ind, u_ind]] = 0.

    for mp in uus_surf_metapath_graph_dict.keys():
        uus_surf_metapath_graph_dict[mp].eliminate_zeros()
        uus_surf_metapath_graph_dict[mp] = convert(uus_surf_metapath_graph_dict[mp], 0.001)
        uus_surf_metapath_graph_dict[mp].eliminate_zeros()

    ss = uus_surf_metapath_graph_dict["s"]

    print("building ss is finished ")

    user_metapath_graph_dict = build_metapath_graphs(u, i, s, ["ui"])
    item_metapath_graph_dict = build_metapath_graphs(u, i, ss, ["iu", "isu"])
    social_metapath_graph_dict = build_metapath_graphs(u, i, s, ["s"])

    user_surf_metapath_graph_dict = surf_metapaths(num_users, user_metapath_graph_dict, 3)
    item_surf_metapath_graph_dict = surf_metapaths(num_items, item_metapath_graph_dict, 2)
    social_surf_metapath_graph_dict = surf_metapaths(num_users, social_metapath_graph_dict, 1)

    u_ind = list(range(num_users))
    for mp in user_surf_metapath_graph_dict.keys():
        user_surf_metapath_graph_dict[mp][[u_ind, u_ind]] = 0.
    i_ind = list(range(num_items))
    for mp in item_surf_metapath_graph_dict.keys():
        item_surf_metapath_graph_dict[mp][[i_ind, i_ind]] = 0.
    for mp in social_surf_metapath_graph_dict.keys():
        social_surf_metapath_graph_dict[mp][[u_ind, u_ind]] = 0.

    for mp in user_surf_metapath_graph_dict.keys():
        user_surf_metapath_graph_dict[mp].eliminate_zeros()
        user_surf_metapath_graph_dict[mp] = convert(user_surf_metapath_graph_dict[mp], 0.005)
        user_surf_metapath_graph_dict[mp].eliminate_zeros()

    # for mp in item_surf_metapath_graph_dict.keys():
    #     item_surf_metapath_graph_dict[mp].eliminate_zeros()
    #     item_surf_metapath_graph_dict[mp] = convert(item_surf_metapath_graph_dict[mp], 0.001)
    #     item_surf_metapath_graph_dict[mp].eliminate_zeros()
    item_surf_metapath_graph_dict["iu"].eliminate_zeros()
    item_surf_metapath_graph_dict["iu"] = convert(item_surf_metapath_graph_dict["iu"], 0.001)
    item_surf_metapath_graph_dict["iu"].eliminate_zeros()
    item_surf_metapath_graph_dict["isu"].eliminate_zeros()
    item_surf_metapath_graph_dict["isu"] = convert(item_surf_metapath_graph_dict["isu"], 0.01)
    item_surf_metapath_graph_dict["isu"].eliminate_zeros()

    for mp in social_surf_metapath_graph_dict.keys():
        social_surf_metapath_graph_dict[mp].eliminate_zeros()
        social_surf_metapath_graph_dict[mp] = convert(social_surf_metapath_graph_dict[mp], 0.001)
        social_surf_metapath_graph_dict[mp].eliminate_zeros()

    user_user_graphs = list(user_surf_metapath_graph_dict.values())
    user_user_graph = user_user_graphs[0]
    for i in range(len(user_user_graphs) - 1):
        user_user_graph += user_user_graphs[i + 1]

    item_item_graphs = list(item_surf_metapath_graph_dict.values())
    item_item_graph = item_item_graphs[0]
    for i in range(len(item_item_graphs) - 1):
        item_item_graph += item_item_graphs[i + 1]

    user_user_social_graphs = list(social_surf_metapath_graph_dict.values())
    user_user_social_graph = user_user_social_graphs[0]
    for i in range(len(user_user_social_graphs) - 1):
        user_user_social_graph += user_user_social_graphs[i + 1]

    save_graph(user_user_graph, "./tmp/usu_user_user_graph.pkl")
    save_graph(item_item_graph, "./tmp/iusui_item_item_graph.pkl")
    save_graph(user_user_social_graph, "./tmp/user_user_social_graph.pkl")


# build_metapaths_user_item_graph()


def base_eval():
    row_col = list(zip(*train_user_item_edges))
    data = np.asarray([1.] * len(train_user_item_edges)).astype(np.float32)
    user_item_adj = csr_matrix((data, (row_col[0], row_col[1])), shape=(num_users, num_items))
    item_user_adj = csr_matrix((data, (row_col[1], row_col[0])), shape=(num_items, num_users))

    user_user_social_adj = csr_matrix((np.asarray([1.] * len(user_user_edges)).astype(np.float32),
                                       list(zip(*user_user_edges))), shape=(num_users, num_users))

    u = norm_user_item_adj = norm_adj_rows(user_item_adj)
    i = norm_item_user_adj = norm_adj_rows(item_user_adj)
    s = norm_user_user_social_adj = norm_adj_rows(user_user_social_adj)

def generate_social_sparse_graph():
    user_user_indics = []
    user_user_values = []
    user_list = sorted(list(user_users_dict.keys()))
    for user in user_list:
        friends = sorted(user_users_dict[user])
        l = len(friends)
        user_user_indics.append([user, user])
        user_user_values.append(1./(l + 1))
        for friend in friends:
            user_user_indics.append([user, friend])
            user_user_values.append(1./(l + 1))

    user_social_sparse_graph = tf.SparseTensor(
        user_user_indics,
        user_user_values,
        [num_users, num_users],
    )

    return user_social_sparse_graph


def generate_norm_social_sparse_graph():
    row = []
    col = []
    data = []
    user_list = sorted(list(user_users_dict.keys()))
    for user in user_list:
        friends = sorted(user_users_dict[user])
        for friend in friends:
            row.append(user)
            col.append(friend)
            data.append(1.)

    adj = csr_matrix((data, (row, col)), shape=(num_users, num_users))
    norm_adj = preprocess_adj(adj)
    print(norm_adj)

    user_social_sparse_graph = tf.SparseTensor(
        norm_adj[0],
        norm_adj[1].astype(np.float32),
        norm_adj[2],
    )

    return user_social_sparse_graph




def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)

# generate_norm_social_sparse_graph()

# def generate_socail_sparse_graph():
#     row = []
#     col = []
#     data = []
#     user_list = sorted(list(user_users_dict.keys()))
#     for user in user_list:
#         friends = sorted(user_users_dict[user])
#         l = len(friends)
#         for friend in friends:
#             row.append(user)
#             col.append(friend)
#             data.append(1./l)
#
#     user_user_sparse_graph = tf.SparseTensor(
#         user_user_indics,
#         user_user_values,
#         [num_users, num_users],
#     )
#
#     return user_user_sparse_graph

# def build_sparse_graphs():
#
#     user_user_sparse_graph = generate_social_neighbors_sparse_matrix(user_users_dict)
#
#
#
#
#     print(user_user_sparse_graph)


# build_sparse_graphs()



def generate_social_neighbors_sparse_matrix():
    user_user_indics = []
    user_user_values = []
    user_list = sorted(list(user_users_dict.keys()))
    for user in user_list:
        friends = sorted(user_users_dict[user])
        l = len(friends)
        for friend in friends:
            user_user_indics.append([user, friend])
            user_user_values.append(1./l)

    user_user_sparse_graph = tf.SparseTensor(
        user_user_indics,
        user_user_values,
        [num_users, num_users],
        )

    return user_user_sparse_graph

# def generate_user_items_sparse_matrix(user_items_dict):
#     user_items_indics = []
#     user_user_values = []
#     user_list = sorted(list(user_users_dict.keys()))
#     for user in user_list:
#         friends = sorted(user_users_dict[user])
#         l = len(friends)
#         for friend in friends:
#             user_user_indics.append([user, friend])
#             user_user_values.append(1./l)
#
#     user_user_sparse_graph = tf.SparseTensor(
#         user_user_indics,
#         user_user_values,
#         [num_users, num_users],
#     )
#
#     return user_user_sparse_graph

# for u in social_neighbors:
#     social_neighbors_dict[u] = sorted(social_neighbors[u])
#
#
# for user in user_list:
#     for friend in social_neighbors_dict[user]:
#         social_neighbors_indices_list.append([user, friend])
#         social_neighbors_values_list.append(1.0/len(social_neighbors_dict[user]))
# self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
# self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)


# def generateConsumedItemsSparseMatrix(self):
#     positive_data = self.positive_data
#     consumed_items_indices_list = []
#     consumed_items_values_list = []
#     consumed_items_dict = defaultdict(list)
#     for u in positive_data:
#         consumed_items_dict[u] = sorted(positive_data[u])
#     user_list = sorted(list(positive_data.keys()))
#     for u in user_list:
#         for i in consumed_items_dict[u]:
#             consumed_items_indices_list.append([u, i])
#             consumed_items_values_list.append(1.0/len(consumed_items_dict[u]))
#     self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
#     self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)