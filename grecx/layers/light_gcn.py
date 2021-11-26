# coding=utf-8

import tf_geometric as tfg
import tensorflow as tf
from tf_geometric import SparseAdj
from tf_sparse import sparse_diag_matmul, diag_sparse_matmul
from tf_geometric.utils.graph_utils import convert_edge_to_directed, add_self_loop_edge


class LightGCN(tf.keras.Model):
    CACHE_KEY = "light_gcn_normed_adj"

    def __init__(self, k, edge_drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k = k
        self.edge_drop_rate = edge_drop_rate

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):

        user_index, item_index = user_item_edge_index[0], user_item_edge_index[1]

        if num_users is None:
            num_users = tf.reduce_max(user_index) + 1

        virtual_item_index = item_index + num_users
        virtual_edge_index = tf.stack([user_index, virtual_item_index], axis=0)
        virtual_edge_index, _ = convert_edge_to_directed(virtual_edge_index, merge_modes="sum")
        return virtual_edge_index

    @classmethod
    def norm_adj(cls, edge_index, num_nodes, cache=None):

        if cache is not None:
            cached_data = cache.get(LightGCN.CACHE_KEY, None)
            if cached_data is not None:
                return SparseAdj(cached_data[0], cached_data[1], cached_data[2])
            else:
                if not tf.executing_eagerly():
                    raise Exception("If you want to use cache inside a tf.function, you should manually build the cache before calling the tf.function")

        adj = tfg.SparseAdj(edge_index, shape=[num_nodes, num_nodes])

        deg = adj.segment_sum(axis=-1)
        deg_inv_sqrt = tf.pow(deg, -0.5)
        deg_inv_sqrt = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
            tf.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt
        )

        # (D^(-1/2)A)D^(-1/2)
        normed_adj = sparse_diag_matmul(diag_sparse_matmul(deg_inv_sqrt, adj), deg_inv_sqrt)

        if cache is not None:
            cache[LightGCN.CACHE_KEY] = normed_adj.index.numpy(), normed_adj.value.numpy(), normed_adj._shape.numpy()

        return normed_adj

    @classmethod
    def norm_adj_bak(cls, edge_index, num_nodes, cache=None):

        if cache is not None:
            cached_data = cache.get(LightGCN.CACHE_KEY, None)
            if cached_data is not None:
                return cached_data

        adj = tfg.SparseAdj(edge_index, shape=[num_nodes, num_nodes])

        deg = adj.segment_sum(axis=-1)
        deg_inv_sqrt = tf.pow(deg, -0.5)
        deg_inv_sqrt = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
            tf.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt
        )

        # (D^(-1/2)A)D^(-1/2)
        normed_adj = sparse_diag_matmul(diag_sparse_matmul(deg_inv_sqrt, adj), deg_inv_sqrt)

        if cache is not None:
            cache[LightGCN.CACHE_KEY] = normed_adj

        return normed_adj

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        if override:
            graph.cache[LightGCN.CACHE_KEY] = None
        LightGCN.norm_adj(graph.edge_index, graph.num_nodes, cache=graph.cache)

    def call(self, inputs, training=None, mask=None, cache=None):

        x, edge_index = inputs
        num_nodes = tf.shape(x)[0]
        normed_adj = self.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = x
        h_list = [h]

        for _ in range(self.k):
            h = normed_adj @ h
            h_list.append(h)

        h_matrix = tf.stack(h_list, axis=0)
        h = tf.reduce_mean(h_matrix, axis=0)

        return h
