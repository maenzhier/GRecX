# coding=utf-8

import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from tf_geometric.nn import gcn_norm_adj
from tf_geometric.utils.graph_utils import convert_edge_to_directed, add_self_loop_edge


class LightGCN(tf.keras.Model):

    CACHE_KEY = "light_gcn_normed_adj"

    def __init__(self, k, edge_drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k = k
        self.edge_drop_rate = edge_drop_rate

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users):

        user_index, item_index = user_item_edge_index[0], user_item_edge_index[1]
        virtual_item_index = item_index + num_users
        virtual_edge_index = tf.stack([user_index, virtual_item_index], axis=0)
        virtual_edge_index, _ = convert_edge_to_directed(virtual_edge_index, merge_modes="sum")
        return virtual_edge_index

    @classmethod
    def norm_adj(cls, edge_index, num_nodes, cache=None):

        if cache is not None:
            cached_data = cache.get(LightGCN.CACHE_KEY, None)
            if cached_data is not None:
                return cached_data

        adj = tfg.SparseAdj(edge_index, shape=[num_nodes, num_nodes])
        adj = adj.add_self_loop()

        deg = tf.math.unsorted_segment_sum(adj.edge_weight, adj.edge_index[0], num_nodes)
        inv_deg = tf.pow(deg + 1e-8, -1)
        transition = adj.rmatmul_diag(inv_deg)

        virtual_gcn_normed_user_item_adj = gcn_norm_adj(transition)

        if cache is not None:
            cache[LightGCN.CACHE_KEY] = virtual_gcn_normed_user_item_adj

        return virtual_gcn_normed_user_item_adj

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
        normed_adj = self.norm_adj(edge_index, num_nodes=num_nodes, cache=cache)\
            .dropout(self.edge_drop_rate, training=training)

        h = x
        h_list = [h]

        for _ in range(self.k):
            h = normed_adj @ h
            h_list.append(h)

        h_matrix = tf.stack(h_list, axis=0)
        h = tf.reduce_mean(h_matrix, axis=0)

        return h




