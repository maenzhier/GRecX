# coding=utf-8

import tf_geometric as tfg
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tf_geometric import SparseAdj
import tf_sparse as tfs
from tf_sparse import sparse_diag_matmul, diag_sparse_matmul
from tf_geometric.utils.graph_utils import convert_edge_to_directed, add_self_loop_edge
from grecx.models.light_gcn import LightGCN


class NGCF(tf.keras.Model):

    def __init__(self, units, k=1, activations=tf.nn.leaky_relu, drop_rates=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.drop_rates = drop_rates

        self.units_list = units if hasattr(units, "__len__") else [units] * k
        self.k = len(self.units_list)
        self.activation_list = activations if hasattr(activations, "__len__") else [activations] * self.k
        self.drop_rate_list = drop_rates if hasattr(units, "__len__") else [drop_rates] * self.k

        assert len(self.units_list) == len(self.drop_rate_list) == len(self.activation_list) == self.k

        self.ngcf_layers = [NGCFConv(unit, activation, drop_rate) for unit, activation, drop_rate in
                            zip(self.units_list, self.activation_list, self.drop_rate_list)]

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h = x
        h_list = [h]

        for i in range(self.k):

            h = self.ngcf_layers[i]([edge_index, h], training=training, cache=cache)
            h_list.append(tf.nn.l2_normalize(h, axis=-1))

        h = tf.concat(h_list, axis=-1)

        return h


class NGCFConv(tf.keras.Model):
    def __init__(self, units, activation=tf.nn.leaky_relu, drop_rate=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.drop_rate = drop_rate

        self.dense1 = Dense(units, activation)
        self.dense2 = Dense(units, activation)
        self.dropout_layer = Dropout(drop_rate)

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
        x, edge_index = inputs[0], inputs[1]

        num_nodes = tfs.shape(x)[0]
        normed_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache)

        h = normed_adj @ x
        h = self.dense1(h) + self.dense2(x * h)
        
        h = self.dropout_layer(h, training=training)

        return h
