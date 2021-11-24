# coding=utf-8

import tensorflow as tf
import tf_sparse as tfs
from grecx.models.light_gcn import LightGCN


class NGCFConv(tf.keras.Model):
    """
    Each NGCF Convolutional Layer
    """

    def __init__(self, dense_activation=tf.nn.leaky_relu, edge_drop_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_activation = dense_activation
        self.gcn_dense = None
        self.interaction_dense = None
        self.edge_drop_rate = edge_drop_rate

    def build(self, input_shape):
        x_shape, _ = input_shape

        self.gcn_dense = tf.keras.layers.Dense(x_shape[1], activation=self.dense_activation)
        self.interaction_dense = tf.keras.layers.Dense(x_shape[1], activation=self.dense_activation)

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
        normed_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache)\
            .dropout(self.edge_drop_rate, training=training)

        h = normed_adj @ x
        h = self.gcn_dense(h) + self.interaction_dense(x * h)

        h = tf.nn.l2_normalize(h, axis=-1)

        return h



class NGCF(tf.keras.Model):

    def __init__(self, units, k=3, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.units = units
        self.ngcf_convs = [NGCFConv() for _ in range(k)]

        for i, ngcf_conv in enumerate(self.ngcf_convs):
            setattr(self, "ngcf_conv{}".format(i), ngcf_conv)

        self.activation = activation

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        self.ngcf_convs[0].norm_adj(graph, override=override)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h = x
        h_list = [h]

        for ngcf_conv in self.ngcf_convs:
            h = ngcf_conv([edge_index, h], training=training, cache=cache)
            h_list.append(h)

        h = tf.concat(h_list, axis=-1)

        if self.activation is not None:
            h = self.activation(h)

        return h
