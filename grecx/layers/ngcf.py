# coding=utf-8

import tensorflow as tf
import tf_sparse as tfs
from grecx.layers.light_gcn import LightGCN


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

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):
        return LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)

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
        normed_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = normed_adj @ x
        h = self.gcn_dense(h) + self.interaction_dense(x * h)

        return h


class NGCF(tf.keras.Model):

    def __init__(self, k=3, dense_activation=tf.nn.leaky_relu, edge_drop_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ngcf_convs = [NGCFConv(dense_activation=dense_activation,edge_drop_rate=edge_drop_rate) for _ in range(k)]

        for i, ngcf_conv in enumerate(self.ngcf_convs):
            setattr(self, "ngcf_conv{}".format(i), ngcf_conv)


    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):
        return LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        self.ngcf_convs[0].build_cache_for_graph(graph, override=override)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h = x
        h_list = []

        for ngcf_conv in self.ngcf_convs:
            h = ngcf_conv([h, edge_index], training=training, cache=cache)
            h_list.append(h)

        h_list = [tf.nn.l2_normalize(h, axis=-1) for h in h_list]

        h = tf.concat([x] + h_list, axis=-1)

        return h
