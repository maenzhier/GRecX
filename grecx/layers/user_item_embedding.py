# coding=utf-8


import tensorflow as tf

from grecx.layers.embedding import Embedding


class UserItemEmbedding(tf.keras.Model):

    def __init__(self, num_users, num_items, embedding_size, drop_rate=0.0, global_dropout=False,
                 initializer=None,
                 embeddings_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.user_embedding_layer = Embedding(num_users, embedding_size=embedding_size, drop_rate=drop_rate,
                                              global_dropout=global_dropout,
                                              initializer=initializer, embeddings_regularizer=embeddings_regularizer,
                                              name="user")
        self.item_embedding_layer = Embedding(num_items, embedding_size=embedding_size, drop_rate=drop_rate,
                                              global_dropout=global_dropout,
                                              initializer=initializer, embeddings_regularizer=embeddings_regularizer,
                                              name="item")

    @property
    def user_embeddings(self):
        return self.user_embedding_layer.embeddings

    @property
    def item_embeddings(self):
        return self.item_embedding_layer.embeddings

    def call(self, inputs, training=None, mask=None):
        """
        Embed users and items.

        :param inputs: [nested_user_indices, nested_item_indices]. The output will have the same structure with the input.
            For example:
            - [user_indices, item_indices]
            - [user_indices, [item_indices, neg_item_indices]]

        :param training:
        :param mask:
        :param cache:
        :return:
        """
        user_indices, item_indices = inputs
        embedded_users = self.user_embedding_layer(user_indices, training=training)
        embedded_items = self.item_embedding_layer(item_indices, training=training)
        return embedded_users, embedded_items
