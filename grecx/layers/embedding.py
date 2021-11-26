# coding=utf-8


import tensorflow as tf
import numpy as np


class Embedding(tf.keras.Model):

    def __init__(self, num_embeddings, embedding_size, drop_rate=0.0, global_dropout=False,
                 initializer=None,
                 embeddings_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_dropout = global_dropout

        if initializer is None:
            initializer = tf.keras.initializers.Constant(value=tf.random.truncated_normal([num_embeddings, embedding_size], stddev=1.0/np.sqrt(embedding_size)))

        self.embeddings = self.add_weight("embeddings", shape=[num_embeddings, embedding_size], initializer=initializer, regularizer=embeddings_regularizer)
        if drop_rate is not None and drop_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(drop_rate)
        else:
            self.dropout = None

    def call(self, inputs, training=None, mask=None, cache=None):


        embeddings = self.embeddings

        if self.global_dropout and self.dropout is not None:
            embeddings = self.dropout(embeddings, training=training)

        def gather_func(indices):
            return tf.gather(embeddings, indices)

        output = tf.nest.map_structure(gather_func, inputs)

        if not self.global_dropout and self.dropout is not None:
            def dropout_func(embedded):
                return self.dropout(embedded, training=training)
            output = tf.nest.map_structure(dropout_func, output)

        return output

    # def call(self, inputs, training=None, mask=None, cache=None):
    #
    #     inputs_is_list = isinstance(inputs, list)
    #
    #     embeddings = self.embeddings
    #
    #     if self.global_dropout and self.dropout is not None:
    #         embeddings = self.dropout(embeddings, training=training)
    #
    #
    #
    #     if inputs_is_list:
    #         embedded_list = [tf.gather(embeddings, indices) for indices in inputs]
    #     else:
    #         embedded = tf.gather(embeddings, inputs)
    #
    #     if not self.global_dropout and self.dropout is not None:
    #         if inputs_is_list:
    #             output = [self.dropout(embedded, training=training) for embedded in embedded_list]
    #         else:
    #             output = self.dropout(embedded, training=training)
    #
    #     return output
    #
