# coding = utf8

import tensorflow as tf
import os
import numpy as np

from grecx.evaluation.ranking import evaluate_mean_candidate_ndcg_score
import grecx as grx

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from grecx.datasets import LightGCNYelp, LightGCNGowalla, LightGCNAmazonbook
import tf_geometric as tfg
from tf_geometric.utils import tf_utils

num_users, num_items, n_train, n_test = LightGCNAmazonbook().load_data()


embedding_size = 64
drop_rate = 0.6
lr = 3e-3
l2 = 3e-2

epoches = 2700
batch_size = 5000


virtual_graph = tfg.Graph(
    x=tf.Variable(tf.random.truncated_normal([num_users + num_items, embedding_size], stddev=1/np.sqrt(embedding_size))),
    edge_index=grx.models.LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)
)

model = grx.models.LightGCN()
model.build_cache_for_graph(virtual_graph)


@tf_utils.function
def forward(virtual_graph, training=False):
    virtual_h = model([virtual_graph.x, virtual_graph.edge_index], training=training, cache=virtual_graph.cache)
    embedded_users = virtual_h[:num_users]
    embedded_items = virtual_h[num_users:]
    return embedded_users, embedded_items


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

@tf_utils.function
def mf_score_func(batch_user_indices, batch_item_indices):
    embedded_users, embedded_items = forward(virtual_graph, training=False)
    embedded_users = tf.gather(user_h, batch_user_indices)
    embedded_items = tf.gather(item_h, batch_item_indices)
    logits = embedded_users @ tf.transpose(embedded_items, [1, 0])
    return logits


for epoch in range(0, epoches):
    print("epoch: ", epoch)
    for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]
        batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)

        with tf.GradientTape() as tape:
            user_h, item_h = forward(training=True)

            embedded_users = tf.gather(user_h, batch_user_indices)
            embedded_items = tf.gather(item_h, batch_item_indices)
            embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

            pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
            neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)

            pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_logits,
                labels=tf.ones_like(pos_logits)
            )
            neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_logits,
                labels=tf.zeros_like(neg_logits)
            )

            losses = pos_losses + neg_losses

            l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
            l2_vars.append(model.user_embeddings)
            l2_vars.append(model.item_embeddings)
            l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
            l2_loss = tf.add_n(l2_losses)

            mf_l2_vars = [user_h, item_h]
            mf_l2_losses = [tf.nn.l2_loss(var) for var in mf_l2_vars]
            mf_l2_loss = tf.add_n(mf_l2_losses)

            loss = tf.reduce_sum(losses) + l2_loss * l2 + 1e-3 * mf_l2_loss

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if epoch % 20 == 0 and step % 1000 == 0:

            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))
            if step == 0:
                evaluate_mean_candidate_ndcg_score(test_user_items_dict, user_neg_items_dict, mf_score_func)

