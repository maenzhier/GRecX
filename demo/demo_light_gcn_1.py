# coding = utf8

import tensorflow as tf
import os
import numpy as np

from grecx.evaluation.ranking import evaluate_mean_global_ndcg_score
import grecx as grx

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from grecx.datasets import LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils

data_dict = LightGCNYelpDataset().load_data()
num_users = data_dict["num_users"]
num_items = data_dict["num_items"]
user_item_edges = data_dict["user_item_edges"]
train_index = data_dict["train_index"]
train_user_items_dict = data_dict["train_user_items_dict"]
test_user_items_dict = data_dict["test_user_items_dict"]

train_user_item_edges = user_item_edges[train_index]
train_user_item_edge_index = train_user_item_edges.transpose()

embedding_size = 64
# drop_rate = 0.6
lr = 5e-3
# lr = 1e-3
l2 = 1e-3
# l2 = 1e-4
k = 3
edge_drop_rate = 0.1
epoches = 2700
batch_size = 5000

initializer = tf.random_normal_initializer(stddev=0.01)

virtual_graph = tfg.Graph(
    x=tf.Variable(
        initializer([int(num_users + num_items), int(embedding_size)]),
        name="virtual_embeddings"
    ),
    edge_index=grx.models.LightGCN.build_virtual_edge_index(train_user_item_edge_index, num_users)
)

model = grx.models.LightGCN(k=k, edge_drop_rate=edge_drop_rate)
model.build_cache_for_graph(virtual_graph)


@tf_utils.function
def forward(virtual_graph, training=False):
    virtual_h = model([virtual_graph.x, virtual_graph.edge_index], training=training, cache=virtual_graph.cache)
    user_h = virtual_h[:num_users]
    item_h = virtual_h[num_users:]
    return user_h, item_h


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

@tf_utils.function
def mf_score_func(batch_user_indices, batch_item_indices):
    user_h, item_h = forward(virtual_graph, training=False)
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
            user_h, item_h = forward(virtual_graph, training=True)

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

            # l2_vars = [var for var in tape.watched_variables() if "embedding" in var.name]
            # l2_vars.append(model.user_embeddings)
            # l2_vars.append(model.item_embeddings)
            # l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
            # l2_loss = tf.add_n(l2_losses)

            mf_l2_vars = [embedded_users, embedded_items, embedded_neg_items]
            mf_l2_losses = [tf.nn.l2_loss(var) for var in mf_l2_vars]
            mf_l2_loss = tf.add_n(mf_l2_losses) / batch_size

            loss = tf.reduce_sum(losses) + mf_l2_loss * l2

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if epoch % 20 == 0 and step % 1000 == 0:

            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))
            if step == 0:
                mean_ndcg_dict = evaluate_mean_global_ndcg_score(test_user_items_dict, train_user_items_dict, num_items, mf_score_func)
                print(mean_ndcg_dict)
