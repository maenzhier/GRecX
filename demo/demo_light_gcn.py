# coding = utf8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
from time import time
from grecx.evaluation.ranking_faiss import evaluate_mean_global_ndcg_score_with_faiss
import grecx as grx
from grecx.datasets import LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils

np.set_printoptions(precision=4)

#lr = 1e-3
# l2 = 1e-4
data_dict = LightGCNYelpDataset().load_data()
# data_dict = LightGCNGowallaDataset().load_data()
# data_dict = LightGCNAmazonbookDataset().load_data()

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
lr = 1e-2
# l2 = 1e-4 'ndcg@20': 0.1481
# l2 = 5e-5 0.146
# l2 = 2e-4 0.1475
# l2 = 9e-5 'ndcg@20': 0.1485
l2 = 2e-4
k = 3
edge_drop_rate = 0.15
epoches = 2700
batch_size = 8000

# initializer = tf.random_normal_initializer(stddev=0.01)

virtual_graph = tfg.Graph(
    x=tf.Variable(
        # initializer([int(num_users + num_items), int(embedding_size)]),
        tf.random.truncated_normal([num_users + num_items, embedding_size], stddev=1/np.sqrt(embedding_size)),
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
def train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices):
    with tf.GradientTape() as tape:
        user_h, item_h = forward(virtual_graph, training=True)

        embedded_users = tf.gather(user_h, batch_user_indices)
        embedded_items = tf.gather(item_h, batch_item_indices)
        embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

        pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
        neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)

        # pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=pos_logits,
        #     labels=tf.ones_like(pos_logits)
        # )
        # neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=neg_logits,
        #     labels=tf.zeros_like(neg_logits)
        # )
        #
        # mf_losses = pos_losses + neg_losses

        mf_losses = tf.nn.softplus(-(pos_logits - neg_logits))

        l2_vars = [var for var in tape.watched_variables() if "embedding" in var.name]
        l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
        l2_loss = tf.add_n(l2_losses)

        loss = tf.reduce_sum(mf_losses) + l2_loss * l2

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss, mf_losses, l2_loss


for epoch in range(1, epoches + 1):
    if epoch % 20 == 0:
        user_h, item_h = forward(virtual_graph, training=False)
        print("\nEvaluation before epoch {}".format(epoch))
        mean_ndcg_dict_faiss = evaluate_mean_global_ndcg_score_with_faiss(test_user_items_dict, train_user_items_dict,
                                                                          user_h, item_h)
        print(mean_ndcg_dict_faiss)
        print()

    step_losses = []
    step_mf_losses_list = []
    step_l2_losses = []

    start_time = time()

    for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]
        batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)

        loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices)

        step_losses.append(loss.numpy())
        step_mf_losses_list.append(mf_losses.numpy())
        step_l2_losses.append(l2_loss.numpy())

    end_time = time()


    if optimizer.learning_rate.numpy() > 1e-5:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.995)
        lr_status = "update lr => {:.4f}".format(optimizer.learning_rate.numpy())
    else:
        lr_status = "current lr => {:.4f}".format(optimizer.learning_rate.numpy())

    print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\t{}\ttime = {:.4f}s".format(
        epoch, np.mean(step_losses), np.mean(np.concatenate(step_mf_losses_list, axis=0)),
        np.mean(step_l2_losses), lr_status, end_time-start_time))

    if epoch == 1:
        print("the first epoch may take a long time to compile tf.function")
