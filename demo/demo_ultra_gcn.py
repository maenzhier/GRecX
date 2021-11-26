# coding = utf8

import tensorflow as tf
import os
import numpy as np
from time import time

from grecx.datasets.light_gcn_dataset import LightGCNDataset, LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
from grecx.layers import UserItemEmbedding

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from grecx.evaluation.ranking import evaluate_mean_global_metrics
from tf_geometric.utils import tf_utils
import scipy.sparse as sp

dataset = "light_gcn_yelp"  # "light_gcn_yelp" | "light_gcn_gowalla" | "light_gcn_amazon-book"

data_dict = LightGCNDataset(dataset).load_data()

num_users = data_dict["num_users"]
num_items = data_dict["num_items"]
user_item_edges = data_dict["user_item_edges"]
train_index = data_dict["train_index"]
train_user_items_dict = data_dict["train_user_items_dict"]
test_user_items_dict = data_dict["test_user_items_dict"]

train_user_item_edges = user_item_edges[train_index]
train_user_item_edge_index = train_user_item_edges.transpose()

if dataset == "light_gcn_yelp":
    drop_rate = 0.15
    lr = 1e-2
    # l2 = 5e-7
    l2 = 5e-4
    # l2 = 1e-4
    # l2 = 1e-4 0.0475
else:
    drop_rate = 0.15
    lr = 1e-2
    # l2 = 5e-7
    l2 = 1e-4
    # l2 = 1e-4 0.0475

embedding_size = 64
epoches = 2700
batch_size = 5000


def build_weight_matrix(user_item_edges):
    user_item_edges = np.array(user_item_edges)

    row = user_item_edges[:, 0]
    col = user_item_edges[:, 1]

    adj = sp.csr_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=[num_users, num_items])

    user_deg = np.array(adj.sum(axis=1)).flatten()  # np.sum(adj, axis=1).reshape(-1)
    item_deg = np.array(adj.sum(axis=0)).flatten()  # np.sum(adj, axis=0).reshape(-1)

    beta_user_deg = (np.sqrt(user_deg + 1) / user_deg).reshape(-1, 1)
    beta_item_deg = (1 / np.sqrt(item_deg + 1)).reshape(1, -1)

    constraint_mat = beta_user_deg @ beta_item_deg  # n_user * m_item
    constraint_mat = np.array(constraint_mat, dtype=np.float32)

    return constraint_mat

weight_matrix = build_weight_matrix(train_user_item_edges)
weight_matrix = tf.Variable(weight_matrix, trainable=False)


"""
Without the loss function, a UltraGCN is equivalent to a MF model, which only relies on embeddings of users and items.
"""
embedding_model = UserItemEmbedding(num_users, num_items, embedding_size, drop_rate=drop_rate, global_dropout=True)



optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf_utils.function(experimental_compile=True)
def train_step(batch_user_indices, batch_item_indices):
    batch_neg_item_indices = tf.random.uniform(shape=[tf.shape(batch_user_indices)[0], 800], maxval=num_items, dtype=tf.int64)

    batch_user_weights = tf.gather(weight_matrix, batch_user_indices)
    pos_weights = tf.gather(batch_user_weights, batch_item_indices, batch_dims=1)
    neg_weights = tf.gather(batch_user_weights, batch_neg_item_indices, batch_dims=1)

    with tf.GradientTape() as tape:
        embedded_users, [embedded_items, embedded_neg_items] = \
            embedding_model([batch_user_indices, [batch_item_indices, batch_neg_item_indices]], training=True)


        pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
        neg_logits = tf.reduce_sum((tf.expand_dims(embedded_users, axis=1) * embedded_neg_items), axis=-1)

        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pos_logits,
            labels=tf.ones_like(pos_logits)
        )
        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_logits,
            labels=tf.zeros_like(neg_logits)
        )

        # mf_losses = pos_losses * (1e-6 + pos_weights) + neg_losses * (1e-6 + 1. * neg_weights) * 300
        mf_losses = pos_losses * (1e-6 + pos_weights) + tf.reduce_mean(neg_losses * (1e-6 + neg_weights), axis=1) * 300

        l2_vars = [embedding_model.user_embeddings, embedding_model.item_embeddings]
        l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
        l2_loss = tf.add_n(l2_losses)

        loss = tf.reduce_sum(mf_losses) + l2_loss * l2

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss, mf_losses, l2_loss


for epoch in range(1, epoches):
    if epoch % 20 == 0:
        print("\nEvaluation before epoch {}".format(epoch))
        mean_results_dict = evaluate_mean_global_metrics(test_user_items_dict, train_user_items_dict,
                                                         embedding_model.user_embeddings, embedding_model.item_embeddings,
                                                         k_list=[10, 20],
                                                         metrics=["precision", "recall", "ndcg"])
        for metrics_name, score in mean_results_dict.items():
            print("{}: {:.4f}".format(metrics_name, score))
        print()

    step_losses = []
    step_mf_losses_list = []
    step_l2_losses = []

    start_time = time()

    for step, batch_edges in enumerate(
            tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]

        loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices)

        step_losses.append(loss.numpy())
        step_mf_losses_list.append(mf_losses.numpy())
        step_l2_losses.append(l2_loss.numpy())

    end_time = time()

    if optimizer.learning_rate.numpy() > 1e-5:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.995)
        lr_status = "update lr => {:.4f}".format(optimizer.learning_rate.numpy())
    else:
        lr_status = "current lr => {:.4f}".format(optimizer.learning_rate.numpy())

    print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\t{}\tepoch_time = {:.4f}s".format(
        epoch, np.mean(step_losses), np.mean(np.concatenate(step_mf_losses_list, axis=0)),
        np.mean(step_l2_losses), lr_status, end_time - start_time))

    if epoch == 1:
        print("the first epoch may take a long time to compile tf.function")
