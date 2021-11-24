# coding = utf8

import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from grecx.data.load_graph import generate_lightGCN_user_item_graph
from grecx.evaluation.ranking import evaluate_mean_global_metrics

from grecx.config import embedding_size
from grecx.datasets import LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils

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

drop_rate = 0.15
lr = 1e-2
# l2 = 5e-7
# l2 = 1e-4
l2 = 1e-4

epoches = 2700
batch_size = 5000


def build_constraint_mat(train_data):
    train_data = list(train_data)
    train_mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = beta_uD.dot(beta_iD)  # n_user * m_item
    constraint_mat = tf.convert_to_tensor(constraint_mat, dtype=tf.float32)

    return constraint_mat


constraint_mat = build_constraint_mat(train_user_item_edges)


class Utrla_MF(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_embeddings = tf.Variable(tf.random.truncated_normal([self.num_users, self.emb_dim], stddev=np.sqrt(0.01)))
        self.item_embeddings = tf.Variable(tf.random.truncated_normal([self.num_items, self.emb_dim], stddev=np.sqrt(0.01)))
        self.drop_layer = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        user_h = self.drop_layer(self.user_embeddings, training=training)
        #
        item_h = self.drop_layer(self.item_embeddings, training=training)

        return user_h, item_h


model = Utrla_MF(num_users, num_items, embedding_size)


@tf_utils.function
def forward(training=False):
    return model([], training=training)


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf_utils.function
def train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices, pos_weights, neg_weights):
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

        mf_losses = pos_losses * (1e-6 + pos_weights) + neg_losses * (1e-6 + 1. * neg_weights)

        l2_vars = [model.user_embeddings, model.item_embeddings]
        l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
        l2_loss = tf.add_n(l2_losses)

        loss = tf.reduce_sum(mf_losses) + l2_loss * l2

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss, mf_losses, l2_loss


for epoch in range(1, epoches):
    if epoch % 20 == 0:
        user_h, item_h = forward(training=False)
        print("\nEvaluation before epoch {}".format(epoch))
        mean_results_dict = evaluate_mean_global_metrics(test_user_items_dict, train_user_items_dict,
                                                         user_h, item_h, k_list=[10, 20], metrics=["recall", "ndcg"])
        print(mean_results_dict)
        print()


    step_losses = []
    step_mf_losses_list = []
    step_l2_losses = []

    start_time = time()

    for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]
        batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)

        pos_weights = tf.gather_nd(constraint_mat, tf.stack([batch_user_indices, batch_item_indices], axis=1))
        neg_weights = tf.gather_nd(constraint_mat, tf.stack([batch_user_indices, batch_neg_item_indices], axis=1))

        loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices, pos_weights, neg_weights)

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
        np.mean(step_l2_losses), lr_status, end_time-start_time))

    if epoch == 1:
        print("the first epoch may take a long time to compile tf.function")
