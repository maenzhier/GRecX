# coding = utf8

import tensorflow as tf
import os
import numpy as np

from grecx.evaluation.ranking_faiss import evaluate_mean_global_ndcg_score_with_faiss

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from grecx.config import embedding_size
from grecx.datasets import LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils


# drop_rate = 0.3
# lr = 2e-3
# l2 = 1e-4
# data_dict = LightGCNYelpDataset().load_data()

data_dict = LightGCNGowallaDataset().load_data()
# data_dict = LightGCNAmazonbookDataset().load_data()
num_users = data_dict["num_users"]
num_items = data_dict["num_items"]
user_item_edges = data_dict["user_item_edges"]
train_index = data_dict["train_index"]
train_user_item_edges = user_item_edges[train_index]
train_user_items_dict = data_dict["train_user_items_dict"]
test_user_items_dict = data_dict["test_user_items_dict"]


drop_rate = 0.3
lr = 2e-3
l2 = 1e-4

epoches = 2700
batch_size = 8000


class MF(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_embeddings = tf.Variable(tf.random.truncated_normal([self.num_users, self.emb_dim], stddev=np.sqrt(1/self.emb_dim)))
        self.item_embeddings = tf.Variable(tf.random.truncated_normal([self.num_items, self.emb_dim], stddev=np.sqrt(1/self.emb_dim)))
        self.drop_layer = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        user_h = self.drop_layer(self.user_embeddings, training=training)

        item_h = self.drop_layer(self.item_embeddings, training=training)

        return user_h, item_h


model = MF(num_users, num_items, embedding_size)


@tf_utils.function
def forward(training=False):
    return model([], training=training)


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf_utils.function
def train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices):
    with tf.GradientTape() as tape:
        user_h, item_h = forward(training=True)

        embedded_users = tf.gather(user_h, batch_user_indices)
        embedded_items = tf.gather(item_h, batch_item_indices)
        embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

        pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
        neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)
        #
        # pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=pos_logits,
        #     labels=tf.ones_like(pos_logits)
        # )
        # neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=neg_logits,
        #     labels=tf.zeros_like(neg_logits)
        # )
        #
        # losses = pos_losses + neg_losses
        mf_losses = tf.nn.softplus(-(pos_logits - neg_logits))

        l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
        l2_vars.append(model.user_embeddings)
        l2_vars.append(model.item_embeddings)
        l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
        l2_loss = tf.add_n(l2_losses)

        loss = tf.reduce_sum(mf_losses) + l2_loss * l2

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss, mf_losses, l2_loss


for epoch in range(0, epoches):

    if epoch % 20 == 0:
        user_h, item_h = forward(training=False)
        print("epoch = {}".format(epoch))
        mean_ndcg_dict_faiss = evaluate_mean_global_ndcg_score_with_faiss(test_user_items_dict, train_user_items_dict,
                                                                          user_h, item_h)
        print(mean_ndcg_dict_faiss)

    step_losses = []
    step_mf_losses_list = []
    step_l2_losses = []


    for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]
        batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)

        loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices)

        step_losses.append(loss.numpy())
        step_mf_losses_list.append(mf_losses.numpy())
        step_l2_losses.append(l2_loss.numpy())

    print("epoch = {}\tloss = {}\tmf_loss = {}\tl2_loss = {}".format(
        epoch, np.mean(step_losses), np.mean(np.concatenate(step_mf_losses_list, axis=0)),
        np.mean(step_l2_losses)))

    if optimizer.learning_rate.numpy() > 1e-5:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.995)
        print("update lr: ", optimizer.learning_rate)
    else:
        print("current lr: ", optimizer.learning_rate)


