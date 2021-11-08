# coding = utf8

import tensorflow as tf
import os
import numpy as np

from grecx.evaluation.ranking import evaluate_mean_global_ndcg_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from grecx.config import embedding_size
from grecx.datasets import LightGCNYelpDataset, LightGCNGowallaDataset, LightGCNAmazonbookDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils

data_dict = LightGCNYelpDataset().load_data()
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
batch_size = 5000


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
        # user_h = self.user_embeddings

        item_h = self.drop_layer(self.item_embeddings, training=training)
        # item_h = self.item_embeddings

        return user_h, item_h


model = MF(num_users, num_items, embedding_size)


@tf_utils.function
def forward(training=False):
    return model([], training=training)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

@tf_utils.function
def mf_score_func(batch_user_indices, batch_item_indices):
    user_h, item_h = forward(training=False)
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
            # losses = tf.reduce_sum(tf.nn.softplus(-(pos_logits - neg_logits)))

            logits = tf.stack([pos_logits, neg_logits], axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.one_hot(tf.zeros([tf.shape(logits)[0]], dtype=tf.int32), depth=2)
            )

            l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
            l2_vars.append(model.user_embeddings)
            l2_vars.append(model.item_embeddings)
            l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
            l2_loss = tf.add_n(l2_losses)

            # mf_l2_vars = [user_h, item_h]
            # mf_l2_losses = [tf.nn.l2_loss(var) for var in mf_l2_vars]
            # mf_l2_loss = tf.add_n(mf_l2_losses)

            # loss = tf.reduce_sum(losses) + l2_loss * l2 + 1e-7 * mf_l2_loss
            loss = tf.reduce_sum(losses) + l2_loss * l2

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if epoch % 20 == 0 and step % 1000 == 0:

            mean_ndcg_dict = evaluate_mean_global_ndcg_score(
                test_user_items_dict, train_user_items_dict, num_items, mf_score_func,
                # user_batch_size=4000,
                # item_batch_size=8000
            )
            print("epoch = {}\tresults = {}".format(epoch, mean_ndcg_dict))

