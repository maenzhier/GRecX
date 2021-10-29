# coding = utf8

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from data.config import embedding_size
from diffnet.data.load_data import *
from diffnet.evaluation.model_evaluation import evaluate

import tf_geometric as tfg
from tf_geometric.utils import tf_utils

from diffnet.data.load_graph import generate_user_item_graph


drop_rate = 0.15
lr = 1e-3
l2 = 5e-5
# l2 = 2e-4
# graph_name = "ionly"
# graph_name = "uonly"
graph_name = "5"

epoches = 2700
batch_size = 5000



class UIGCN(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_gcn = tfg.layers.GCN(self.emb_dim, activation=tf.nn.relu)
        self.item_gcn = tfg.layers.GCN(self.emb_dim, activation=tf.nn.relu)

        self.drop_layer = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        user_graph, item_graph = inputs

        user_h = self.user_gcn([user_graph.x, user_graph.edge_index, user_graph.edge_weight],
                               cache=user_graph.cache, training=training)
        user_h = self.drop_layer(user_h, training=training)

        # user_h = user_h + user_h_1

        item_h = self.item_gcn([item_graph.x, item_graph.edge_index, item_graph.edge_weight],
                               cache=item_graph.cache, training=training)
        item_h = self.drop_layer(item_h, training=training)

        # item_h = item_h + item_h_1

        return user_h, item_h


class IGCN(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_embeddings = tf.Variable(tf.random.truncated_normal([self.num_users, self.emb_dim], stddev=np.sqrt(1/self.emb_dim)))
        self.item_gcn = tfg.layers.GCN(self.emb_dim, activation=tf.nn.relu)

        self.drop_layer = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        user_graph, item_graph = inputs

        user_h = self.user_embeddings
        user_h = self.drop_layer(user_h, training=training)
        item_h = self.item_gcn([item_graph.x, item_graph.edge_index, item_graph.edge_weight],
                               cache=item_graph.cache)
        item_h = self.drop_layer(item_h, training=training)

        return user_h, item_h


class UGCN(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.item_embeddings = tf.Variable(tf.random.truncated_normal([self.num_items, self.emb_dim], stddev=np.sqrt(1/self.emb_dim)))
        self.user_gcn = tfg.layers.GCN(self.emb_dim, activation=tf.nn.relu)

        self.drop_layer = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        user_graph, item_graph = inputs

        user_h = self.user_gcn([user_graph.x, user_graph.edge_index, user_graph.edge_weight],
                               cache=user_graph.cache)
        user_h = self.drop_layer(user_h, training=training)
        item_h = self.item_embeddings
        item_h = self.drop_layer(item_h, training=training)

        return user_h, item_h


model = UIGCN(num_users, num_items, embedding_size)


@tf_utils.function
def forward(user_graph, item_graph, training=False):
    return model([user_graph, item_graph], training=training)


user_user_graph, item_item_graph, user_user_social_graph = generate_user_item_graph(graph_name)
u_graph = user_user_graph.tocoo()
i_graph = item_item_graph.tocoo()
user_graph = tfg.Graph(
    x=tf.sparse.eye(num_users),
    edge_index=[u_graph.row, u_graph.col]
)

item_graph = tfg.Graph(
    x=tf.sparse.eye(num_items),
    edge_index=[i_graph.row, i_graph.col]
)

model.user_gcn.build_cache_for_graph(user_graph)
model.item_gcn.build_cache_for_graph(item_graph)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)




# @tf_utils.function
# def train_step():
#     with tf.GradientTape() as tape:
#
#         loss = compute_loss(logits, train_index, tape.watched_variables())
#
#     vars = tape.watched_variables()
#     grads = tape.gradient(loss, vars)
#     optimizer.apply_gradients(zip(grads, vars))
#     return loss
#
@tf_utils.function
def mf_score_func(batch_user_indices, batch_item_indices):
    user_h, item_h = forward(user_graph, item_graph, training=False)
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
            user_h, item_h = forward(user_graph, item_graph, training=True)

            embedded_users = tf.gather(user_h, batch_user_indices)
            embedded_items = tf.gather(item_h, batch_item_indices)
            embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

            pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
            # pos_logits_1 = tf.nn.sigmoid(tf.reduce_sum(embedded_users * embedded_items, axis=-1))
            neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)
            # neg_logits_1 = tf.nn.sigmoid(tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1))

            pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_logits,
                labels=tf.ones_like(pos_logits)
            )
            # pos_losses_1 = tf.nn.l2_loss(pos_logits-tf.ones_like(pos_logits))
            neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_logits,
                labels=tf.zeros_like(neg_logits)
            )
            # neg_losses_1 = tf.nn.l2_loss(neg_logits-tf.zeros_like(neg_logits))

            losses = pos_losses + neg_losses

            l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
            # l2_vars.append(model.user_embeddings)
            # l2_vars.append(model.item_embeddings)
            l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
            l2_loss = tf.add_n(l2_losses)
            # loss = tf.reduce_sum(losses) + l2_loss * 1e-2
            loss = tf.reduce_sum(losses) + l2_loss * l2

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if epoch % 20 == 0 and step % 1000 == 0:

            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))
            if step == 0:
                evaluate(mf_score_func)

