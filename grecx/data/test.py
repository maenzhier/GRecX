
# import scipy.sparse as sp
# import numpy as np
#
# t = np.random.random((3,4))
# print(t[t > 0.1])
#
# s = sp.csr_matrix(t)
# print(type(s[s>0.1]))

# P0 = sp.eye(10)
# P = P0
#
# print(P0)
# print(P)
#
# P0 = sp.eye(3)
#
# print("next")
# print(P0)
# print(P)

# a = 3
# b = [1,2,3]
#
# for s in list("ab"):
#     print(eval(s))

# test = {
#     "a":
# }
# test["a"] += 1
# test["a"] += 1
# print(test["a"])

# import numpy as np
# import tensorflow as tf
#
# d = list(np.random.rand(10, 2))
#
# def create_user_user_edges_generator():
#     while True:
#         for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(d).shuffle(1000000).batch(2)):
#             print(batch_edges)
#             a = batch_edges[:, 0]
#             b = batch_edges[:, 1]
#             yield a, b
#
#
# user_user_edges_generator = create_user_user_edges_generator()
#
# print(d)
#
# from diffnet.logging.remote_logging import log
# a, b = next(user_user_edges_generator)
# print(a,b)
# log(str(a))
# a, b = next(user_user_edges_generator)
# print(a,b)
# a, b = next(user_user_edges_generator)
# print(a,b)
# a, b = next(user_user_edges_generator)
# print(a,b)
# a, b = next(user_user_edges_generator)
# print(a,b)
#
# print("====")
# a, b = next(user_user_edges_generator)
# print(a,b)


import numpy as np
import tensorflow as tf
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
# d1 = np.random.rand(3, 4)
# print(d1)
# d2 = np.random.rand(3, 4)
# print(d2)
#
# d1 = tf.constant(d1)
# d2 = tf.constant(d2)
#
# # res = tf.math.maximum(d1, d2)
# res = (d1 + d2) / 2
#
# print(res)


# def consis_loss(logps, temp=0.5):
#     ps = [tf.math.exp(p) for p in logps]
#     sum_p = 0.
#     for p in ps:
#         sum_p = sum_p + p
#     avg_p = sum_p/len(ps)
#     #p2 = torch.exp(logp2)
#
#     sharp_p = (tf.math.pow(avg_p, 1./temp) / tf.stop_gradient(tf.math.reduce_sum(tf.math.pow(avg_p, 1./temp), axis=1, keepdims=True)))
#     loss = 0.
#     for p in ps:
#         loss += tf.math.reduce_mean(tf.reduce_sum(tf.math.pow((p-sharp_p), 2), axis=1))
#     loss = loss/len(ps)
#     return loss
#
# d1 = tf.constant(np.random.rand(5, 4))
# print(d1)
#
# d2 = tf.constant(np.random.rand(5, 4))
# print(d2)
#
# d3 = tf.constant(np.random.rand(5, 4))
# print(d3)
#
# ds = [d1, d2, d3]
# dls = []
# for d in ds:
#     dl = tf.nn.log_softmax(d, axis=-1)
#     dls.append(dl)
#     # dle = tf.math.exp(dl)
#     # print(dle)
#
# loss = consis_loss(dls)
# print(loss)

# d = np.zeros((3, 4))
# print(d)
#
# d1 = np.random.rand(4)
# print(d1)
#
# d[1] = d1
# print(d)

# import scipy.sparse as sp
# from scipy.sparse import *
#
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
#
# sd = csr_matrix((data, (row, col)), shape=(3, 3))
#
# ssd = sd.copy().toarray()
#
# print(sd.toarray())
# np.random.shuffle(ssd)
# print(ssd)


import numpy as np

# d = np.load("../../data/yelp/item_vector.npy")
#
# print(d)


# d = np.random.randint(1, 10, (10))
# print(d)
# d1 = map(lambda x: x+10, d)
# print(list(d1))
def fn(x):
    x = int(x / 4)
    return x
d = tf.map_fn(fn=fn, elems=tf.constant([3, 5, 2]))
print(d)

