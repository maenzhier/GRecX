# coding=utf-8
from itertools import chain

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from grecx.metrics.ranking import ndcg_score, precision_score, recall_score
from grecx.vector_search.vector_search import VectorSearchEngine


# def pred2ndcg_dict(pred_match, num_items, k_list):
#     ndcg_dict = {}
#     for k in k_list:
#         gold = [1] * num_items
#         if len(gold) > k:
#             gold = gold[:k]
#         else:
#             gold = gold + [0] * (k - len(gold))
#
#         ndcg = ndcg_score(gold, pred_match[:k])
#         ndcg_dict["ndcg@{}".format(k)] = ndcg
#     return ndcg_dict
#
#
# def ndcg2mean_ndcg_dict(results, k_list):
#     metrics = ["ndcg@{}".format(K) for K in k_list]
#     mean_ndcg_dict = {}
#     for metric in metrics:
#         scores = [result[metric] for result in results]
#         mean_ndcg = np.mean(scores)
#         mean_ndcg_dict[metric] = mean_ndcg
#     return mean_ndcg_dict


# def evaluate_mean_global_ndcg_score(user_items_dict, user_mask_items_dict, num_items,
#                                     ranking_score_func,
#                                     k_list=[20], user_batch_size=1000, item_batch_size=5000):
#
#     results = []
#     test_users = list(user_items_dict.keys())
#     for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):
#
#         user_rank_score_matrix = []
#
#         for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
#             user_batch_rank_score_matrix = ranking_score_func(batch_user_indices.numpy(), batch_item_indices.numpy())
#             user_rank_score_matrix.append(user_batch_rank_score_matrix)
#
#         user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)
#
#         for user, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):
#
#             result = {}
#             results.append(result)
#
#             user = user.numpy()
#             # train_items = train_user_items_dict[user_index]
#             items = user_items_dict[user]
#
#             mask_items = user_mask_items_dict[user]
#
#             # candidate_items = np.array(list(items) + list(user_neg_items_dict[user_index]))
#             pred_items = np.argsort(user_rank_scores)[::-1][:k_list[-1] + len(mask_items)]
#
#             pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
#
#             pred_match = [1.0 if item in items else 0.0 for item in pred_items]
#
#             for k in k_list:
#
#                 gold = [1] * len(items)
#                 if len(gold) > k:
#                     gold = gold[:k]
#                 else:
#                     gold = gold + [0] * (k - len(gold))
#
#                 ndcg = ndcg_score(gold, pred_match[:k])
#                 result["ndcg@{}".format(k)] = ndcg
#
#     metrics = ["ndcg@{}".format(K) for K in k_list]
#     mean_ndcg_dict = {}
#     for metric in metrics:
#         scores = [result[metric] for result in results]
#         mean_ndcg = np.mean(scores)
#         # print(metric, mean_ndcg, len(scores))
#         mean_ndcg_dict[metric] = mean_ndcg
#     return mean_ndcg_dict



# def evaluate_mean_global_all_score(user_items_dict, user_mask_items_dict, num_items,
#                                     ranking_score_func,
#                                     k_list=[20], user_batch_size=1000, item_batch_size=5000):
#
#     results = []
#     test_users = list(user_items_dict.keys())
#     for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):
#
#         user_rank_score_matrix = []
#
#         for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
#             user_batch_rank_score_matrix = ranking_score_func(batch_user_indices.numpy(), batch_item_indices.numpy())
#             user_rank_score_matrix.append(user_batch_rank_score_matrix)
#
#         user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)
#
#         for user, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):
#
#             result = {}
#             results.append(result)
#
#             user = user.numpy()
#             # train_items = train_user_items_dict[user_index]
#             items = user_items_dict[user]
#
#             mask_items = user_mask_items_dict[user]
#
#             # candidate_items = np.array(list(items) + list(user_neg_items_dict[user_index]))
#             pred_items = np.argsort(user_rank_scores)[::-1][:k_list[-1] + len(mask_items)]
#
#             pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
#
#             pred_match = [1.0 if item in items else 0.0 for item in pred_items]
#
#             for k in k_list:
#
#                 gold = [1] * len(items)
#                 pr_gold = [1] * len(items)
#                 if len(gold) > k:
#                     gold = gold[:k]
#                 else:
#                     gold = gold + [0] * (k - len(gold))
#
#                 ndcg = ndcg_score(gold, pred_match[:k])
#                 p = precision(pr_gold, pred_match[:k])
#                 r = recall(pr_gold, pred_match[:k])
#                 result["ndcg@{}".format(k)] = ndcg
#                 result["pre@{}".format(k)] = p
#                 result["recall@{}".format(k)] = r
#
#     metrics = [("ndcg@{}".format(K), "pre@{}".format(K), "recall@{}".format(K)) for K in k_list]
#     mean_ndcg_dict = {}
#     mean_pre_dict = {}
#     mean_recall_dict = {}
#     for metric in metrics:
#         ndcg_scores = [result[metric[0]] for result in results]
#         mean_ndcg = np.mean(ndcg_scores)
#         mean_ndcg_dict[metric[0]] = mean_ndcg
#         pre_scores = [result[metric[1]] for result in results]
#         mean_pre = np.mean(pre_scores)
#         mean_pre_dict[metric[1]] = mean_pre
#         recall_scores = [result[metric[2]] for result in results]
#         mean_recall = np.mean(recall_scores)
#         mean_recall_dict[metric[2]] = mean_recall
#     return mean_ndcg_dict, mean_pre_dict, mean_recall_dict

# def evaluate_mean_candidate_ndcg_score(user_items_dict, user_neg_items_dict,
#                                     ranking_score_func,
#                                     k_list=[20], user_batch_size=1000, item_batch_size=5000, num_items=None):
#
#     if num_items is None:
#         num_items = max(max(items) for items in tqdm(chain(user_items_dict.values(), user_neg_items_dict.values())))+1
#         print(num_items)
#
#     results = []
#     test_users = list(user_items_dict.keys())
#     for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):
#
#         user_rank_score_matrix = []
#
#         for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
#             user_batch_rank_score_matrix = ranking_score_func(batch_user_indices.numpy(), batch_item_indices.numpy())
#             user_rank_score_matrix.append(user_batch_rank_score_matrix)
#
#         user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)
#
#         for user, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):
#
#             result = {}
#             results.append(result)
#
#             user = user.numpy()
#             # train_items = train_user_items_dict[user_index]
#             items = user_items_dict[user]
#
#             candidate_items = np.array(list(items) + list(user_neg_items_dict[user]))
#             candidate_scores = user_rank_scores[candidate_items]
#             candidate_rank = np.argsort(candidate_scores)[::-1][:k_list[-1]]
#             pred_items = candidate_items[candidate_rank]
#
#             pred_match = [1.0 if item in items else 0.0 for item in pred_items]
#
#             for k in k_list:
#
#                 gold = [1] * len(items)
#                 if len(gold) > k:
#                     gold = gold[:k]
#                 else:
#                     gold = gold + [0] * (k - len(gold))
#
#                 ndcg = ndcg_score(gold, pred_match[:k])
#                 result["ndcg@{}".format(k)] = ndcg
#
#     metrics = ["ndcg@{}".format(K) for K in k_list]
#     mean_ndcg_dict = {}
#     for metric in metrics:
#         scores = [result[metric] for result in results]
#         mean_ndcg = np.mean(scores)
#         # print(metric, mean_ndcg, len(scores))
#         mean_ndcg_dict[metric] = mean_ndcg
#     return mean_ndcg_dict

def score(ground_truth, pred_items, k_list, metrics):
    pred_match = [1 if item in ground_truth else 0 for item in pred_items]

    max_k = k_list[-1]
    if len(ground_truth) > max_k:
        ndcg_gold = [1] * max_k
    else:
        ndcg_gold = [1] * len(ground_truth) + [0] * (max_k - len(ground_truth))

    res_score = []
    for metric in metrics:
        if metric == "ndcg":
            score_func = ndcg_score
        elif metric == "precision":
            score_func = precision_score
        elif metric == "recall":
            score_func = recall_score
        else:
            raise Exception("Not Found Metric : {}".format(metric))

        for k in k_list:
            if metric == "ndcg":
                res_score.append(score_func(ndcg_gold[:k], pred_match[:k]))
            else:
                res_score.append(score_func(ground_truth, pred_match[:k]))

    return res_score


def evaluate_mean_global_metrics(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    v_search = VectorSearchEngine(item_embedding)

    if tf.is_tensor(user_embedding):
        user_embedding = user_embedding.numpy()
    else:
        user_embedding = np.asarray(user_embedding)

    user_indices = list(user_items_dict.keys())
    embedded_users = user_embedding[user_indices]
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

    _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)

    res_scores = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]

        res_score = score(items, pred_items, k_list, metrics)

        res_scores.append(res_score)

    res_scores = np.asarray(res_scores)
    names = []
    for metric in metrics:
        for k in k_list:
            names.append("{}@{}".format(metric, k))

    # return list(zip(names, np.mean(res_scores, axis=0, keepdims=False)))
    return dict(zip(names, np.mean(res_scores, axis=0, keepdims=False)))


# def evaluate_mean_global_metrics(user_items_dict, user_mask_items_dict,
#                                  user_embedding, item_embedding,
#                                  k_list=[10, 20], metrics=["ndcg"]):
#
#     v_search = VectorSearch(item_embedding)
#
#     if isinstance(user_embedding, tf.Tensor):
#         user_embedding = np.asarray(user_embedding)
#
#     user_indices = list(user_items_dict.keys())
#     embedded_users = user_embedding[user_indices]
#     max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)
#
#     _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)
#
#     results = []
#     for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):
#
#         items = user_items_dict[user]
#         mask_items = user_mask_items_dict[user]
#         pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
#
#         pred_match = [1.0 if item in items else 0.0 for item in pred_items]
#
#         results.append(pred2ndcg_dict(pred_match, len(items), k_list))
#
#     return ndcg2mean_ndcg_dict(results, k_list)


