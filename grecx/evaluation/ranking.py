# coding=utf-8
from itertools import chain

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from grecx.metrics.ranking import ndcg_score


def evaluate_mean_global_ndcg_score(user_items_dict, user_mask_items_dict, num_items,
                                    ranking_score_func,
                                    k_list=[5, 10, 15], user_batch_size=1000, item_batch_size=5000):


    results = []
    test_users = list(user_items_dict.keys())
    for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):

        user_rank_score_matrix = []

        for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
            user_batch_rank_score_matrix = ranking_score_func(batch_user_indices.numpy(), batch_item_indices.numpy())
            user_rank_score_matrix.append(user_batch_rank_score_matrix)

        user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)

        for user_index, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):

            result = {}
            results.append(result)

            user_index = user_index.numpy()
            # train_items = train_user_items_dict[user_index]
            items = user_items_dict[user_index]

            mask_items = user_mask_items_dict[user_index]

            # candidate_items = np.array(list(items) + list(user_neg_items_dict[user_index]))
            pred_items = np.argsort(user_rank_scores)[::-1][:k_list[-1] + len(mask_items)]

            pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]


            # pred_items = candidate_items[candidate_rank]

            pred_match = [1.0 if item in items else 0.0 for item in pred_items]

            for k in k_list:

                gold = [1] * len(items)
                if len(gold) > k:
                    gold = gold[:k]
                else:
                    gold = gold + [0] * (k - len(gold))

                ndcg = ndcg_score(gold, pred_match[:k])
                result["ndcg@{}".format(k)] = ndcg

    metrics = ["ndcg@{}".format(K) for K in k_list]
    mean_ndcg_dict = {}
    for metric in metrics:
        scores = [result[metric] for result in results]
        mean_ndcg = np.mean(scores)
        print(metric, mean_ndcg, len(scores))
        mean_ndcg_dict[metric] = mean_ndcg
    return mean_ndcg_dict



def evaluate_mean_candidate_ndcg_score(user_items_dict, user_neg_items_dict,
                                    ranking_score_func,
                                    k_list=[5, 10, 15], user_batch_size=1000, item_batch_size=5000, num_items=None):

    if num_items is None:
        num_items = 0
        for items in tqdm(chain(user_items_dict.values(), user_neg_items_dict.values())):
            for item in items:
                if item + 1 > num_items:
                    num_items = item + 1

        print(num_items)

    results = []
    test_users = list(user_items_dict.keys())
    for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):

        user_rank_score_matrix = []

        for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
            user_batch_rank_score_matrix = ranking_score_func(batch_user_indices.numpy(), batch_item_indices.numpy())
            user_rank_score_matrix.append(user_batch_rank_score_matrix)

        user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)

        for user_index, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):

            result = {}
            results.append(result)

            user_index = user_index.numpy()
            # train_items = train_user_items_dict[user_index]
            items = user_items_dict[user_index]

            candidate_items = np.array(list(items) + list(user_neg_items_dict[user_index]))
            candidate_scores = user_rank_scores[candidate_items]
            candidate_rank = np.argsort(candidate_scores)[::-1][:k_list[-1]]
            pred_items = candidate_items[candidate_rank]

            pred_match = [1.0 if item in items else 0.0 for item in pred_items]

            for k in k_list:

                gold = [1] * len(items)
                if len(gold) > k:
                    gold = gold[:k]
                else:
                    gold = gold + [0] * (k - len(gold))

                ndcg = ndcg_score(gold, pred_match[:k])
                result["ndcg@{}".format(k)] = ndcg

    metrics = ["ndcg@{}".format(K) for K in k_list]
    mean_ndcg_dict = {}
    for metric in metrics:
        scores = [result[metric] for result in results]
        mean_ndcg = np.mean(scores)
        print(metric, mean_ndcg, len(scores))
        mean_ndcg_dict[metric] = mean_ndcg
    return mean_ndcg_dict


# user_items_dict = {
#     0: [1, 2],
#     1: [2, 3]
# }
# user_neg_items_dict = {
#     0: [5, 6],
#     1: [1, 5]
# }



# def random_score_func(batch_user_indices, batch_item_indices):
#
#     batch_user_indices = batch_user_indices
#     batch_item_indices = batch_item_indices
#
#     score_matrix = []
#     for user_index in batch_user_indices:
#         test_item_indices = user_items_dict[user_index]
#         scores = [1.0 if item_index in test_item_indices else 0.0 for item_index in batch_item_indices]
#         score_matrix.append(scores)
#
#     return np.array(score_matrix)
#
# print(evaluate_mean_ndcg_score(user_items_dict, user_neg_items_dict, random_score_func))