# coding=utf-8


import tensorflow as tf
# tf.enable_eager_execution()

from grecx.data.load_data import *
from grecx.evaluation.ncf_measures import find_ndcg


def evaluate(score_func, k_list=[10, 20], user_batch_size=1000, item_batch_size=1000):
    results = []
    test_users = list(test_user_items_dict.keys())
    for batch_user_indices in tqdm(tf.data.Dataset.from_tensor_slices(test_users).batch(user_batch_size)):

        user_rank_score_matrix = []

        for batch_item_indices in tf.data.Dataset.range(num_items).batch(item_batch_size):
            user_batch_rank_score_matrix = score_func(batch_user_indices, batch_item_indices)
            user_rank_score_matrix.append(user_batch_rank_score_matrix)

        user_rank_score_matrix = np.concatenate(user_rank_score_matrix, axis=1)

        for user_index, user_rank_scores in zip(batch_user_indices, user_rank_score_matrix):


            result = {}
            results.append(result)

            user_index = user_index.numpy()
            # train_items = train_user_items_dict[user_index]
            test_items = test_user_items_dict[user_index]

            candidate_items = np.array(list(test_items) + list(user_neg_items_dict[user_index]))
            candidate_scores = user_rank_scores[candidate_items]
            candidate_rank = np.argsort(candidate_scores)[::-1][:k_list[-1]]
            pred_items = candidate_items[candidate_rank]

            # rank = np.argsort(user_rank_scores)[::-1]
            #
            # pred_rank = []
            # for item_index in rank:
            #     if item_index in train_items:
            #         continue
            #     else:
            #         pred_rank.append(item_index)
            #         if len(pred_rank) >= k_list[-1]:
            #             break

            pred_match = [1.0 if item_index in test_items else 0.0 for item_index in pred_items]

            for k in k_list:

                # if len(test_items) != 1:
                #     raise Exception("wrong number of test items: ", test_items)

                # print(user_index, pred_rank[:k], test_items)

                gold = [1] * len(test_items)
                if len(gold) > k:
                    gold = gold[:k]
                else:
                    gold = gold + [0] * (k - len(gold))

                ndcg = find_ndcg(gold, pred_match[:k])
                result["ndcg@{}".format(k)] = ndcg

        print(len(results))

    metrics = ["ndcg@{}".format(K) for K in k_list]
    mean_scores = {}
    for metric in metrics:
        scores = [result[metric] for result in results]
        mean_score = np.mean(scores)
        mean_scores[metric] = mean_score
        print(metric, mean_score, len(scores))

    return metrics, mean_scores





def random_score_func(batch_user_indices, batch_item_indices):
    # user_batch_rank_score_matrix = np.random.randn(batch_user_indices.shape[0], batch_item_indices.shape[0])
    # return user_batch_rank_score_matrix

    batch_user_indices = batch_user_indices.numpy()
    batch_item_indices = batch_item_indices.numpy()

    score_matrix = []
    for user_index in batch_user_indices:
        test_item_indices = test_user_items_dict[user_index]
        scores = [1.0 if item_index in test_item_indices else 0.0 for item_index in batch_item_indices]
        score_matrix.append(scores)

    return np.array(score_matrix)


# evaluate(random_score_func)