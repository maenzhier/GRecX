# coding=utf-8

# from ngcf.data.load_adj_data import *

import numpy as np
import os
from tqdm import tqdm

from diffnet.config import data_dir, train_path, test_path, links_path, num_negs, cached_data_path, cached_neg_data_path
from diffnet.utils.data_utils import load_cache


def read_edge_info(file_path):
    edge_dict = {}
    edges = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):

            items = [int(item) for item in line.split()[:3]]
            a = items[0]
            b = items[1]
            rating = items[2]

            if a not in edge_dict:
                item_set = set()
                edge_dict[a] = item_set
            else:
                item_set = edge_dict[a]

            if rating > 0:
                item_set.add(b)
                edges.append([a, b])

    edges = np.array(edges)
    return edge_dict, edges



def read_data():
    train_user_items_dict, train_user_item_edges = read_edge_info(train_path)
    test_user_items_dict, test_user_item_edges = read_edge_info(test_path)
    user_users_dict, user_user_edges = read_edge_info(links_path)
    return train_user_items_dict, train_user_item_edges, \
           test_user_items_dict, test_user_item_edges, \
           user_users_dict, user_user_edges

train_user_items_dict, train_user_item_edges, test_user_items_dict, test_user_item_edges, user_users_dict, user_user_edges \
    = load_cache(cached_data_path, create_func=read_data)
user_item_edges = np.concatenate([train_user_item_edges, test_user_item_edges], axis=0)
num_users, num_items = user_item_edges.max(axis=0) + 1



def build_user_neg_items_dict():
    user_neg_items_dict = {}
    all_items = list(range(num_items))
    for user in tqdm(test_user_items_dict):
        p = np.ones([num_items])
        p[list(train_user_items_dict[user])] = 0.0
        p[list(test_user_items_dict[user])] = 0.0
        p /= p.sum()
        neg_items = np.random.choice(all_items, num_negs, replace=False, p=p)
        user_neg_items_dict[user] = neg_items
    return user_neg_items_dict

user_neg_items_dict = load_cache(cached_neg_data_path, build_user_neg_items_dict)


