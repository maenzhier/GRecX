# coding=utf-8
from multiprocessing import Pool

from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import random as rd
from time import time


class Data(object):
    def __init__(self, path):
        self.path = path

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0

        self.train_user_items_dict, self.train_user_item_edges= self.read_edge_info(train_file)
        self.test_user_items_dict, self.test_user_item_edges = self.read_edge_info(test_file)

        self.user_item_edges = np.concatenate([self.train_user_item_edges, self.test_user_item_edges], axis=0)
        self.num_users, self.num_items = self.user_item_edges.max(axis=0) + 1

    def read_edge_info(self, file_path):
            edge_dict = {}
            edges = []

            with open(file_path, "r", encoding="utf-8") as f:
                for l in f.readlines():
                    if len(l) > 0:
                        try:
                            l = l.strip('\n').split(' ')
                            items = []
                            uid = int(l[0])
                            for i in l[1:]:
                                i = int(i)
                                items.append(i)
                                edges.append([uid, i])
                            if uid not in edge_dict:
                                edge_dict[uid] = set(items)
                            else:
                                item_set = edge_dict[uid]
                                edge_dict[uid] = set(items).union(item_set)
                        except Exception:
                            continue

            edges = np.array(edges)
            return edge_dict, edges


class LightGCNDataset(DownloadableDataset):
    def __init__(self, dataset_name, featureless=True, download_urls=None, dataset_root_path=None):
        super().__init__(dataset_name, download_urls,
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path)
        self.featureless = featureless

    def process(self):
        if "yelp" in self.dataset_name:
            name = "yelp"
        elif "gowalla" in self.dataset_name:
            name = "gowalla"
        else:
            name = "amazon-book"

        data_dir = os.path.join(self.raw_root_path, "light_gcn_{}".format(name))
        print(data_dir)

        data = Data(path=data_dir)

        train_user_items_dict, train_user_item_edges = data.train_user_items_dict, data.train_user_item_edges
        test_user_items_dict, test_user_item_edges = data.test_user_items_dict, data.test_user_item_edges

        user_item_edges = data.user_item_edges
        num_users, num_items = data.num_users, data.num_items

        num_negs = 1000

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

        user_neg_items_dict = build_user_neg_items_dict()

        return num_users, num_items, train_user_item_edges, test_user_items_dict, user_neg_items_dict


class LightGCNYelp(LightGCNDataset):
    def __init__(self, dataset_root_path=None, batch_size=1024, featureless=True):
        super().__init__(
            dataset_name="light_gcn_yelp",
            featureless=featureless,
            download_urls="https://github.com/maenzhier/grecx_datasets/raw/main/yelp/light_gcn_yelp.zip",
            dataset_root_path=dataset_root_path,
        )


class LightGCNGowalla(LightGCNDataset):
    def __init__(self, dataset_root_path=None, batch_size=1024, featureless=True):
        super().__init__(
            dataset_name="light_gcn_gowalla",
            featureless=featureless,
            download_urls="https://github.com/maenzhier/grecx_datasets/raw/main/gowalla/light_gcn_gowalla.zip",
            dataset_root_path=dataset_root_path,
        )


class LightGCNAmazonbook(LightGCNDataset):
    def __init__(self, dataset_root_path=None, batch_size=1024, featureless=True):
        super().__init__(
            dataset_name="light_gcn_amazon-book",
            featureless=featureless,
            download_urls="https://github.com/maenzhier/grecx_datasets/raw/main/amazon-book/light_gcn_amazon-book.zip",
            dataset_root_path=dataset_root_path,
        )
