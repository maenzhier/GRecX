# coding=utf-8
from multiprocessing import Pool

from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class DiffNetDataset(DownloadableDataset):

    """
    Datasets used in the DiffNet model: https://github.com/maenzhier/grecx_datasets
    """

    def __init__(self, dataset_name, featureless=True, download_urls=None, dataset_root_path=None):
        """

        :param dataset_name: "diff_net_yelp" | "diff_net_flickr"
        :param featureless:
        :param download_urls:
        :param dataset_root_path:
        """
        super().__init__(dataset_name, download_urls,
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path)
        self.featureless = featureless

    def read_edge_info(self, file_path):
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


    def process(self):
        if "Yelp" in self.dataset_name:
            name = "yelp"
        else:
            name = "flickr"
        data_dir = os.path.join(self.raw_root_path, "diffnet_{}".format(name))
        print(data_dir)

        train_path = os.path.join(data_dir, "{}.train.rating".format(name))
        test_path = os.path.join(data_dir, "{}.test.rating".format(name))
        links_path = os.path.join(data_dir, "{}.links".format(name))

        train_user_items_dict, train_user_item_edges = self.read_edge_info(train_path)
        test_user_items_dict, test_user_item_edges = self.read_edge_info(test_path)
        user_users_dict, user_user_edges = self.read_edge_info(links_path)

        user_item_edges = np.concatenate([train_user_item_edges, test_user_item_edges], axis=0)
        num_users, num_items = user_item_edges.max(axis=0) + 1

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

        return num_users, num_items, train_user_item_edges, test_user_items_dict, user_user_edges, user_neg_items_dict


class DiffNetYelp(DiffNetDataset):

    """
    The Yelp dataset used in the DiffNet model: https://github.com/maenzhier/grecx_datasets/tree/main/yelp
    """

    def __init__(self, dataset_root_path=None, featureless=True):
        super().__init__(
            dataset_name="DiffNetYelp",
            featureless=featureless,
            download_urls="https://github.com/maenzhier/grecx_datasets/raw/main/yelp/diff_net_yelp.zip",
            dataset_root_path=dataset_root_path,
        )


class DiffNetFlickr(DiffNetDataset):
    """
    The Flickr dataset used in the DiffNet model: https://github.com/maenzhier/grecx_datasets/tree/main/yelp
    """
    def __init__(self, dataset_root_path=None, featureless=True):
        super().__init__(
            dataset_name="diff_net_flickr",
            featureless=featureless,
            download_urls="https://github.com/maenzhier/grecx_datasets/raw/main/flickr/diff_net_flickr.zip",
            dataset_root_path=dataset_root_path,
        )