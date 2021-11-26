# coding=utf-8

from tf_geometric.data.dataset import DownloadableDataset
import os
import numpy as np


class LightGCNDataset(DownloadableDataset):
    """
    Datasets used in the LightGCN model: https://github.com/maenzhier/grecx_datasets
    """

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: "light_gcn_yelp" | "light_gcn_gowalla" | "light_gcn_amazon-book"
        :param featureless:
        :param download_urls:
        :param dataset_root_path:
        """
        super().__init__(dataset_name,
                         download_urls=[
                             "https://github.com/maenzhier/grecx_datasets/raw/main/{}/{}.zip".format(
                                 dataset_name.replace("light_gcn_", ""), dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path)

    def _read_edge_info(self, file_path):
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

    def process(self):

        name = self.dataset_name.replace("light_gcn_", "")
        path = os.path.join(self.raw_root_path, "light_gcn_{}".format(name))

        # data = _LightGCNData(path=data_dir)

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        train_user_items_dict, train_user_item_edges = self._read_edge_info(train_file)
        test_user_items_dict, test_user_item_edges = self._read_edge_info(test_file)

        user_item_edges = np.concatenate([train_user_item_edges, test_user_item_edges], axis=0)
        index = np.arange(user_item_edges.shape[0])
        num_train_edges = train_user_item_edges.shape[0]
        train_index, test_index = index[:num_train_edges], index[num_train_edges:]
        num_users, num_items = user_item_edges.max(axis=0) + 1

        return {
            "num_users": num_users,
            "num_items": num_items,
            "user_item_edges": user_item_edges,
            "train_index": train_index,
            "test_index": test_index,
            "train_user_items_dict": train_user_items_dict,
            "test_user_items_dict": test_user_items_dict,
        }


class LightGCNYelpDataset(LightGCNDataset):
    """
    The Yelp dataset used in the LightGCN model: https://github.com/maenzhier/grecx_datasets/tree/main/yelp
    """

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="light_gcn_yelp", dataset_root_path=dataset_root_path)


class LightGCNGowallaDataset(LightGCNDataset):
    """
    The Gowalla dataset used in the LightGCN model: https://github.com/maenzhier/grecx_datasets/tree/main/gowalla
    """

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="light_gcn_gowalla", dataset_root_path=dataset_root_path)


class LightGCNAmazonbookDataset(LightGCNDataset):
    """
    The Amazonbook dataset used in the LightGCN model: https://github.com/maenzhier/grecx_datasets/tree/main/amazon-book
    """

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="light_gcn_amazon-book", dataset_root_path=dataset_root_path)
