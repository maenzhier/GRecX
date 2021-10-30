# coding=utf-8
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# replace data_base_dir with path of your own data directory
# data_base_dir = os.path.join(os.path.dirname(__file__), "../data")
# dataset = "yelp"
# diff_dataset = "flickr"
# data_dir = os.path.join(data_base_dir, dataset)


# train_path = os.path.join(data_dir, "{}.train.rating".format(dataset))
# test_path = os.path.join(data_dir, "{}.test.rating".format(dataset))
# links_path = os.path.join(data_dir, "{}.links".format(dataset))
# cached_data_path = os.path.join(data_dir, "cached.p")
# cached_neg_data_path = os.path.join(data_dir, "cached_neg.p")

num_negs = 1000
# test_neg_path = os.path.join(data_dir, "{}.test.negative".format(diff_dataset))

embedding_size = 64
# embedding_size = 256

# print(os.listdir(data_dir))
