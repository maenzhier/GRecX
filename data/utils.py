# coding = utf8
from scipy.sparse import *


def save_graph(data, path):
    import pickle
    f = open(path, 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def read_graph(path)->csr_matrix:
    import pickle
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph