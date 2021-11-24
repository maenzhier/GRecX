# coding = utf8

import numpy as np
import faiss


class VectorSearch(object):
    def __init__(self, base_vector):
        super(VectorSearch, self).__init__()
        self.base_vector = np.asarray(base_vector)
        self.dim = self.base_vector.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.base_vector)

    def search(self, target_vector, topK=10):
        target_vector = np.asarray(target_vector)
        topK_distances, topK_indices = self.index.search(target_vector, topK)

        return topK_distances, topK_indices