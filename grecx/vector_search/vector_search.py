# coding = utf8

import numpy as np
import faiss


class VectorSearch(object):
    def __init__(self, vectors):
        super().__init__()
        self.vectors = np.array(vectors)
        self.dim = self.vectors.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.vectors)

    def search(self, target_vector, topK=10):
        target_vector = np.asarray(target_vector)
        topK_distances, topK_indices = self.index.search(target_vector, topK)

        return topK_distances, topK_indices