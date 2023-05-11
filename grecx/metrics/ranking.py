# coding=utf-8

import numpy as np
import ctypes
import os

def dcg_score(element_list):
    """
    Discounted Cumulative Gain (DCG)
    The definition of DCG can be found in this paper:
        Azzah Al-Maskari, Mark Sanderson, and Paul Clough. 2007.
        "The relationship between IR effectiveness measures and user satisfaction."

    Parameters:
        element_list - a list of ranks Ex: [5,4,2,2,1]

    Returns:
        score
    """
    score = 0.0
    for order, rel in enumerate(element_list):
        score += (np.power(2.0, rel) - 1.0) / np.log(order + 2)
        # if rel > 0:
        #     print("====")
        #     # score += math.log(2) / math.log((order+2))
        #     score += np.log(2) / np.log(order + 2)
    return score


def ndcg_score(reference, hypothesis):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG(hypothesis)/DCG(reference)

    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]

    Returns:
        ndcg_score  - normalized score
    """

    return dcg_score(hypothesis)/dcg_score(reference)


def load_c_ndcg_score():

    lib_name = "libranking.dll" if os.name == "nt" else "libranking.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)
    lib = ctypes.CDLL(lib_path)
    c_ndcg_score = lib.ndcg_score
    c_ndcg_score.restype = ctypes.c_double

    def ndcg_score(reference, hypothesis):

        reference = list(reference)
        hypothesis = list(hypothesis)

        num_rels = len(reference)
        rel_type = ctypes.c_int * num_rels

        return c_ndcg_score(
            rel_type(*reference),
            rel_type(*hypothesis),
            ctypes.c_int(num_rels)
        )
    return ndcg_score

try:
    ndcg_score = load_c_ndcg_score()
except:
    print("cannot load c_ndcg_score, use Python version instead")
    pass



def precision_score(reference, hypothesis):
    result = np.sum(hypothesis, dtype=np.float32)/len(hypothesis)
    return result


def recall_score(reference, hypothesis):
    result = np.sum(hypothesis, dtype=np.float32) / len(reference)
    return result
