#coding=utf-8
"""ranking_measures.measures: a rank-ordering evaluation package for Python
=========================================================================

ranking_eval is a set of common ranking algorithms such as:
*dcg
*ndcg
*precision
*precision_k
*average_precision
*rankdcg

rankdcg is a new measure and it is described in this paper:
RankDCG is described in this paper:
"RankDCG: Rank-Ordering Evaluation Measure," Denys Katerenchuk, Andrew Rosenberg
http://www.dk-lab.com/wp-content/uploads/2014/07/RankDCG.pdf

"""

__author__ = "Denys Katerenchuk, The Graduate Center, CUNY"

__license__ = """The MIT License (MIT)

Copyright (c) [2015] [Denys Katerenchuk]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


__version__ = "1.0.1"

import math


def find_dcg(element_list):
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
    for order, rank in enumerate(element_list):
        score += float(rank)/math.log((order+2))
    return score


def find_ndcg(reference, hypothesis):
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

    return find_dcg(hypothesis)/find_dcg(reference)


def find_precision_k(reference, hypothesis, k):
    """
    Precision at k
    This measure is similar to precision but takes into account first k elements

    Description reference:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.

    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]
        k           - a number of top element to consider

    Returns:
        precision   - a score
    """
    precision = 0.0
    relevant = 0.0
    for i, value in enumerate(hypothesis[:k]):
        if value == reference[i]:
            relevant += 1.0
    precision = relevant/k

    return precision


def find_precision(reference, hypothesis):
    """
    Presision

    Description reference:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.

    Parameters:
        reference    - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis   - a proposed ordering Ex: [5,2,2,3,1]

    Returns:
        precision    - a score
    """

    return find_precision_k(reference, hypothesis, len(reference))


def find_average_precision(reference, hypothesis):
    """
    Average Precision

    Description reference:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.

    Parameters:
        reference    - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis   - a proposed ordering Ex: [5,2,2,3,1]

    Returns:
        precision    - a score
    """

    s_total = sum([find_precision_k(reference, hypothesis, k+1) for k in \
                   range(len(reference))])

    return s_total/len(reference)


def _order_lists(reference, hypothesis):
    """
    Maps and orders both lists. Ex: ref:[2,5,1,1] and hyp:[2,2,3,1] =>
                                     ref:[5,2,1,1] and hyp:[1,2,5,1]
    """
    pair_ref_list = sorted([x for x in enumerate(reference)], key=lambda x: x[1])
    mapped_hyp_list = [hypothesis[x[0]] for x in pair_ref_list]

    return [x[1] for x in pair_ref_list], mapped_hyp_list


def find_rankdcg(reference, hypothesis):
    """
    RankDCG - modified version of well known DCG measure.
    This measure was designed to work with ties and non-normal rank distribution.

    Description reference:
    RankDCG is described in this paper:
    "RankDCG: Rank-Ordering Evaluation Measure," Denys Katerenchuk, Andrew Rosenberg
    http://www.dk-lab.com/wp-content/uploads/2014/07/RankDCG.pdf

    Cost function: relative_rank(i)/reversed_rel_rank(i)

    Params:
        reference_list - list: original list with correct user ranks
        hypothesis_list - list: predicted user ranks

    Returns:
        score - double: evaluation score
    """

    #Ordering to avoid bias with majority class output
    reference_list, hypothesis_list = _order_lists(reference, hypothesis)

    ordered_list = reference_list[:] # creating ordered list
    ordered_list.sort(reverse=True)

    high_rank = float(len(set(reference_list))) # max rank
    reverse_rank = 1.0            # min score (reversed rank)
    relative_rank_list = [high_rank]
    reverse_rank_list = [reverse_rank]

    for index, rank in enumerate(ordered_list[:-1]):
        if ordered_list[index+1] != rank:
            high_rank -= 1.0
            reverse_rank += 1.0
        relative_rank_list.append(high_rank)
        reverse_rank_list.append(reverse_rank)

    # map real rank to relative rank
    reference_pair_list = [x for x in enumerate(reference_list)]
    sorted_reference_pairs = sorted(reference_pair_list, key=lambda p: p[1], \
                                    reverse=True)
    rel_rank_reference_list = [0] * len(reference_list)
    for position, rel_rank in enumerate(relative_rank_list):
        rel_rank_reference_list[sorted_reference_pairs[position][0]] = rel_rank

    # computing max/min values (needed for normalization)
    max_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(relative_rank_list)])
    min_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(reversed(relative_rank_list))])

    # computing and mapping hypothesis to reference
    hypothesis_pair_list = [x for x in enumerate(hypothesis_list)]
    sorted_hypothesis_pairs = sorted(hypothesis_pair_list, \
                                     key=lambda p: p[1], reverse=True)
    eval_score = sum([rel_rank_reference_list[pair[0]] / reverse_rank_list[index] \
                      for index, pair in enumerate(sorted_hypothesis_pairs)])

    return (eval_score - min_score) / (max_score - min_score)
