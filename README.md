<p align="center">
<img src="GRecX_LOGO_SQUARE.png" width="300"/>
</p>

# GRecX
A Fair Benchmark for GNN-based Recommendation

### Homepage and Documentation

+ Homepage: []()
+ Documentation: []() ([中文版]())
+ Paper: [GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation](https://arxiv.org/pdf/2111.10342.pdf)


### Preliminary Comparison


#### LightGCN-Yelp dataset (featureless)

| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| NGCF | 0.04118 | 0.02302 | 0.05034 |
| lightGCN| 0.05260 | 0.06397 | 0.02876 |
| UltraGCN (oc) | 0.03408 | 0.04154 | 0.01928 |
| our-UltraGCN | 0.03540 | --- | --- |

Note that: oc means orignal code with negative_num=1  and negative_weight=1. 

* MF-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.031168 | 0.033510 | 0.037817 | 0.042061 (epoch:1300) |
| our-lightGCN| 0.034872 | 0.037350 | 0.041520 | 0.045872 (epoch:1300) |

* BPR-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.034672 | 0.037321 | 0.041864 | 0.046112 |
| our-lightGCN| 0.040223 | 0.042649 | 0.047568 | 0.052489 (epoch:1540) |

***

#### LightGCN-Gowalla dataset (featureless)

| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| NGCF | 0.11804 | 0.14375 | 0.04404 |
| lightGCN| 0.15271 | 0.17801 | 0.05474 |
| UltraGCN (oc) | 0.10846 | 0.12202 | 0.03826 |

Note that: oc means orignal code with negative_num=1  and negative_weight=1.


* MF-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| - | 0.033510 | 0.037817 | 0.042061 (epoch:1300) |
| our-lightGCN| - | 0.037350 | 0.041520 | 0.045872 (epoch:1300) |


* BPR-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.116182 | 0.117339 | 0.123564 | 0.129682 (epoch:1800) |
| our-lightGCN| --- | 0.042649 | 0.047568 | 0.052489 (epoch:1540) |


#### LightGCN-Amazon-book dataset (featureless)




### Cite

If you use GRecX in a scientific publication, we would appreciate citations to the following paper:

```html
@misc{cai2021grecx,
title={GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation},
author={Desheng Cai and Jun Hu and Shengsheng Qian and Quan Fang and Quan Zhao and Changsheng Xu},
year={2021},
eprint={2111.10342},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```