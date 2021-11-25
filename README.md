<p align="center">
<img src="GRecX_LOGO_SQUARE.png" width="300"/>
</p>

# GRecX
An Efficient and Unified Benchmark for GNN-based Recommendation.

### Homepage and Documentation

+ Homepage: []()
+ Documentation: []()
+ Paper: [GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation](https://arxiv.org/pdf/2111.10342.pdf)


### Preliminary Comparison


#### LightGCN-Yelp dataset (featureless)

[comment]: <> (| Algo | nDCG@20 | recall@20 | precision@20 |)

[comment]: <> (| --- | --- | --- | --- | )

[comment]: <> (| NGCF | 0.04118 | 0.02302 | 0.05034 |)

[comment]: <> (| lightGCN| 0.05260 | 0.06397 | 0.02876 |)

[comment]: <> (| UltraGCN &#40;oc&#41; | 0.03408 | 0.04154 | 0.01928 |)

[comment]: <> (| our-UltraGCN | 0.03540 | --- | --- |)

[comment]: <> (Note that: oc means orignal code with negative_num=1  and negative_weight=1. )

* BCE-loss

[comment]: <> (| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |)

[comment]: <> (| --- | --- | --- | --- | --- |)

[comment]: <> (| MF| 0.031168 | 0.033510 | 0.037817 | 0.042061 &#40;epoch:1300&#41; |)

[comment]: <> (| our-lightGCN| 0.034872 | 0.037350 | 0.041520 | 0.045872 &#40;epoch:1300&#41; |)

| Algo | Precision@10 | Precision@20 | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| MF |  0.029597 | 0.025495 | 0.032733 | 0.056086 | 0.037332  | 0.045805 |
| NGCF | --- | --- | --- | --- | --- | --- |
| LightGCN | --- | --- | --- | --- | 0.037350 | 0.045872 |
| UltraGCN-single | 0.030652 |  0.026790 | 0.033913 | 0.058886 | 0.038576 | 0.047766 |
| UltraGCN | 0.03553 | 0.030230 | 0.039461 | 0.066791 | 'ndcg@10', 0.045463 | 0.05527 |
[('precision@10', 0.03553113553113766), ('precision@20', 0.030230832386000845), ('recall@10', 0.03946159749657537), ('recall@20', 0.06679137357842446), ('ndcg@10', 0.045463614610655706), ('ndcg@20', 0.055273164430104695)]

* BPR-loss

[comment]: <> (| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |)

[comment]: <> (| --- | --- | --- | --- | --- |)

[comment]: <> (| MF| 0.034672 | 0.037321 | 0.041864 | 0.046112 |)

[comment]: <> (| our-lightGCN| 0.040223 | 0.042649 | 0.047568 | 0.052569 &#40;epoch:760&#41; |)


| Algo | Precision@10 | Precision@20 | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| MF |  0.03009 | 0.026291 | 0.033211 | 0.057794 | 0.038203 | 0.047216 |
| NGCF | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.033544 | 0.028996 | 0.037277 | 0.064128 | 0.042907 | 0.052667 |
| UltraGCN-single | --- | --- | --- | --- | --- | --- |
| UltraGCN | --- | --- | --- | --- | --- | --- |

Note that "UltraGCN-single" uses loss with one negative sample and one negatvie loss weight

***

#### LightGCN-Gowalla dataset (featureless)

[comment]: <> (| Algo | nDCG@20 | recall@20 | precision@20 |)

[comment]: <> (| --- | --- | --- | --- | )

[comment]: <> (| NGCF | 0.11804 | 0.14375 | 0.04404 |)

[comment]: <> (| lightGCN| 0.15271 | 0.17801 | 0.05474 |)

[comment]: <> (| UltraGCN &#40;oc&#41; | 0.10846 | 0.12202 | 0.03826 |)

[comment]: <> (Note that: oc means orignal code with negative_num=1  and negative_weight=1.)


* BCE-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| --- | --- | --- | 0.1298 |
| our-lightGCN| --- | --- | --- | 0.1300 |


* BPR-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.116182 | 0.117339 | 0.123564 | 0.1400 |
| our-lightGCN| --- | --- | --- | 0.1485 |


#### LightGCN-Amazon-book dataset (featureless)

| Algo | nDCG@20 |
| --- | --- | 
| lightGCN| --- |



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