<p align="center">
<img src="GRecX_LOGO_SQUARE.png" width="300"/>
</p>

# GRecX
A Fair Benchmark for GNN-based Recommendation

### Preliminary Comparison

***

#### DiffNet-Yelp dataset (featureless)

| Algo | nDCG@5 | nDCG@10 | nDCG@15 |
| --- | --- | --- | --- |
| MF| 0.158707 | 0.196456 |	0.218138 |
| Ours-MF | 0.166521 | 0.206430 | 0.230114 |

#### DiffNet-Flickr dataset (featureless)

| Algo | nDCG@5 | nDCG@10 | nDCG@15 |
| --- | --- | --- | --- |
| MF| 0.097722 | 0.107193 | 0.115850 |
| Ours-MF | 0.100690 | 0.110089 | 0.119592 |

***

#### LightGCN-Yelp dataset (featureless)

| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| NGCF | 0.04118 | 0.02302 | 0.05034 |
| lightGCN| 0.05260 | 0.06397 | 0.02876 |


| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| （original code） UltraGCN (negative_num=1) (negative_weight=1)| 0.03408 | 0.04154 | 0.01928 |
| ours-UltraGCN | 0.03540 |  |  |


* MF-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.031168 | 0.033510 | 0.037817 | 0.042061 (epoch:1300) |
| ours-lightGCN| 0.034872 | 0.037350 | 0.041520 | 0.045872 (epoch:1300) |
| Ours-MF | --- | --- | --- | --- |

* Softmax-loss (CL loss)

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.035439 | 0.037995 | 0.042330 | 0.046524 (eppch:2420) |
| ours-lightGCN| 0.039535 | 0.041681 | 0.046427 | 0.051316 (eppch:520) |
| Ours-MF | --- | --- | --- | --- |


* BPR-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.034672 | 0.037321 | 0.041864 | 0.046112 |
| ours-lightGCN| 0.040223 | 0.042649 | 0.047568 | 0.052489 (epoch:1540) |
| Ours-MF | --- | --- | --- | --- |

***

#### LightGCN-Gowalla dataset (featureless)

| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| NGCF | 0.11804 | 0.14375 | 0.04404 |
| lightGCN| 0.15271 | 0.17801 | 0.05474 |


| Algo | nDCG@20 | recall@20 | precision@20 |
| --- | --- | --- | --- | 
| （original code） UltraGCN (negative_num=1) (negative_weight=1)| 0.10846 | 0.12202 | 0.03826 |
| ours-UltraGCN | |  |  |

* MF-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| - | 0.033510 | 0.037817 | 0.042061 (epoch:1300) |
| ours-lightGCN| - | 0.037350 | 0.041520 | 0.045872 (epoch:1300) |
| Ours-MF | --- | --- | --- | --- |

* Softmax-loss (CL loss)

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.115557 | 0.116847 | 0.123106 | 0.129477 (epoch:1920) |
| ours-lightGCN| --- | 0.041681 | 0.046427 | 0.051316 (epoch:520) |
| Ours-MF | --- | --- | --- | --- |


* BPR-loss

| Algo | nDCG@5 | nDCG@10 | nDCG@15 | nDCG@20 |
| --- | --- | --- | --- | --- |
| MF| 0.116182 | 0.117339 | 0.123564 | 0.129682 (epoch:1800) |
| ours-lightGCN| --- | 0.042649 | 0.047568 | 0.052489 (epoch:1540) |
| Ours-MF | --- | --- | --- | --- |


#### LightGCN-Amazon-book dataset (featureless)

| Algo | nDCG@20 |
| --- | --- | 
| lightGCN| --- |
