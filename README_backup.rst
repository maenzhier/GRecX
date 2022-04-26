

.. image:: https://raw.githubusercontent.com/maenzhier/GRecX/main/GRecX_LOGO_SQUARE.png
   :align: center
   :width: 300px


GRecX
=====

An Efficient and Unified Benchmark for GNN-based Recommendation.

Homepage and Documentation
--------------------------


* Homepage: `https://github.com/maenzhier/GRecX <https://github.com/maenzhier/GRecX>`_
* Paper: `GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation <https://arxiv.org/pdf/2111.10342.pdf>`_

Example Benchmark: Performance on Yelp and Gowalla with BPR Loss
----------------------------------------------------------------

Performance on Yelp with BPR Loss:


.. image:: https://raw.githubusercontent.com/maenzhier/GRecX/main/plots/bpr_yelp.png
   :align: center
   :width: 500px


Performance on Gowalla with BPR Loss:


.. image:: https://raw.githubusercontent.com/maenzhier/GRecX/main/plots/bpr_gowalla.png
   :align: center
   :width: 500px

Demo
----

We recommend you get started with some demos.


* `Matrix Factorization (MF) <demo/demo_mf.py>`_
* `MLP + MF <demo/demo_mf_fc.py>`_
* `NGCF <demo/demo_ngcf.py>`_
* `LightGCN <demo/demo_light_gcn.py>`_
* `UltraGCN <demo/demo_ultra_gcn.py>`_

Preliminary Comparison
----------------------

LightGCN-Yelp dataset (featureless)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* BCE-loss

.. list-table::
   :header-rows: 1

   * - Algo
     - Precision\@10
     - Precision\@20
     - Recall\@10
     - Recall\@20
     - nDCG\@10
     - nDCG\@20
   * - MF
     - 0.029597
     - 0.025495
     - 0.032733
     - 0.056086
     - 0.037332
     - 0.045805
   * - NGCF
     - 0.024713
     - 0.021893
     - 0.028251
     - 0.049611
     - 0.031357
     - 0.039549
   * - LightGCN
     - ---
     - ---
     - ---
     - ---
     - 0.037350
     - 0.045872
   * - UltraGCN-single
     - 0.030652
     - 0.026790
     - 0.033913
     - 0.058886
     - 0.038576
     - 0.047766
   * - UltraGCN
     - 0.03553
     - 0.030346
     - 0.039526
     - 0.067028
     - 0.045365
     - 0.055376



* BPR-loss

.. list-table::
   :header-rows: 1

   * - Algo
     - Precision\@10
     - Precision\@20
     - Recall\@10
     - Recall\@20
     - nDCG\@10
     - nDCG\@20
   * - MF
     - 0.031489
     - 0.027303
     - 0.034733
     - 0.060333
     - 0.040103
     - 0.049406
   * - NGCF
     - 0.030375
     - 0.026699
     - 0.034502
     - 0.059984
     - 0.038732
     - 0.048351
   * - LightGCN
     - 0.033544
     - 0.028996
     - 0.037277
     - 0.064128
     - 0.042907
     - 0.052667
   * - UltraGCN-single
     - ---
     - ---
     - ---
     - ---
     - ---
     - ---
   * - UltraGCN
     - ---
     - ---
     - ---
     - ---
     - ---
     - ---


Note that "UltraGCN-single" uses loss with one negative sample and one negatvie loss weight

Cite
----

If you use GRecX in a scientific publication, we would appreciate citations to the following paper:

.. code-block:: html

   @misc{cai2021grecx,
   title={GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation},
   author={Desheng Cai and Jun Hu and Shengsheng Qian and Quan Fang and Quan Zhao and Changsheng Xu},
   year={2021},
   eprint={2111.10342},
   archivePrefix={arXiv},
   primaryClass={cs.IR}
   }
