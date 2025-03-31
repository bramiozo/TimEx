# TimEx
Repository for extracting time-series features and clusters

# **Normalisation**


# **Interpolation**


# **Smoothing** 


# **Crossectional feature extraction** (CFE)
Using features  from
* catch22
* tsfresh
* kats
* antropy
* nolds
* cesium
* CNN/RNN based bottlenecks
* LLM-based bottlenecks
* Custom functions:
  * Added:
  * wavelet
  * FFT
  * Mann-Kendall
* Neurokit2
 
 TODO:  use tsflex to improve performance of feature  extraction.

# **Timeseries distance matrices** (TDM):
Using 
* tslearn
* sktime
* Custom:
  * log rank
  * distance correlation
  * wavelet embeddings

# Direct sparse coding of timeseries (DSC)

```timeseries -> low-ranking representation```


# Cluster
```CFE/TDM/DSC -> clustering```

## TS specific
* [MPF](https://matrixprofile.docs.matrixprofile.org/examples/Hierarchical_Clustering_Accelerometer_Walk_Stand_etc.html)
* [TiCC](https://github.com/davidhallac/TICC)
* dtwclust
* [LCMM](https://cran.r-project.org/web/packages/lcmm/lcmm.pdf)

## Cross-sectional
* [SNN](https://github.com/felipeangelimvieira/SharedNearestNeighbors/tree/main)
* Sklearn clustering methods
* HDBSCAN



# Examples
https://www.kaggle.com/code/slythe/feature-extraction-tsflex-catch22


# Acknowledgments

* https://github.com/nikdon/pyEntropy

# Sources
* https://link.springer.com/article/10.1186/s12938-023-01075-1
  
