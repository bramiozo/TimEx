# TimEx
Repository for extracting time-series features and clusters for heterogeneous multivariate data. This library is not meant for end-to-end timeseries-classification or segmentation.

# **Standardisation**

```python

from timex import ts_scaler

scaler = ts_scaler(by='id', how='standardisation')

TS_standardised = scaler.fit_transform(TS_data)
```

# **Interpolation**


```python

from timex import ts_interp

interpol = ts_interp(how='mono_cube', unit='days', resolution=1, knots=10, relative=True)

TS_interp = interpol.fit_transform(TS_data)
```


# **Smoothing** 

```python

from timex import ts_smoother

smoother = ts_smoother(unit='days', resolution=1, window=10, how='gaussian')

TS_smoothed = smoother.fit_transform(TS_data)
```


# **Detrending**



# **Crossectional feature extraction** (CFE)
Using features  from
* catch22
* tsfresh
* kats
* antropy
* nolds
* cesium
* CNN/RNN based bottlenecks
* LLM-based bottlenecks??
* Custom functions:
  * Added:
  * wavelet
  * FFT
  * Mann-Kendall
* Neurokit2 for EEG/ECG preprocessing
 
*TODO:*  use tsflex to improve performance of feature  extraction.

# **Timeseries distance matrices** (TDM):
Using 
* tslearn
* sktime
* Custom:
  * log rank
  * distance correlation
  * wavelet embeddings

# Direct sparse coding of timeseries (DSC)

```timeseries -> low-ranking representation with SAEs```


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
  
