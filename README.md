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

```python

from timex import ts_detrender

detrender = ts_detrender(unit='days')
TS_detrended = detrender.fit_transform(TS_data)
```


# **Crossectional feature extraction** (CFE)

Using features from
* catch22
* tsfresh
* kats
* antropy
* nolds
* cesium
* tsfel
* tsflex
* CNN/RNN based bottlenecks
* LLM-based bottlenecks??
* Custom functions: including custom shapelets
* Neurokit2 for EEG/ECG preprocessing
 
# **Timeseries distance matrices** (TDM):
Using 
* tslearn
* sktime
* aeon
* Custom:
  * log rank
  * distance correlation
  * wavelet embeddings

# Direct sparse coding of timeseries (DSC)

```timeseries -> low-ranking representation with SAEs```


# Clustering

## CFE/TDM/DSC -> clustering

Clustering follows the extraction of a distance matrix, either directly created using ```timex.tdm``` methods (i.e. a clustering algorithm with a pre-computed distance matrix), or 

As clustering methods following CFE, TDM or DSC we have 

* [SNN](https://github.com/felipeangelimvieira/SharedNearestNeighbors/tree/main)
* Sklearn clustering methods: OPTICS, k-means
* k-Medoid: basically rank-based k-Means
* HDBSCAN

## Latent class modeling


## TS specific

* [MPF](https://matrixprofile.docs.matrixprofile.org/examples/Hierarchical_Clustering_Accelerometer_Walk_Stand_etc.html)
* [TiCC](https://github.com/davidhallac/TICC)
* dtwclust
* [LCMM](https://cran.r-project.org/web/packages/lcmm/lcmm.pdf)
* Aeon - clustering methods
* TScluster - clustering methods


# Acknowledgments

This library could not have been built without the following libraries
* TSFel
* TSfresh
* Catch22
* Katz
* Nolds
* [pyEntropy](https://github.com/nikdon/pyEntropy)
* Aeon
* TSlearn
* TScluster
  
