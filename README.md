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



# Clustering

## CFE/TDM/DSC -> clustering

Clustering follows the extraction of a distance matrix, either directly created using ```timex.tdm``` methods (i.e. a clustering algorithm with a pre-computed distance matrix), or

As clustering methods following CFE, TDM or DSC we have

* [SNN](https://github.com/felipeangelimvieira/SharedNearestNeighbors/tree/main)
* Sklearn clustering methods: OPTICS, k-means
* k-Medoids; PAM, CLARANS, etc.
* k-Median; basically rank-based k-means
* HDBSCAN

## Latent class modeling

GBTM, GMM

## TS specific

* [MPF](https://matrixprofile.docs.matrixprofile.org/examples/Hierarchical_Clustering_Accelerometer_Walk_Stand_etc.html)
* [TiCC](https://github.com/davidhallac/TICC)
* dtwclust
* [LCMM](https://cran.r-project.org/web/packages/lcmm/lcmm.pdf)
* Aeon - clustering methods
* TScluster - clustering methods

# New methods

## Greedy

It is relatively easy to come up with greedy algorithms to find the best separation of timeseries; for example

**Algorithm timex-greedy-cluster-1**

1. **Init**; randomly select $P$ timeseries (which should be small fraction of the total); determine the $K$ exemplars/groups using MSM (or DTW etc.) and e.g. Affinity Propagation
2. **Model**; regress a model on each group.
3. **Expand**; expand each group by taking the top-$N$ matches with the model
* Repeat 2->3->2->3 until the number of series is exhausted
4. **Remix**; shift the timeseries with the highest residuals to the group with the lowest model residual
5. **Model**; regress a model on each group
* Repeat 4->5->4->5 until all timeseries are in the group with the lowest residual

**Algorithm timex-greedy-cluster-2**

1. **Model**; regress a model on all series
2. **Split**; divide the series in groups with net positive and net negative sum of the residual errors, select the the top-$K$ largest positive/negative errors as two seperate clusters
3. **Model**; regress a model on the remaining timeseries
* Repeated 2->3 until all timeseries in the group are assigned to a cluster


## HMM-based

Ingredients; ```pomegranate```, ```hmmlearn```
**Algorithm timex-hmm-discrete**
1. concatenate all timeseries
2. symbolizer (e.g. SAX)
3. HMM with K states, initialized with uniform state/emission probas
4. get states per timeseries
5. feature:
 * extract Bag-of-States; counts per state, count of state-state transitions, etc. -> feature-based clustering
 * perform DTW with custom cost function; ```python def cost(a, b): return 0 if a == b else 1``` -> TSKMeans etc.

**Algorithm timex-hmm-continuous**
1. concatenate all timeseries
2. continuous HMM with K states, initialized with uniform state/emission probas
3. get states per timeseries
4. feature:
 * extract Bag-of-States; counts per state, ount of state-state transitions -> feature-based clustering
 * perform DTW with custom cost function; ```python def cost(a, b): return 0 if a == b else 1``` -> TSKMeans etc.

For multivariate, reduce with PCA or CCA.


# Benchmarking

We use the [UCR time series](https://www.cs.ucr.edu/%7Eeamonn/time_series_data/) archive as a univariate ts benchmark. We allow for "heterogenisation" through random pruning of the time series.

```python

from timex import benchmark
from timex.clustering import TSKmeans, TSKernelKmeans

UCRBench = benchmark(which="UCR", random_pruning=True, seed=42, hyper_parameters=None, multivariate=False)
Methods = [TSKmeans, TSKernelKmeans]

res = UCRBench.go(Methods)
```

We use the [MTS collection](https://github.com/MTS-BenchMark/MvTS) as a multivariate ts benchmark. Again, we allow for "heterogenisation" through random pruning of the time series.


# Tip

For more comprehensive model-based clustering techniques we refer the user to ```mixtools```, ```flexmix```, ```mclust```,  ```lcmm``` and ```latrend```, all R-packages.

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
* torchaudio
