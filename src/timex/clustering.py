from statistics import stdev
# exposes clustering methods from tslearn, aeon, tscluster, deeptime, and pypots
# [ ] tslearn
# [ ] aeon
# [ ] tscluster
# [ ] deeptime
# [ ] pypots

from asyncio import SelectorEventLoop
from timex import extractor
from timex import preprocessing

from pandas import DataFrame
from numpy import ndarray

from tslearn import clustering as tslearn_clustering

from sklearn.cluster import GaussianMixture, BayesianGaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin

from scipy.stats import entropy
# distance based clustering
class TSKMeans:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans
    def __init__(self, backend="tslearn", n_clusters=3, **kwargs):
        if backend == "tslearn":
            self.clusterer = tslearn_clustering.TimeSeriesKMeans(
                n_clusters=n_clusters, **kwargs
            )

    def fit(self, ts):
        pass

# distance based clustering
class TSKernelKMeans:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html#tslearn.clustering.KernelKMeans
    def __init__(self):
        pass

    def fit(self, ts):
        pass

# distance based clustering
class TSKShape:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html#tslearn.clustering.KShape
    def __init__(self):
        pass

    def fit(self, ts):
        pass


class CrossSectionalClustering(BaseEstimator, ClusterMixin):
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.CrossSectionalClustering.html#tslearn.clustering.CrossSectionalClustering
    def __init__(
        self,
        smoothing: bool=False,
        smoothing_type: Literal["gaussian_kernel", "rolling_mean"],
        smoothing_window_size: int = 10,
        interpolation: bool = False,
        interpolation_resolution: int = 30,
        interpolation_keep_init: bool = False,
        min_measurements_per_id: int = 10,
        min_time: Optional[int] = None,
        max_time: Optional[int] = None,
        n_clusters: int = 3,
        random_state: int = None,
        extractors: List[
            Literal[
                "custom",
                "tsfresh",
                "catch22",
                "cesium",
                "antropy",
                "nolds",
                "katz",
                "tsfel",
            ]
        ] = None,
        clustering_algorithm: Literal["gmm", "bgmm", "kmeans", "optics", "hdbscan", "spectral", "hierarchical"],
        normalise_timeseries: Optional[Literal["bulk", "group"]] = None,
        normalisation_method: Optional[Literal["standard", "minmax"]] = None,
        id_column: str,
        time_column: str,
        feature_columns: str,
        add_ts_meta: bool = False,
        multivariate_join: str = 'inner'
    ):
        """
        Initialize the CrossSectionalClustering class.

        We assume that we have multiple time series per id.
        We assume that the time_column is numeric and represents the unit of time.
        We assume that the feature_columns are numeric and represent the features of the time series

        Args:
            smoothing (bool, optional): Whether to apply smoothing to the time series. Defaults to False.
            smoothing_type (Literal["gaussian_kernel", "rolling_mean"], optional): The type of smoothing to apply. Defaults to "gaussian_kernel".
            smoothing_window_size (int, optional): The window size for smoothing. Defaults to 10.
            interpolation (bool, optional): Whether to interpolate the time series. Defaults to False.
            interpolation_resolution (int, optional): The resolution for interpolation. Defaults to 30.
            min_measurements_per_id (int, optional): The minimum number of measurements per time series. Defaults to 10.
            min_time (int, optional): The minimum time for time series. Defaults to None.
            max_time (int, optional): The maximum time for time series. Defaults to None.
            n_clusters (int, optional): The number of clusters to form. Defaults to 3.
            random_state (int, optional): The random state for reproducibility. Defaults to None.
            extractors (List[Literal["custom", "tsfresh", "catch22", "cesium", "antropy", "nolds", "katz", "tsfel"]], optional): The feature extractors to use. Defaults to None.
            normalise_timeseries (bool, optional): Whether to normalise the time series. Defaults to False.
            normalisation_method (Literal["zscore", "minmax", "robust"], optional): The method for normalising the time series. Defaults to "zscore".
            id_column (str): The column to use for identifying time series.
            time_column (str): The column to use for time information.
            feature_columns (List[str]): The columns to use for feature extraction.
            add_ts_meta: bool = False: Whether to add time series metadata to the output. Defaults to False.
            multivariate_join: str='inner' : How to join the cross-sectional results over the variables
        """

        # TODO: make compatible with multiple features

        extractors = [e.lower() for e in extractors]

        assert(max_time<= min_time), "Max time has to be equal or smaller than min time"

        self.smoothing = smoothing
        self.smoothing_type = smoothing_type
        self.smoothing_window_size = smoothing_window_size
        self.interpolation = interpolation
        self.interpolation_resolution = interpolation_resolution
        self.interpolation_keep_init = interpolation_keep_init
        self.min_measurements_per_id = min_measurements_per_id
        self.min_time = min_time
        self.max_time = max_time
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.extractors = extractors
        self.normalise_timeseries = normalise_timeseries
        self.normalisation_method = normalisation_method
        self.id_column = id_column
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.add_ts_meta = add_ts_meta
        self.multivariate_join = multivariate_join

        if clustering_algorithm == "gmm":
            self.clustering_algorithm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        elif clustering_algorithm == "bgmm":
            self.clustering_algorithm = BayesianGaussianMixture(n_components=n_clusters, random_state=random_state)
        else:
            raise ValueError("Invalid clustering algorithm")

        self.extractors = {
            'custom_features': 'custom' in extractors,
            'tsfel_features': 'tsfel' in extractors,
            'catch22_features': 'catch22' in extractors,
            'tsfresh_features': 'tsfresh' in extractors,
            'cesium_features': 'cesium' in extractors,
            'antropy_features': 'antropy' in extractors,
            'nolds_features': 'nolds' in extractors,
            'katz_features': 'katz' in extractors,
        }

    def fit(self, tsdf: DataFrame, y = None):
        id_kwargs = {
            "id_col": self.id_column,
            "time_col": self.time_column,
        }

        ts_cross_combined = DataFrame()
        ts_cross = DataFrame()
        durations = None

        # TODO: (OPTION) for cross-channel features combine into combi-channel, before cross-sect extraction

        for var_num, _feature_column in enumerate(self.val_cols):
            _feature_column = self.val_cols[0]
            id_kwargs['val_col'] = _feature_column
            ts = tsdf[*id_kwargs.values()]
            ts = preprocessing.get_filtered_df(ts, min_time=self.min_time, min_measurements=self.min_measurements_per_id, **id_kwargs.pop(key="val_col"))

            # get meta data
            if self.add_ts_meta:
                # per id extract number of measurements, max_time, time_per_measurement, time_var
                #
                ts_meta = {
                    'NumMeas': ts.groupby(self.id_column).size(),
                    'MaxTime': ts.groupby(self.id_column)[self.time_column].max(),
                    'MeanTimeDiff': ts.groupby(self.id_column)[self.time_column].diff().mean(),
                    'RelTimeDiffVar': ts.groupby(self.id_column)[self.time_column].diff().std() / ts.groupby(self.id_column)[self.time_column].diff().mean(),
                    'TimeStdev': ts.groupby(self.id_column)[self.time_column].std()
                }


            if self.interpolation:
                ts = preprocessing.get_interpolated(ts,
                    time_res=self.interpolation_resolution,
                    max_time=self.max_time,
                    keep_t0_value=self.interpolation_keep_init,
                    time_before=self.time_before,
                    df_out=True,
                    **id_kwargs)

            if self.smoothing:
                if self.smoothing_type == 'gaussian_kernel':
                    ts = preprocessing.get_smoothed_gaussian_kernel(ts, window=self.smoothing_window_size, Nskip=3, df_out=True, **id_kwargs)
                elif self.smoothing_type == 'savgol':
                    ts = preprocessing.get_smoothed_savgol(ts, window=self.smoothing_window_size, polyorder=self.smoothing_polyorder, df_out=True, **id_kwargs)
                elif self.smoothing_type == 'rolling_mean':
                    ts = preprocessing.get_smoothed_rolling_mean(ts, window=self.smoothing_window_size, df_out=True, **id_kwargs)
                else:
                    raise ValueError("Invalid smoothing type")

            if self.normalise_timeseries:
                ts = preprocessing.normalise_ts(ts, scaler=self.normalisation_method, df_out=True, id_kwargs**)

            # extract cross sectional
            ts_cross, durations = extractor.get_crossectional(tsdf, **extractors, **id_cols)

            if self.add_ts_meta:
                for k, v in ts_meta.items():
                    v = v.set_index(self.id_column)
                    v.columns = [f'Meta_{k}']
                    ts_cross = ts_cross.join(v, how='inner')

            # add _feature_column as prefix
            #
            ts_cross.columns = [f"{_feature_column}_{c}" for c in ts_cross.columns]

            if var_num==0:
                ts_cross_combined = ts_cross
            if var_num>0:
                ts_cross_combined = ts_cross_combined.join(ts_cross, how=self.multivariate_join)

        # TODO: (OPTION) for cross-channel features combine cross-sect features a posteriori
        #

        self.clustering_algorithm.fit(ts_cross)
        self.durations = durations

    def predict(self, ts: DataFrame):
        return self.clustering_algorithm.predict(ts)
