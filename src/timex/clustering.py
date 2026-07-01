import logging
import time

# exposes clustering methods from tslearn, aeon, tscluster, deeptime, and pypots
# [ ] tslearn
# [ ] aeon
# [ ] tscluster
# [ ] deeptime
# [ ] pypots
from asyncio import SelectorEventLoop

# from statistics import stdev
# from tabnanny import verbose
from typing import Any, List, Literal, Optional

import miceforest as mf
import numpy as np
from numpy import inf, isnan, nan, ndarray
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
    OPTICS,
    DBSCAN,
)
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from umap import UMAP

from timex import extractor, preprocessing

# adding logger
#
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # This will output to the notebook cell
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Suppress verbose libraries
logging.getLogger("umap").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


VALID_SMOOTHING_METHODS = {
    "gaussian_kernel",
    "gaussian_kernel_simple",
    "rolling_mean",
    "box_kernel",
}


def build_imputer(
    imputation_method: str,
    imputation_kwargs: dict[str, Any] | None,
    random_state: int,
    verbose: bool,
):
    """Build and return an imputer instance (or miceforest sentinel)."""
    imputation_kwargs = imputation_kwargs or {}

    if imputation_method == "knn":
        knn_kwargs = {"n_neighbors": 20}
        knn_kwargs.update(imputation_kwargs)
        return KNNImputer(**knn_kwargs)

    if imputation_method == "iterative":
        iterative_kwargs = {
            "max_iter": 10,
            "random_state": random_state,
            "verbose": verbose,
        }
        iterative_kwargs.update(imputation_kwargs)
        return IterativeImputer(**iterative_kwargs)

    if imputation_method == "miceforest":
        return "mice"

    if imputation_method in {"mean", "median", "most_frequent", "constant"}:
        simple_kwargs = {"strategy": imputation_method}
        if imputation_method == "constant":
            simple_kwargs["fill_value"] = 0
        simple_kwargs.update(imputation_kwargs)
        return SimpleImputer(**simple_kwargs)

    raise ValueError("Invalid imputation method")


def apply_interpolation(
    ts: DataFrame,
    id_kwargs: dict[str, str],
    interpolation_resolution: int,
    max_time: Optional[int],
    interpolation_keep_init: bool,
    interpolation_kwargs: dict[str, Any] | None = None,
) -> DataFrame:
    """Apply interpolation to a long-format time-series DataFrame."""
    _interpolation_kwargs = dict(interpolation_kwargs or {})
    return preprocessing.get_interpolated(
        ts,
        time_res=interpolation_resolution,
        max_time=max_time,
        keep_t0_value=interpolation_keep_init,
        df_out=True,
        **id_kwargs,
        **_interpolation_kwargs,
    )


def apply_smoothing(
    ts: DataFrame,
    id_kwargs: dict[str, str],
    smoothing_type: Literal[
        "gaussian_kernel", "gaussian_kernel_simple", "box_kernel", "rolling_mean"
    ],
    smoothing_window_size: int,
    n_skip: int,
    smoothing_kwargs: dict[str, Any] | None = None,
) -> DataFrame:
    """Apply smoothing to a long-format time-series DataFrame."""
    if smoothing_type not in VALID_SMOOTHING_METHODS:
        raise ValueError("Invalid smoothing type")

    _smoothing_kwargs = dict(smoothing_kwargs or {})
    return preprocessing.get_smoothed(
        ts_df=ts,
        window=smoothing_window_size,
        Nskip=n_skip,
        df_out=True,
        smoothing_method=smoothing_type,
        **id_kwargs,
        **_smoothing_kwargs,
    )


def impute_cross_sectional(
    ts_cross_combined: DataFrame,
    imputer: Any,
    imputation_kwargs: dict[str, Any] | None,
) -> DataFrame:
    """Impute missing values for cross-sectional features."""
    imputation_kwargs = dict(imputation_kwargs or {})

    if imputer == "mice":
        mice_iterations = imputation_kwargs.pop("mice_iterations", 10)
        n_estimators = imputation_kwargs.pop("n_estimators", 50)

        mice_kwargs = {
            "save_all_iterations": True,
            "random_state": 100,
            "num_datasets": 1,
        }
        mice_kwargs.update(imputation_kwargs)

        imp_kernel = mf.ImputationKernel(ts_cross_combined, **mice_kwargs)
        imp_kernel.mice(mice_iterations, n_estimators=n_estimators)
        return imp_kernel.complete_data()

    # Some sklearn imputers can effectively drop all-missing columns in output.
    # Keep shape stable by imputing only columns with >=1 observed value, then
    # restoring all-missing columns with 0.0.
    non_empty_cols = ts_cross_combined.columns[~ts_cross_combined.isna().all(axis=0)]
    empty_cols = ts_cross_combined.columns[ts_cross_combined.isna().all(axis=0)]

    out = ts_cross_combined.copy()

    if len(non_empty_cols) > 0:
        transformed = imputer.fit_transform(ts_cross_combined[non_empty_cols])
        out.loc[:, non_empty_cols] = transformed

    if len(empty_cols) > 0:
        out.loc[:, empty_cols] = 0.0

    return out


def preprocess_timeseries_feature(
    ts: DataFrame,
    *,
    id_column: str,
    time_column: str,
    feature_column: str,
    min_measurements_per_id: int = 1,
    min_time: Optional[int] = None,
    max_time: Optional[int] = None,
    interpolation: bool = False,
    interpolation_resolution: int = 1,
    interpolation_keep_init: bool = False,
    interpolation_kwargs: dict[str, Any] | None = None,
    smoothing: bool = False,
    smoothing_type: Literal[
        "gaussian_kernel", "gaussian_kernel_simple", "box_kernel", "rolling_mean"
    ] = "gaussian_kernel",
    smoothing_window_size: int = 4,
    n_skip: int = 1,
    smoothing_kwargs: dict[str, Any] | None = None,
    analysis_resolution: int = 1,
    dropna_before_normalisation: bool = False,
    normalise_timeseries: Optional[Literal["bulk", "group"]] = None,
    normalisation_method: Optional[Literal["standard", "minmax"]] = "standard",
    return_details: bool = False,
):
    """Shared preprocessing pipeline for a single feature column.

    Used by both CrossSectionalClustering and TSDistanceBasedClustering.
    """
    out = ts[[id_column, time_column, feature_column]].copy()
    stages: dict[str, DataFrame] = {}
    timings: dict[str, float] = {
        "filter": 0.0,
        "interpolation": 0.0,
        "smoothing": 0.0,
        "analysis_selection": 0.0,
        "prune_nans": 0.0,
        "normalization": 0.0,
    }

    if max_time is not None and min_time is not None and max_time < min_time:
        raise ValueError("max_time must be equal or greater than min_time")
    if analysis_resolution % interpolation_resolution != 0:
        raise ValueError(
            "Analysis resolution must be a multiple of interpolation resolution"
        )

    filter_start = time.perf_counter()
    if max_time is not None:
        out = out.loc[out[time_column] <= max_time]

    if min_time is not None:
        max_per_id = out.groupby(id_column)[time_column].max()
        keep_ids = max_per_id[max_per_id >= min_time].index
        out = out.loc[out[id_column].isin(keep_ids)]

    if min_measurements_per_id is not None and min_measurements_per_id > 1:
        counts = out.groupby(id_column).size()
        keep_ids = counts[counts >= min_measurements_per_id].index
        out = out.loc[out[id_column].isin(keep_ids)]
    timings["filter"] = time.perf_counter() - filter_start
    stages["filtered"] = out.copy()

    id_kwargs = {
        "id_col": id_column,
        "time_col": time_column,
        "val_col": feature_column,
    }

    if interpolation:
        interp_start = time.perf_counter()
        out = apply_interpolation(
            ts=out,
            id_kwargs=id_kwargs,
            interpolation_resolution=interpolation_resolution,
            max_time=max_time,
            interpolation_keep_init=interpolation_keep_init,
            interpolation_kwargs=interpolation_kwargs,
        )
        timings["interpolation"] = time.perf_counter() - interp_start
        stages["interpolated"] = out.copy()

    if smoothing:
        smooth_start = time.perf_counter()
        out = apply_smoothing(
            ts=out,
            id_kwargs=id_kwargs,
            smoothing_type=smoothing_type,
            smoothing_window_size=smoothing_window_size,
            n_skip=n_skip,
            smoothing_kwargs=smoothing_kwargs,
        )
        timings["smoothing"] = time.perf_counter() - smooth_start
        stages["smoothed"] = out.copy()

    if analysis_resolution != interpolation_resolution:
        sel_start = time.perf_counter()
        step = analysis_resolution // interpolation_resolution
        out = out.groupby(id_column, group_keys=False).apply(
            lambda x: x.sort_values(time_column).iloc[::step]
        )
        timings["analysis_selection"] = time.perf_counter() - sel_start
        stages["analysis_selected"] = out.copy()

    if dropna_before_normalisation:
        prune_start = time.perf_counter()
        out = out.dropna(subset=[feature_column])
        timings["prune_nans"] = time.perf_counter() - prune_start
        stages["dropna"] = out.copy()

    if normalise_timeseries == "group":
        norm_start = time.perf_counter()
        out = preprocessing.normalise_ts(
            out,
            id_col=id_column,
            time_col=time_column,
            val_col=feature_column,
            scaler=normalisation_method,
            df_out=True,
        )
        timings["normalization"] = time.perf_counter() - norm_start
        stages["normalised"] = out.copy()
    elif normalise_timeseries == "bulk":
        norm_start = time.perf_counter()
        if normalisation_method == "standard":
            scaler = StandardScaler()
        elif normalisation_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid normalisation_method")
        out[feature_column] = scaler.fit_transform(out[[feature_column]])
        timings["normalization"] = time.perf_counter() - norm_start
        stages["normalised"] = out.copy()

    if return_details:
        return {"ts": out, "stages": stages, "timings": timings}
    return out


class CrossSectionalClustering(BaseEstimator, ClusterMixin):
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.CrossSectionalClustering.html#tslearn.clustering.CrossSectionalClustering
    def __init__(
        self,
        smoothing: bool = False,
        smoothing_type: Literal[
            "gaussian_kernel", "gaussian_kernel_simple", "box_kernel", "rolling_mean"
        ] = "gaussian_kernel",
        smoothing_window_size: int = 4,
        n_skip: int = 1,
        interpolation: bool = False,
        interpolation_resolution: int = 1,
        interpolation_keep_init: bool = False,
        interpolation_kwargs: dict[str, Any] | None = None,
        analysis_resolution: int = 90,
        min_measurements_per_id: int = 10,
        min_time: Optional[int] = None,
        max_time: Optional[int] = None,
        n_clusters: int = 3,
        random_state: int = 42,
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
        ] = ["custom"],
        clustering_algorithm: Literal[
            "gmm", "bgmm", "kmeans", "optics", "hdbscan", "spectral", "hierarchical"
        ] = "gmm",
        cluster_kwargs: dict[str, Any] | None = None,
        normalise_timeseries: Optional[Literal["bulk", "group"]] = "group",
        normalisation_method: Optional[Literal["standard", "minmax"]] = "standard",
        id_column: str = "id",
        time_column: str = "time",
        feature_columns: List[str] = None,
        add_ts_meta: bool = False,
        multivariate_join: str = "inner",
        imputation_method: Literal[
            "knn",
            "iterative",
            "miceforest",
            "mean",
            "median",
            "most_frequent",
            "constant",
        ] = "knn",
        imputation_kwargs: dict[str, Any] | None = None,
        smoothing_kwargs: dict[str, Any] | None = None,
        cross_standardisation: bool = True,
        max_cross_missingness: float = 0.75,
        verbose: bool = False,
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
            interpolation_resolution (int, optional): The resolution for interpolation. Defaults to 1
            analysis_resolution (int): The resolution used to perform the feature aggregations. Defaults to 30
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
            multivariate_join: str='inner' : How to join the cross-sectional results over the variables,
            imputation_method: What imputation method to use before the clustering
            imputation_kwargs: Dict of keyword arguments to pass to the imputation method.
                For 'knn': n_neighbors, weights, metric, etc. (default: n_neighbors=20)
                For 'iterative': max_iter, estimator, n_nearest_features, etc. (default: max_iter=10)
                For 'miceforest': save_all_iterations, random_state, num_datasets for ImputationKernel,
                    mice_iterations (default: 10), n_estimators (default: 50) for mice() method
                For SimpleImputer methods ('mean', 'median', etc.): missing_values, copy, etc.
                Examples:
                    - KNN: {'n_neighbors': 10, 'weights': 'distance'}
                    - Iterative: {'max_iter': 20, 'estimator': RandomForestRegressor()}
                    - MiceForest: {'random_state': 42, 'mice_iterations': 15, 'n_estimators': 100}
                    - SimpleImputer: {'missing_values': -999, 'copy': False}
            cross_standardisation: what standardisation method to use
            verbose: bool = False: Whether to print progress information. Defaults to False.
            max_cross_missingness: float = 0.5: Maximum allowed missingness in cross-validation data.
        """

        assert analysis_resolution % interpolation_resolution == 0, (
            "Analysis resolution must be a multiple of the interpolation resolution"
        )
        # TODO: make compatible with multiple features

        start_time = time.perf_counter()

        extractors = [e.lower() for e in extractors]

        if max_time is not None and min_time is not None:
            assert max_time >= min_time, (
                "Max time has to be equal or greater than min time"
            )

        self.smoothing = smoothing
        self.smoothing_type = smoothing_type
        self.smoothing_window_size = smoothing_window_size
        self.n_skip = n_skip
        self.interpolation = interpolation
        self.interpolation_resolution = interpolation_resolution
        self.interpolation_keep_init = interpolation_keep_init
        self.interpolation_kwargs = interpolation_kwargs or {}
        self.analysis_resolution = analysis_resolution
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
        self.feature_columns = feature_columns or []
        self.val_cols = self.feature_columns
        self.add_ts_meta = add_ts_meta
        self.multivariate_join = multivariate_join
        self.imputation_method = imputation_method
        self.imputation_kwargs = imputation_kwargs or {}
        self.smoothing_kwargs = smoothing_kwargs or {}
        self.cross_standardisation = cross_standardisation
        self.max_cross_missingness = max_cross_missingness
        self.cluster_kwargs = cluster_kwargs

        if clustering_algorithm == "gmm":
            cluster_kwargs = (
                {"reg_covar": 1e-4, "covariance_type": "diag"}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = GaussianMixture(
                n_components=n_clusters, random_state=random_state, **cluster_kwargs
            )
        elif clustering_algorithm == "bgmm":
            cluster_kwargs = (
                {"reg_covar": 1e-4, "covariance_type": "diag"}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = BayesianGaussianMixture(
                n_components=n_clusters, random_state=random_state, **cluster_kwargs
            )
        elif clustering_algorithm == "kmeans":
            cluster_kwargs = (
                {"n_init": 10, "max_iter": 500}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = KMeans(
                n_clusters=n_clusters, random_state=random_state, **cluster_kwargs
            )
        elif clustering_algorithm == "optics":
            cluster_kwargs = (
                {"min_samples": 5, "max_eps": 0.5, "p": 1}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = OPTICS(**cluster_kwargs)
        elif clustering_algorithm == "hdbscan":
            cluster_kwargs = (
                {"min_samples": 5, "max_cluster_size": 1000}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = HDBSCAN(**cluster_kwargs)
        elif clustering_algorithm == "spectral":
            cluster_kwargs = (
                {"n_init": 10, "max_iter": 500, "affinity": "rbf"}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = SpectralClustering(
                n_clusters=n_clusters, random_state=random_state, **cluster_kwargs
            )
        elif clustering_algorithm == "hierarchical":
            cluster_kwargs = (
                {"n_clusters": n_clusters, "linkage": "ward", "metric": "manhattan"}
                if cluster_kwargs is None
                else cluster_kwargs
            )
            self.clustering_algorithm = AgglomerativeClustering(**cluster_kwargs)
        else:
            raise ValueError("Invalid clustering algorithm")

        self.imputer = build_imputer(
            imputation_method=imputation_method,
            imputation_kwargs=self.imputation_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.extractors = {
            "custom_features": "custom" in extractors,
            "tsfel_features": "tsfel" in extractors,
            "catch22_features": "catch22" in extractors,
            "tsfresh_features": "tsfresh" in extractors,
            "cesium_features": "cesium" in extractors,
            "antropy_features": "antropy" in extractors,
            "nolds_features": "nolds" in extractors,
            "katz_features": "katz" in extractors,
        }

        self.verbose = verbose
        self.is_fitted = False

        if self.verbose:
            if self.imputation_kwargs:
                logger.info(f"Using imputation method: {imputation_method}")
                logger.info(f"Imputation kwargs: {self.imputation_kwargs}")
            logger.debug("CrossSectionalClustering initialized")

    def fit(self, tsdf: DataFrame, y=None):
        fit_start_time = time.perf_counter()
        logger.info(f"Starting CrossSectionalClustering.fit(); TS shape {tsdf.shape}")

        id_kwargs = {
            "id_col": self.id_column,
            "time_col": self.time_column,
        }

        ts_cross_combined = DataFrame()
        ts_cross = DataFrame()
        durations = None

        # Timing dictionary
        self.timings = {}

        for var_num, _feature_column in enumerate(self.val_cols):
            print(_feature_column, flush=True)
            step_start_time = time.perf_counter()
            logger.info(f"Processing feature column: {_feature_column}")

            id_kwargs["val_col"] = _feature_column
            ts = tsdf[[self.id_column, self.time_column, _feature_column]]

            preproc = preprocess_timeseries_feature(
                ts,
                id_column=self.id_column,
                time_column=self.time_column,
                feature_column=_feature_column,
                min_measurements_per_id=self.min_measurements_per_id,
                min_time=self.min_time,
                max_time=self.max_time,
                interpolation=self.interpolation,
                interpolation_resolution=self.interpolation_resolution,
                interpolation_keep_init=self.interpolation_keep_init,
                interpolation_kwargs=self.interpolation_kwargs,
                smoothing=self.smoothing,
                smoothing_type=self.smoothing_type,
                smoothing_window_size=self.smoothing_window_size,
                n_skip=self.n_skip,
                smoothing_kwargs=self.smoothing_kwargs,
                analysis_resolution=self.analysis_resolution,
                dropna_before_normalisation=True,
                normalise_timeseries=self.normalise_timeseries,
                normalisation_method=self.normalisation_method,
                return_details=True,
            )
            ts = preproc["ts"]
            stages = preproc["stages"]
            prep_timings = preproc["timings"]

            self.timings[f"{_feature_column}_filter"] = prep_timings["filter"]
            self.timings[f"{_feature_column}_interpolation"] = prep_timings[
                "interpolation"
            ]
            self.timings[f"{_feature_column}_smoothing"] = prep_timings["smoothing"]
            self.timings[f"{_feature_column}_prune_nans"] = prep_timings["prune_nans"]
            self.timings[f"{_feature_column}_normalization"] = prep_timings[
                "normalization"
            ]

            if self.verbose:
                if "filtered" in stages:
                    self.ts_filtered = stages["filtered"]
                if "interpolated" in stages:
                    self.ts_interpolated = stages["interpolated"]
                if "smoothed" in stages:
                    self.ts_smoothed = stages["smoothed"]
                self.ts_smoothed_filtered = ts
                if "normalised" in stages:
                    self.ts_normalized = stages["normalised"]

            # get meta data
            if self.add_ts_meta:
                meta_start = time.perf_counter()
                ts_meta_src = stages.get("filtered", ts)
                ts_meta = {
                    "NumMeas": ts_meta_src.groupby(self.id_column).size(),
                    "MaxTime": ts_meta_src.groupby(self.id_column)[
                        self.time_column
                    ].max(),
                    "MeanTimeDiff": ts_meta_src.groupby(self.id_column)[
                        [self.time_column]
                    ]
                    .diff()
                    .set_index(ts_meta_src[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .mean(),
                    "RelTimeDiffVar": ts_meta_src.groupby(self.id_column)[
                        [self.time_column]
                    ]
                    .diff()
                    .set_index(ts_meta_src[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .std()
                    / ts_meta_src.groupby(self.id_column)[[self.time_column]]
                    .diff()
                    .set_index(ts_meta_src[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .mean(),
                    "TimeStdev": ts_meta_src.groupby(self.id_column)[
                        self.time_column
                    ].std(),
                    "MeanVal": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].mean(),
                    "StdVal": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].std(),
                    "SkewVal": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].skew(),
                    "Q91": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.91),
                    "Q95": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.95),
                    "Q99": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.99),
                    "Q50": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.50),
                    "Q01": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.01),
                    "Q05": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.05),
                    "Q09": ts_meta_src.groupby(self.id_column)[
                        _feature_column
                    ].quantile(0.10),
                }
                meta_time = time.perf_counter() - meta_start
                self.timings[f"{_feature_column}_meta"] = meta_time
                if self.verbose:
                    logger.info(
                        f"Meta data extraction completed for {_feature_column} in {meta_time:.4f} seconds"
                    )

            # extract cross sectional
            extract_start = time.perf_counter()
            ts_cross, durations = extractor.get_crossectional(
                ts, **self.extractors, **id_kwargs
            )
            extract_time_cross = time.perf_counter() - extract_start
            self.timings[f"{_feature_column}_cross_sectional"] = extract_time_cross
            if self.verbose:
                self.ts_cross = ts_cross
                logger.info(
                    f"Cross-sectional extraction completed for {_feature_column} in {extract_time_cross:.4f} seconds, TS cross: {ts_cross.shape}"
                )

            if self.add_ts_meta:
                meta_add_start = time.perf_counter()
                logger.info(f"--- ts shape --- : {ts_cross.shape}")
                for k, v in ts_meta.items():
                    v = DataFrame(v).reset_index()
                    v = v.set_index(self.id_column)
                    v.columns = [f"Meta_{k}"]
                    ts_cross = ts_cross.join(v, how="inner")
                meta_add_time = time.perf_counter() - meta_add_start
                self.timings[f"{_feature_column}_meta_add"] = meta_add_time
                if self.verbose:
                    self.ts_cross = ts_cross
                    logger.info(
                        f"Adding meta features completed for {_feature_column} in {meta_add_time:.4f} seconds"
                    )

            # add _feature_column as prefix
            ts_cross.columns = [f"{_feature_column}_{c}" for c in ts_cross.columns]
            step_start_time = time.perf_counter()
            if var_num == 0:
                ts_cross_combined = ts_cross
            if var_num > 0:
                ts_cross_combined = ts_cross_combined.join(
                    ts_cross, how=self.multivariate_join
                )

            step_time = time.perf_counter() - step_start_time
            self.timings[f"{_feature_column}_join"] = step_time
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(
                    f"Joining completed for {_feature_column} in {step_time:.4f} seconds"
                )

        # Replace all inf's and -inf's by NaN's
        replace_start = time.perf_counter()
        ts_cross_combined = ts_cross_combined.replace([inf, -inf], nan)
        replace_time = time.perf_counter() - replace_start
        self.timings["replace_inf"] = replace_time
        if self.verbose:
            logger.info(f"Replaced inf's by NaN's in {replace_time:.4f} seconds")

        # Remove columns with >P% missingness
        remove_start = time.perf_counter()
        num_cols = ts_cross_combined.shape[1]
        ts_cross_combined = ts_cross_combined.dropna(
            axis=1, thresh=int(ts_cross_combined.shape[0] * self.max_cross_missingness)
        )
        remove_time_missing = time.perf_counter() - remove_start
        self.timings["remove_missing"] = remove_time_missing
        if self.verbose:
            logger.info(
                f"Removed {num_cols - ts_cross_combined.shape[1]} columns with more than {self.max_cross_missingness * 100}% missingness in {remove_time_missing:.4f} seconds"
            )

        # Remove columns with zero variance
        remove_start = time.perf_counter()
        num_cols = ts_cross_combined.shape[1]
        ts_cross_combined = ts_cross_combined.loc[:, ts_cross_combined.var() > 0]
        remove_time_zerovar = time.perf_counter() - remove_start
        self.timings["remove_zerovar"] = remove_time_zerovar
        if self.verbose:
            logger.info(
                f"Removed {num_cols - ts_cross_combined.shape[1]} columns with zero variance {remove_time_zerovar:.4f} seconds"
            )

        # Remove perfectly correlated features
        remove_start = time.perf_counter()
        cols = ts_cross_combined.columns
        droplist = []
        keeplist = []
        duplicated = set()
        for i, cl in tqdm(enumerate(cols)):
            for cr in cols[i + 1 :]:
                if ts_cross_combined[cl].equals(ts_cross_combined[cr]):
                    keeplist.append(cl)
                    droplist.append(cr)
                    duplicated.add((cl, cr))
        to_drop = list(set(droplist).difference(set(keeplist)))
        ts_cross_combined = ts_cross_combined.drop(columns=to_drop)
        remove_time = time.perf_counter() - remove_start
        self.timings["remove_duplicates"] = remove_time
        if self.verbose:
            logger.info(
                f"Removed {len(duplicated)} columns because of duplication in {remove_time:.4f} seconds"
            )

        # StandardScaling
        if self.cross_standardisation:
            scale_start = time.perf_counter()
            ts_cross_combined = DataFrame(
                StandardScaler().fit_transform(ts_cross_combined),
                index=ts_cross_combined.index,
                columns=ts_cross_combined.columns,
            )
            scale_time = time.perf_counter() - scale_start
            self.timings["standardization"] = scale_time
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(f"Standardization completed in {scale_time:.4f} seconds")

        # Imputation
        if ts_cross_combined.isna().sum().sum() > 0:
            impute_start = time.perf_counter()
            missing_count = ts_cross_combined.isna().sum().sum()
            logger.info(f"Found {missing_count} missing values, starting imputation")
            logger.info(
                "Running %s imputation",
                "miceforest" if self.imputer == "mice" else type(self.imputer).__name__,
            )
            ts_cross_combined = impute_cross_sectional(
                ts_cross_combined=ts_cross_combined,
                imputer=self.imputer,
                imputation_kwargs=self.imputation_kwargs,
            )
            impute_time = time.perf_counter() - impute_start
            self.timings["imputation"] = impute_time
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(f"Imputation completed in {impute_time:.4f} seconds")
        else:
            logger.debug(f"No missing values in cross-sectional data")
            self.timings["imputation"] = 0.0

        # Clustering
        cluster_start = time.perf_counter()
        self.clustering_algorithm.fit(ts_cross_combined)
        cluster_time = time.perf_counter() - cluster_start
        self.timings["clustering"] = cluster_time
        logger.info(f"Clustering completed in {cluster_time:.4f} seconds")
        self.is_fitted = True
        self.ts_cross_combined = ts_cross_combined

        self.durations = durations
        if self.verbose:
            logger.debug(f"Clustering completed")

        fit_total_time = time.perf_counter() - fit_start_time
        self.timings["total"] = fit_total_time
        logger.info(
            f"CrossSectionalClustering.fit() completed in {fit_total_time:.4f} seconds"
        )

        return self

    def predict(self, X=None):
        return self.clustering_algorithm.predict(self.ts_cross_combined)

    def predict_proba(self, X=None):
        assert hasattr(self.clustering_algorithm, "predict_proba"), (
            "Clustering algorithm does not support predict_proba"
        )
        return self.clustering_algorithm.predict_proba(self.ts_cross_combined)

    def get_scores(self):
        # Davies-Bouldin
        # Calinski-Harabasz
        # Silhouette
        # mean entropy over class probabilities (if there are probas) (we call this MEP)
        # mean max proba over class probabilities (we call this MMP)
        # AIC and BIC if available
        #

        # assert that clustering_algorithm is fitted
        assert self.is_fitted, "Clustering algorithm is not fitted"

        score_dict = {}
        cluster_labels = self.clustering_algorithm.predict(self.ts_cross_combined)
        # Davies-Bouldin
        score_dict["davies_bouldin"] = davies_bouldin_score(
            self.ts_cross_combined, cluster_labels
        )
        # Calinski-Harabasz
        score_dict["calinski_harabasz"] = calinski_harabasz_score(
            self.ts_cross_combined, cluster_labels
        )
        # Silhouette
        score_dict["silhouette"] = silhouette_score(
            self.ts_cross_combined, cluster_labels
        )

        # add AIC and BIC if available
        if hasattr(self.clustering_algorithm, "aic"):
            score_dict["aic"] = self.clustering_algorithm.aic(self.ts_cross_combined)
        else:
            score_dict["aic"] = None
        if hasattr(self.clustering_algorithm, "bic"):
            score_dict["bic"] = self.clustering_algorithm.bic(self.ts_cross_combined)
        else:
            score_dict["bic"] = None

        # if self.clustering_algorithm has predict_proba, extract the MEP and MPP
        if hasattr(self.clustering_algorithm, "predict_proba"):
            probas = self.clustering_algorithm.predict_proba(self.ts_cross_combined)
            score_dict["mep"] = np.mean(-np.sum(probas * np.log(probas), axis=1))
            score_dict["mpp"] = np.mean(np.max(probas, axis=1))
        return score_dict

    def viz(self):
        # use Umap to visualize the clusters
        umap = UMAP(n_components=2)
        embedding = umap.fit_transform(self.ts_cross_combined)
        cluster_labels = self.clustering_algorithm.predict(self.ts_cross_combined)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=cluster_labels,
            cmap="viridis",
            alpha=0.5,
        )
        plt.colorbar(scatter)
        plt.title("UMAP Visualization of Clusters")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.show()


class TSDistanceBasedClustering(BaseEstimator, ClusterMixin):
    """Distance-based clustering wrapper for tslearn models.

    Supports:
    - TimeSeriesKMeans
    - KernelKMeans
    - KShape
    - TimeSeriesDBSCAN
    """

    def __init__(
        self,
        method: Literal[
            "kmeans",
            "kernelkmeans",
            "kshape",
            "kmedoids",
            "dbscan",
            "kvisibility",
            "stdbscan",
        ] = "kmeans",
        distance_metric: Literal[
            "euclidean",
            "dtw",
            "softdtw",
            "softdtw_normalized",
            "precomputed",
            "ctw",
            "frechet",
            "precomputed",
        ] = "euclidean",
        backend: Literal["tslearn", "sktime"] = "tslearn",
        n_clusters: int = 3,
        random_state: int = 42,
        model_kwargs: dict[str, Any] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        interpolation: bool = False,
        interpolation_resolution: int = 1,
        interpolation_keep_init: bool = False,
        interpolation_kwargs: dict[str, Any] | None = None,
        analysis_resolution: int = 1,
        min_measurements_per_id: int = 1,
        min_time: Optional[int] = None,
        max_time: Optional[int] = None,
        smoothing: bool = False,
        smoothing_type: Literal[
            "gaussian_kernel", "gaussian_kernel_simple", "box_kernel", "rolling_mean"
        ] = "gaussian_kernel",
        smoothing_window_size: int = 4,
        n_skip: int = 1,
        smoothing_kwargs: dict[str, Any] | None = None,
        imputation_method: Optional[
            Literal[
                "knn",
                "iterative",
                "miceforest",
                "mean",
                "median",
                "most_frequent",
                "constant",
            ]
        ] = None,
        imputation_kwargs: dict[str, Any] | None = None,
        normalise_timeseries: Optional[Literal["bulk", "group"]] = None,
        normalisation_method: Optional[Literal["standard", "minmax"]] = "standard",
        cross_standardisation: bool = False,
        id_column: str = "id",
        time_column: str = "time",
        value_columns: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        assert method in [
            "kmeans",
            "kernelkmeans",
            "kshape",
            "kmedoids",
            "dbscan",
            "stdbscan",
            "kvisibility",
        ], (
            f"We only support kmeans/kmedoids/dbscan/kernelkmeans/stdbscan and kshapes at the moment."
        )

        acceptable_distance_metrics = {
            "backend": {
                "tslearn": {
                    "kmeans": ["euclidean", "softdtw", "dtw", "precomputed"],
                    "dbscan": [
                        "euclidean",
                        "dtw",
                        "ctw",
                        "frechet",
                        "softdtw_normalized",
                        "precomputed",
                    ],
                    "kernelkmeans": [
                        "gak",
                        "additive_chi2",
                        "chi2",
                        "linear",
                        "poly",
                        "polynomial",
                        "rbf",
                        "laplacian",
                        "sigmoid",
                        "cosine",
                    ],
                    "kshape": [],
                },
                "sktime": {
                    "kmeans": [
                        "dtw",
                        "euclidean",
                        "erp",
                        "edr",
                        "lcss",
                        "squared",
                        "ddtw",
                        "wdtw",
                        "wddtw",
                    ],
                    "kmedoids": [
                        "dtw",
                        "euclidean",
                        "erp",
                        "edr",
                        "lcss",
                        "squared",
                        "ddtw",
                        "wdtw",
                        "wddtw",
                    ],
                    "dbscan": [
                        "dtw",
                        "euclidean",
                        "erp",
                        "edr",
                        "lcss",
                        "squared",
                        "ddtw",
                        "wdtw",
                        "wddtw",
                    ],
                    "kvisibility": [],
                    "stdbscan": [
                        "euclidean",
                        "manhattan",
                        "chebyshev",
                        "minkowski",
                        "cosine",
                        "haversine",
                        "sqeuclidean",
                        "jensenshannon",
                        "canberra",
                        "correlationbraycurtis",
                    ],
                },
            }
        }
        if backend not in acceptable_distance_metrics["backend"].keys():
            raise ValueError(
                f"Only {list(acceptable_distance_metrics['backend'].keys())} are currently supported as a backend"
            )
        if method not in acceptable_distance_metrics["backend"][backend].keys():
            raise ValueError(
                f"Only {list(acceptable_distance_metrics['backend'][backend].keys())} are currently supported as methods for {backend}"
            )
        if (
            distance_metric
            not in acceptable_distance_metrics["backend"][backend][method]
        ) and (len(acceptable_distance_metrics["backend"][backend][method]) > 0):
            raise ValueError(
                f"Only {list(acceptable_distance_metrics['backend'][backend][method])} are currently supported as metric for {backend}/{method}"
            )

        self.method = method.lower()
        self.distance_metric = distance_metric
        self.backend = backend
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_kwargs = model_kwargs or {}
        self.metric_kwargs = metric_kwargs or {}

        if analysis_resolution % interpolation_resolution != 0:
            raise ValueError(
                "Analysis resolution must be a multiple of interpolation resolution"
            )
        if max_time is not None and min_time is not None and max_time < min_time:
            raise ValueError("max_time must be equal or greater than min_time")

        self.interpolation = interpolation
        self.interpolation_resolution = interpolation_resolution
        self.interpolation_keep_init = interpolation_keep_init
        self.interpolation_kwargs = interpolation_kwargs or {}
        self.analysis_resolution = analysis_resolution
        self.min_measurements_per_id = min_measurements_per_id
        self.min_time = min_time
        self.max_time = max_time

        self.smoothing = smoothing
        self.smoothing_type = smoothing_type
        self.smoothing_window_size = smoothing_window_size
        self.n_skip = n_skip
        self.smoothing_kwargs = smoothing_kwargs or {}

        self.imputation_method = imputation_method
        self.imputation_kwargs = imputation_kwargs or {}

        self.normalise_timeseries = normalise_timeseries
        self.normalisation_method = normalisation_method
        self.cross_standardisation = cross_standardisation

        self.id_column = id_column
        self.time_column = time_column
        self.value_columns = value_columns or ["value"]
        if len(self.value_columns) == 0:
            raise ValueError("value_columns must contain at least one column name")
        self.verbose = verbose

        self._imputer = (
            build_imputer(
                imputation_method=imputation_method,
                imputation_kwargs=self.imputation_kwargs,
                random_state=random_state,
                verbose=verbose,
            )
            if imputation_method is not None
            else None
        )

        self.clusterer = self._build_clusterer()
        self.is_fitted = False

    def _build_clusterer(self):
        kwargs = dict(self.model_kwargs)

        if self.backend == "tslearn":
            from tslearn import clustering as tslearn_clustering

            if self.method == "kmeans":
                kwargs.setdefault("n_clusters", self.n_clusters)
                kwargs.setdefault("metric", self.distance_metric)
                kwargs.setdefault("random_state", self.random_state)
                kwargs.setdefault("metric_params", self.metric_kwargs)
                return tslearn_clustering.TimeSeriesKMeans(**kwargs)

            if self.method == "kernelkmeans":
                kwargs.setdefault("n_clusters", self.n_clusters)
                kwargs.setdefault("random_state", self.random_state)
                kwargs.setdefault("kernel", self.distance_metric)
                kwargs.setdefault("kernel_params", self.metric_kwargs)
                return tslearn_clustering.KernelKMeans(**kwargs)

            if self.method == "kshape":
                kwargs.setdefault("n_clusters", self.n_clusters)
                kwargs.setdefault("random_state", self.random_state)
                return tslearn_clustering.KShape(**kwargs)

            if self.method == "dbscan":
                if not hasattr(tslearn_clustering, "TimeSeriesDBSCAN"):
                    raise ValueError(
                        "TimeSeriesDBSCAN is not available in this tslearn version"
                    )
                kwargs.setdefault("metric", self.distance_metric)
                kwargs.setdefault("metric_params", self.metric_kwargs)
                return tslearn_clustering.TimeSeriesDBSCAN(**kwargs)

            raise ValueError(
                "Invalid method. Choose from: kmeans, kernelkmeans, kshape, dbscan"
            )
        elif self.backend == "sktime":
            from sktime.clustering.k_means import TimeSeriesKMeans as SkTime_TSKMeans
            from sktime.clustering.k_medoids import (
                TimeSeriesKMedoids as SkTime_TSKMedoids,
            )
            from sktime.clustering.dbscan import TimeSeriesDBSCAN as SkTime_TSDBSCAN
            from sktime.clustering.kvisibility import (
                TimeSeriesKvisibility as SkTime_TSKViz,
            )
            from sktime.clustering.spatio_temporal import STDBSCAN

            if self.method == "kmeans":
                kwargs.setdefault("n_clusters", self.n_clusters)
                kwargs.setdefault("metric", self.distance_metric)
                kwargs.setdefault("random_state", self.random_state)
                kwargs.setdefault("distance_params", self.metric_kwargs)
                return SkTime_TSKMeans(**kwargs)
            elif self.method == "kmedoids":
                kwargs.setdefault("n_clusters", self.n_clusters)
                kwargs.setdefault("metric", self.distance_metric)
                kwargs.setdefault("random_state", self.random_state)
                kwargs.setdefault("distance_params", self.metric_kwargs)
                return SkTime_TSKMedoids(**kwargs)
            elif self.method == "dbscan":
                kwargs.setdefault("distance", self.distance_metric)
                kwargs.setdefault("distance_params", self.metric_kwargs)
                return SkTime_TSDBSCAN(**kwargs)
            elif self.method == "kvisibility":
                kwargs.setdefault("n_clusters", self.n_clusters)
                return SkTime_TSKViz(**kwargs)
            elif self.method == "stdbscan":
                kwargs.setdefault("metric", self.distance_metric)
                return STDBSCAN(**kwargs)

        raise ValueError("Invalid backend. Choose from: tslearn, sktime")

    def _prepare_from_long_df(self, ts_df: DataFrame) -> tuple[ndarray, ndarray]:
        base_cols = [self.id_column, self.time_column, *self.value_columns]
        ts = ts_df[base_cols].copy()

        filtered_for_meta: Optional[DataFrame] = None

        processed_features: list[DataFrame] = []
        for feature in self.value_columns:
            feat_details = preprocess_timeseries_feature(
                ts,
                id_column=self.id_column,
                time_column=self.time_column,
                feature_column=feature,
                min_measurements_per_id=self.min_measurements_per_id,
                min_time=self.min_time,
                max_time=self.max_time,
                interpolation=self.interpolation,
                interpolation_resolution=self.interpolation_resolution,
                interpolation_keep_init=self.interpolation_keep_init,
                interpolation_kwargs=self.interpolation_kwargs,
                smoothing=self.smoothing,
                smoothing_type=self.smoothing_type,
                smoothing_window_size=self.smoothing_window_size,
                n_skip=self.n_skip,
                smoothing_kwargs=self.smoothing_kwargs,
                analysis_resolution=self.analysis_resolution,
                dropna_before_normalisation=False,
                normalise_timeseries=self.normalise_timeseries,
                normalisation_method=self.normalisation_method,
                return_details=False,
            )

            feat_ts = feat_details

            processed_features.append(
                feat_ts[[self.id_column, self.time_column, feature]]
            )

        ts_processed = processed_features[0]
        for feat_ts in processed_features[1:]:
            ts_processed = ts_processed.merge(
                feat_ts,
                on=[self.id_column, self.time_column],
                how="inner",
            )

        wide_per_feature: list[DataFrame] = []
        base_index = None
        base_columns = None
        for feature in self.value_columns:
            wide = (
                ts_processed.pivot(
                    index=self.id_column,
                    columns=self.time_column,
                    values=feature,
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )

            if self._imputer is not None and wide.isna().sum().sum() > 0:
                wide = impute_cross_sectional(
                    ts_cross_combined=wide,
                    imputer=self._imputer,
                    imputation_kwargs=self.imputation_kwargs,
                )

            if wide.isna().sum().sum() > 0:
                raise ValueError(
                    f"Feature '{feature}' contains NaNs after preprocessing. Enable imputation or provide denser data."
                )

            if base_index is None:
                base_index = wide.index
                base_columns = wide.columns
            else:
                wide = wide.reindex(index=base_index, columns=base_columns)

            wide_per_feature.append(wide)

        arr = np.stack([w.to_numpy(dtype=float) for w in wide_per_feature], axis=2)

        if self.cross_standardisation:
            arr_flat = arr.reshape(arr.shape[0], -1)
            arr_flat = StandardScaler().fit_transform(arr_flat)
            arr = arr_flat.reshape(arr.shape)

        if self.backend == "tslearn":
            # expects  array with dimensions [n_instances, series_length, n_dimensions]
            return arr, base_index.to_numpy()
        elif self.backend == "sktime":
            # expects  array with dimensions [n_instances, n_dimensions, series_length]
            return np.einsum("ijk->ikj", arr), base_index.to_numpy()

    def _prepare_X(self, X: DataFrame | ndarray) -> tuple[ndarray, ndarray]:
        if isinstance(X, DataFrame):
            return self._prepare_from_long_df(X)

        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3:
            raise ValueError("X must be a long DataFrame, 2D array, or 3D array")

        if self.normalise_timeseries == "group":
            if self.normalisation_method == "standard":
                mean = arr.mean(axis=1, keepdims=True)
                std = arr.std(axis=1, keepdims=True)
                std[std == 0] = 1.0
                arr = (arr - mean) / std
            elif self.normalisation_method == "minmax":
                mn = arr.min(axis=1, keepdims=True)
                mx = arr.max(axis=1, keepdims=True)
                denom = mx - mn
                denom[denom == 0] = 1.0
                arr = (arr - mn) / denom
        elif self.normalise_timeseries == "bulk":
            if self.normalisation_method == "standard":
                mean = arr.mean(axis=(0, 1), keepdims=True)
                std = arr.std(axis=(0, 1), keepdims=True)
                std[std == 0] = 1.0
                arr = (arr - mean) / std
            elif self.normalisation_method == "minmax":
                mn = arr.min(axis=(0, 1), keepdims=True)
                mx = arr.max(axis=(0, 1), keepdims=True)
                denom = mx - mn
                denom[denom == 0] = 1.0
                arr = (arr - mn) / denom

        if self.cross_standardisation:
            arr_flat = arr.reshape(arr.shape[0], -1)
            arr_flat = StandardScaler().fit_transform(arr_flat)
            arr = arr_flat.reshape(arr.shape)

        index = np.arange(arr.shape[0])
        return arr, index

    def fit(self, X: DataFrame | ndarray, y=None):
        X_prepared, index = self._prepare_X(X)

        print(50 * "+")
        print(f"Final shape of fit-matrix:{X_prepared.shape}")
        print(50 * "+")

        if hasattr(self.clusterer, "fit_predict"):
            labels = self.clusterer.fit_predict(X_prepared)
        else:
            self.clusterer.fit(X_prepared)
            labels = getattr(self.clusterer, "labels_", None)

        self.labels_ = labels
        self.X_ = X_prepared
        self.index_ = index
        self.is_fitted = True
        return self

    def predict(self, X: Optional[DataFrame | ndarray] = None):
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        if X is None:
            if self.labels_ is None:
                raise ValueError("Labels are not available for this clustering model")
            return self.labels_

        X_prepared, _ = self._prepare_X(X)
        if hasattr(self.clusterer, "predict"):
            return self.clusterer.predict(X_prepared)

        raise ValueError("This clustering model does not support out-of-sample predict")

    def viz(self):
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        labels = self.labels_
        if labels is None:
            if hasattr(self.clusterer, "predict"):
                labels = self.clusterer.predict(self.X_)
            else:
                raise ValueError("No labels available for visualization")

        X_flat = self.X_.reshape(self.X_.shape[0], -1)
        embedding = UMAP(n_components=2, random_state=self.random_state).fit_transform(
            X_flat
        )

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
        )
        plt.colorbar(scatter)
        plt.title("UMAP Visualization of Time-Series Clusters")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.show()

        return embedding


class TimeSeriesKMeansClustering(TSDistanceBasedClustering):
    def __init__(self, **kwargs):
        super().__init__(method="timeserieskmeans", **kwargs)


class KernelKMeansClustering(TSDistanceBasedClustering):
    def __init__(self, **kwargs):
        super().__init__(method="kernelkmeans", **kwargs)


class KShapeClustering(TSDistanceBasedClustering):
    def __init__(self, **kwargs):
        super().__init__(method="kshape", **kwargs)


class TimeSeriesDBSCANClustering(TSDistanceBasedClustering):
    def __init__(self, **kwargs):
        super().__init__(method="timeseriesdbscan", **kwargs)


# TODO: add class for ModelBasedClustering (Latent, Deeplearning)
#

# TODO: add class for EvolutionaryClustering
#

# TODO: add class for MarkovModelBasedClustering
#

# TODO: add class for GnomeClustering
#
