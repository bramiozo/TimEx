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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tslearn import clustering as tslearn_clustering
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
        imputation_method: Literal["knn", "iterative", "miceforest"] = "knn",
        imputation_kwargs: dict[str, Any] | None = None,
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
        else:
            raise ValueError("Invalid clustering algorithm")

        # Set up imputer with custom kwargs
        if imputation_method == "knn":
            knn_kwargs = (
                {"n_neighbors": 20} if imputation_kwargs is None else imputation_kwargs
            )
            self.imputer = KNNImputer(**knn_kwargs)
        elif imputation_method == "iterative":
            iterative_kwargs = (
                {
                    "max_iter": 10,
                    "random_state": random_state,
                    "verbose": verbose,
                }
                if imputation_kwargs is None
                else imputation_kwargs
            )
            self.imputer = IterativeImputer(**iterative_kwargs)
        elif imputation_method == "miceforest":
            self.imputer = "mice"  # placeholder for miceforest
        elif imputation_method == "mean":
            simple_kwargs = {"strategy": "mean"}
            simple_kwargs.update(self.imputation_kwargs)
            self.imputer = SimpleImputer(**simple_kwargs)
        elif imputation_method == "median":
            simple_kwargs = {"strategy": "median"}
            simple_kwargs.update(self.imputation_kwargs)
            self.imputer = SimpleImputer(**simple_kwargs)
        elif imputation_method == "most_frequent":
            simple_kwargs = {"strategy": "most_frequent"}
            simple_kwargs.update(self.imputation_kwargs)
            self.imputer = SimpleImputer(**simple_kwargs)
        elif imputation_method == "constant":
            simple_kwargs = {"strategy": "constant", "fill_value": 0}
            simple_kwargs.update(self.imputation_kwargs)
            self.imputer = SimpleImputer(**simple_kwargs)
        else:
            raise ValueError("Invalid imputation method")

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

        # TODO: (OPTION) for cross-channel features combine into combi-channel, before cross-sect extraction

        for var_num, _feature_column in enumerate(self.val_cols):
            print(_feature_column, flush=True)
            step_start_time = time.perf_counter()
            logger.info(f"Processing feature column: {_feature_column}")

            id_kwargs["val_col"] = _feature_column
            ts = tsdf[[self.id_column, self.time_column, _feature_column]]

            # Filtering step
            filter_start = time.perf_counter()
            ts = preprocessing.get_filtered_df(
                ts,
                min_time=self.min_time,
                min_measurements=self.min_measurements_per_id,
                **{k: v for k, v in id_kwargs.items() if k != "val_col"},
            )
            filter_time = time.perf_counter() - filter_start
            if self.verbose:
                self.ts_filtered = ts
                logger.info(
                    f"Filtering completed for {_feature_column} in {filter_time:.4f} seconds. TS: {ts.shape}"
                )

            # get meta data
            if self.add_ts_meta:
                meta_start = time.perf_counter()
                # per id extract number of measurements, max_time, time_per_measurement, time_var
                #
                ts_meta = {
                    "NumMeas": ts.groupby(self.id_column).size(),
                    "MaxTime": ts.groupby(self.id_column)[self.time_column].max(),
                    "MeanTimeDiff": ts.groupby(self.id_column)[[self.time_column]]
                    .diff()
                    .set_index(ts[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .mean(),
                    "RelTimeDiffVar": ts.groupby(self.id_column)[[self.time_column]]
                    .diff()
                    .set_index(ts[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .std()
                    / ts.groupby(self.id_column)[[self.time_column]]
                    .diff()
                    .set_index(ts[self.id_column])
                    .reset_index()
                    .groupby(self.id_column)
                    .mean(),
                    "TimeStdev": ts.groupby(self.id_column)[self.time_column].std(),
                    "MeanVal": ts.groupby(self.id_column)[_feature_column].mean(),
                    "StdVal": ts.groupby(self.id_column)[_feature_column].std(),
                    "SkewVal": ts.groupby(self.id_column)[_feature_column].skew(),
                    "Q91": ts.groupby(self.id_column)[_feature_column].quantile(0.91),
                    "Q95": ts.groupby(self.id_column)[_feature_column].quantile(0.95),
                    "Q99": ts.groupby(self.id_column)[_feature_column].quantile(0.99),
                    "Q50": ts.groupby(self.id_column)[_feature_column].quantile(0.50),
                    "Q01": ts.groupby(self.id_column)[_feature_column].quantile(0.01),
                    "Q05": ts.groupby(self.id_column)[_feature_column].quantile(0.05),
                    "Q09": ts.groupby(self.id_column)[_feature_column].quantile(0.10),
                }
                meta_time = time.perf_counter() - meta_start
                if self.verbose:
                    logger.info(
                        f"Meta data extraction completed for {_feature_column} in {meta_time:.4f} seconds"
                    )

            if self.interpolation:
                interp_start = time.perf_counter()
                ts = preprocessing.get_interpolated(
                    ts,
                    time_res=self.interpolation_resolution,
                    max_time=self.max_time,
                    keep_t0_value=self.interpolation_keep_init,
                    df_out=True,
                    **id_kwargs,
                )
                interp_time = time.perf_counter() - interp_start
                if self.verbose:
                    self.ts_interpolated = ts
                    logger.info(
                        f"Interpolation completed for {_feature_column} in {interp_time:.4f} seconds, TS: {ts.shape}"
                    )

            if self.smoothing:
                # TODO: refactor wrap into one function..
                smooth_start = time.perf_counter()
                if self.smoothing_type in [
                    "gaussian_kernel",
                    "gaussian_kernel_simple",
                    "rolling_mean",
                    "box_kernel",
                ]:
                    ts = preprocessing.get_smoothed(
                        ts_df=ts,
                        window=self.smoothing_window_size,
                        Nskip=self.n_skip,
                        df_out=True,
                        smoothing_method=self.smoothing_type,
                        **id_kwargs,
                    )
                else:
                    raise ValueError("Invalid smoothing type")
                smooth_time = time.perf_counter() - smooth_start
                if self.verbose:
                    self.ts_smoothed = ts
                    logger.info(
                        f"Smoothing completed for {_feature_column} in {smooth_time:.4f} seconds, TS: {ts.shape}"
                    )

            if self.analysis_resolution != self.interpolation_resolution:
                # We need to keep only every self.analysis_resolution/self.interpolation_resolution values, per ID
                ts = ts.groupby(self.id_column).apply(
                    lambda x: x.iloc[
                        :: self.analysis_resolution // self.interpolation_resolution
                    ]
                )
                if self.verbose:
                    self.ts_smoothed_filtered = ts
                    logger.info(
                        f"Selection after smoothing for {_feature_column} in {smooth_time:.4f} seconds, TS: {ts.shape}"
                    )

            # prune all NaNs values
            ts = ts.dropna(subset=[_feature_column])
            if self.verbose:
                self.ts_smoothed_filtered = ts
                logger.info(
                    f"Selection after pruning NaNs for {_feature_column} in {smooth_time:.4f} seconds, TS: {ts.shape}"
                )

            if self.normalise_timeseries:
                norm_start = time.perf_counter()
                ts = preprocessing.normalise_ts(
                    ts, scaler=self.normalisation_method, df_out=True, **id_kwargs
                )
                norm_time = time.perf_counter() - norm_start
                if self.verbose:
                    self.ts_normalized = ts
                    logger.info(
                        f"Normalization completed for {_feature_column} in {norm_time:.4f} seconds, TS: {ts.shape}"
                    )

            # extract cross sectional
            extract_start = time.perf_counter()
            ts_cross, durations = extractor.get_crossectional(
                ts, **self.extractors, **id_kwargs
            )
            extract_time = time.perf_counter() - extract_start
            if self.verbose:
                self.ts_cross = ts_cross
                logger.info(
                    f"Cross-sectional extraction completed for {_feature_column} in {extract_time:.4f} seconds, TS cross: {ts_cross.shape}"
                )

            if self.add_ts_meta:
                meta_add_start = time.perf_counter()
                logger.info(f"--- ts shape --- : {ts_cross.shape}")
                for k, v in ts_meta.items():
                    v = DataFrame(v).reset_index()
                    v = v.set_index(self.id_column)
                    v.columns = [f"Meta_{k}"]
                    ts_cross = ts_cross.join(v, how="inner")
                    # print(f"--- ts shape --- : {ts_cross.shape}, + {k}")
                meta_add_time = time.perf_counter() - meta_add_start
                if self.verbose:
                    self.ts_cross = ts_cross
                    logger.info(
                        f"Adding meta features completed for {_feature_column} in {meta_add_time:.4f} seconds"
                    )

            # add _feature_column as prefix
            #
            ts_cross.columns = [f"{_feature_column}_{c}" for c in ts_cross.columns]

            if var_num == 0:
                ts_cross_combined = ts_cross
            if var_num > 0:
                ts_cross_combined = ts_cross_combined.join(
                    ts_cross, how=self.multivariate_join
                )

            step_time = time.perf_counter() - step_start_time
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(
                    f"Joining completed for {_feature_column} in {extract_time:.4f} seconds"
                )

        # Replace all inf's and -inf's by NaN's
        #
        replace_start = time.perf_counter()
        ts_cross_combined = ts_cross_combined.replace([inf, -inf], nan)
        if self.verbose:
            replace_time = time.perf_counter() - replace_start
            logger.info(f"Replaced inf's by NaN's in {replace_time:.4f} seconds")

        # Remove columns with >P% missingness
        #
        remove_start = time.perf_counter()
        num_cols = ts_cross_combined.shape[1]
        ts_cross_combined = ts_cross_combined.dropna(
            axis=1, thresh=int(ts_cross_combined.shape[0] * self.max_cross_missingness)
        )
        if self.verbose:
            remove_time = time.perf_counter() - remove_start
            logger.info(
                f"Removed {num_cols - ts_cross_combined.shape[1]} columns with more than {self.max_cross_missingness * 100}% missingness in {remove_time:.4f} seconds"
            )

        # Remove columns with zero variance
        #
        remove_start = time.perf_counter()
        num_cols = ts_cross_combined.shape[1]
        ts_cross_combined = ts_cross_combined.loc[:, ts_cross_combined.var() > 0]
        if self.verbose:
            remove_time = time.perf_counter() - remove_start
            logger.info(
                f"Removed {num_cols - ts_cross_combined.shape[1]} columns with zero variance {remove_time:.4f} seconds"
            )

        # Remove perfectly correlated features
        #
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
        if self.verbose:
            remove_time = time.perf_counter() - remove_start
            logger.info(
                f"Removed {len(duplicated)} columns because of duplication in {remove_time:.4f} seconds"
            )
        # TODO: (OPTION) for cross-channel features combine cross-sect features a posteriori
        #

        # StandardScaling
        #
        if self.cross_standardisation:
            scale_start = time.perf_counter()
            ts_cross_combined = DataFrame(
                StandardScaler().fit_transform(ts_cross_combined),
                index=ts_cross_combined.index,
                columns=ts_cross_combined.columns,
            )
            scale_time = time.perf_counter() - scale_start
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(f"Standardization completed in {scale_time:.4f} seconds")

        # Imputation
        #
        if ts_cross_combined.isna().sum().sum() > 0:
            impute_start = time.perf_counter()
            missing_count = ts_cross_combined.isna().sum().sum()
            logger.info(f"Found {missing_count} missing values, starting imputation")
            if self.imputer == "mice":
                # Default miceforest kwargs
                mice_kwargs = {
                    "save_all_iterations": True,
                    "random_state": 100,
                    "num_datasets": 1,
                }
                mice_kwargs.update(self.imputation_kwargs)

                imp_kernel = mf.ImputationKernel(ts_cross_combined, **mice_kwargs)

                # Mice iteration kwargs
                mice_iter_kwargs = {"n_estimators": 50}
                if "mice_iterations" in self.imputation_kwargs:
                    mice_iterations = self.imputation_kwargs.pop("mice_iterations")
                else:
                    mice_iterations = 10
                if "n_estimators" in self.imputation_kwargs:
                    mice_iter_kwargs["n_estimators"] = self.imputation_kwargs[
                        "n_estimators"
                    ]

                logger.info(
                    f"Running miceforest with {mice_iterations} iterations and kwargs: {mice_iter_kwargs}"
                )
                imp_kernel.mice(mice_iterations, **mice_iter_kwargs)
                ts_cross_combined = imp_kernel.complete_data()
            else:
                logger.info(f"Running {type(self.imputer).__name__} imputation")
                ts_cross_combined = DataFrame(
                    self.imputer.fit_transform(ts_cross_combined),
                    index=ts_cross_combined.index,
                    columns=ts_cross_combined.columns,
                )
            impute_time = time.perf_counter() - impute_start
            if self.verbose:
                self.ts_cross_combined = ts_cross_combined
                logger.info(f"Imputation completed in {impute_time:.4f} seconds")
        else:
            logger.debug(f"No missing values in cross-sectional data")

        # Clustering
        #
        cluster_start = time.perf_counter()
        self.clustering_algorithm.fit(ts_cross_combined)
        cluster_time = time.perf_counter() - cluster_start
        logger.info(f"Clustering completed in {cluster_time:.4f} seconds")
        self.is_fitted = True
        self.ts_cross_combined = ts_cross_combined

        self.durations = durations
        if self.verbose:
            logger.debug(f"Clustering completed")

        fit_total_time = time.perf_counter() - fit_start_time
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


# TODO: add class for DistanceBasedClustering
# TODO: add class for ModelBasedClustering (Latent, Deeplearning)
# TODO: add class for EvolutionaryClustering
# TODO: add class for MarkovModelBasedClustering
# TODO: add class for GnomeClustering
