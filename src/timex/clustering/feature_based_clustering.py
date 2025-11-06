from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklego.meta import GroupedTransformer
from typing import Literal

from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from typing import List, Literal

from timex import extractor
from timex import preprocessing


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA similarity between two representations X and Y.
    """
    # center features
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n < 2:
        return 0.0
    # linear kernels
    Kx = Xc @ Xc.T
    Ky = Yc @ Yc.T
    # centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H
    # unbiased-ish HSIC estimates (scaled similarly)
    hsic = np.trace(Kx_c @ Ky_c) / ((n - 1) ** 2)
    var_x = np.trace(Kx_c @ Kx_c) / ((n - 1) ** 2)
    var_y = np.trace(Ky_c @ Ky_c) / ((n - 1) ** 2)
    denom = np.sqrt(var_x * var_y)
    if denom <= 0:
        return 0.0
    return hsic / denom


def fpca_clustering(
    data,
    time_col,
    value_col,
    series_col,
    n_components=5,
    n_clusters=3,
    rng_seed=7,
    normalize=True,
    cluster_method: Literal["gmm", "kmeans"] = "gmm",
):
    if normalize:
        data.loc[:, value_col] = GroupedTransformer(
            StandardScaler(), groups=series_col
        ).fit_transform(data[[series_col, value_col]])[:, 0]

    #  this requires that each series is of equal length and homogeneous
    pivot = data.pivot(index=series_col, columns=time_col, values=value_col)
    pivot = pivot.sort_index(axis=1)
    X = pivot.values

    X_centered = X - X.mean(axis=1, keepdims=True)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_centered)

    if cluster_method == "kmeans":
        # KMeans clustering in feature space
        kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed)
        labels = kmeans.fit_predict(scores)
    elif cluster_method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=rng_seed)
        labels = gmm.fit_predict(scores)

    cluster_d = dict(zip(pivot.index.tolist(), labels))
    clusters = [[] for _ in range(n_clusters)]
    for _id, _clust in cluster_d.items():
        clusters[_clust] = clusters[_clust] + [_id]
    return clusters


def multivariate_fpca_clustering(
    data,
    time_col,
    value_cols,
    series_col,
    n_components=5,
    n_clusters=3,
    rng_seed=7,
    normalize=True,
    cluster_method="gmm",
    n_spline_knots=8,
    spline_degree=3,
    ridge_alpha=1e-3,
):
    # value_cols: list of column names (multivariate)
    # pivot to shape (n_series, n_time, n_vars)
    time_grid = np.sort(data[time_col].unique())
    series_ids = data[series_col].unique()
    n_time = len(time_grid)
    n_vars = len(value_cols)
    series_list = list(series_ids)

    # build array: (n_series, n_time, n_vars)
    X = np.empty((len(series_list), n_time, n_vars))
    for vi, var in enumerate(value_cols):
        pivot = data.pivot(index=series_col, columns=time_col, values=var)
        pivot = pivot.reindex(index=series_list)  # ensure ordering
        pivot = pivot[time_grid]  # ensure time order
        X[:, :, vi] = pivot.values  # assume no missing values

    if normalize:
        # per-variable, per-series z-score (flattened)
        for vi in range(n_vars):
            flat = X[:, :, vi].reshape(-1, 1)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(flat).reshape(X[:, :, vi].shape)
            X[:, :, vi] = scaled

    # build spline basis on time
    transformer = SplineTransformer(
        n_knots=n_spline_knots, degree=spline_degree, include_bias=False
    )
    B = transformer.fit_transform(time_grid.reshape(-1, 1))  # (n_time, n_basis)
    n_basis = B.shape[1]

    # for each series and each variable, fit ridge to get coefficients
    coefs = np.zeros((len(series_list), n_basis * n_vars))
    ridge = Ridge(alpha=ridge_alpha)
    for si in range(len(series_list)):
        for vi in range(n_vars):
            y = X[si, :, vi]
            ridge.fit(B, y)
            coefs[si, vi * n_basis : (vi + 1) * n_basis] = ridge.coef_

    # joint PCA on stacked coefficients
    pca = PCA(n_components=n_components, random_state=rng_seed)
    scores = pca.fit_transform(coefs)

    # clustering
    if cluster_method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=rng_seed)
        labels = model.fit_predict(scores)
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=rng_seed)
        labels = model.fit_predict(scores)

    clusters = [[] for _ in range(n_clusters)]
    for sid, lab in zip(series_list, labels):
        clusters[lab].append(sid)
    return clusters


def multiview_fpca_fusion_clustering(
    data: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    series_col: str,
    n_components_per_view: int = 3,
    n_fused_components: int | None = None,
    n_clusters: int = 3,
    rng_seed: int = 7,
    normalize: bool = True,
    scale_views: bool = True,
    fusion_method: Literal["concat", "pca", "cca", "cka"] = "pca",
    cca_output: Literal["average", "concat"] = "average",
    cluster_method: Literal["gmm", "kmeans"] = "gmm",
) -> List[List]:
    """
    Per-variable FPCA + fusion (multi-view) clustering with optional fusion via PCA, CCA, or CKA.

    Args:
        data: long-form DataFrame containing time series.
        time_col: column name for the time axis.
        value_cols: list of variable columns (each is a univariate series).
        series_col: column identifying each series.
        n_components_per_view: number of PCA components to keep per variable/view.
        n_fused_components: target dimensionality after fusion (used for 'pca' and can be used to reduce
                            the concatenated CKA-weighted / concat embedding). If None, defaults depend on method:
                            - 'pca': uses n_components_per_view
                            - 'concat'/'cka': keeps full fused size
                            - 'cca': uses n_components_per_view unless cca_output='concat' (then doubled)
        n_clusters: number of clusters.
        rng_seed: random seed.
        normalize: if True, each (series, variable) curve is z-scored before FPCA.
        scale_views: if True, each view's score matrix is standardized before fusion.
        fusion_method: one of 'concat', 'pca', 'cca', or 'cka'.
        cca_output: if using 'cca', whether to fuse the two canonical variates by averaging or concatenating.
        cluster_method: 'gmm' or 'kmeans'.

    Returns:
        clusters: list of lists, where each sublist contains the series identifiers assigned to that cluster.
    """
    # Prepare consistent ordering
    time_grid = np.sort(data[time_col].unique())
    series_list = np.sort(data[series_col].unique())
    n_series = len(series_list)

    per_view_scores = []

    for var in value_cols:
        # pivot to (series x time)
        pivot = data.pivot(index=series_col, columns=time_col, values=var)
        pivot = pivot.reindex(index=series_list)  # enforce series order
        pivot = pivot.reindex(columns=time_grid)  # enforce time order

        X = pivot.values  # shape: (n_series, n_time)

        if normalize:
            # per-series z-score (row-wise)
            row_means = X.mean(axis=1, keepdims=True)
            row_stds = X.std(axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0
            X = (X - row_means) / row_stds
        else:
            # center per series (like FPCA)
            X = X - X.mean(axis=1, keepdims=True)

        # PCA on the curve (series are samples, timepoints are features)
        pca_var = PCA(n_components=n_components_per_view, random_state=rng_seed)
        scores_var = pca_var.fit_transform(X)  # (n_series, n_components_per_view)

        if scale_views:
            scaler = StandardScaler()
            scores_var = scaler.fit_transform(scores_var)

        per_view_scores.append(scores_var)

    # Fuse views
    if fusion_method == "concat":
        fused = np.hstack(per_view_scores)
        if n_fused_components is not None:
            pca_fuse = PCA(n_components=n_fused_components, random_state=rng_seed)
            fused_scores = pca_fuse.fit_transform(fused)
        else:
            fused_scores = fused

    elif fusion_method == "pca":
        target = (
            n_fused_components
            if n_fused_components is not None
            else n_components_per_view
        )
        fused_all = np.hstack(per_view_scores)
        pca_fuse = PCA(n_components=target, random_state=rng_seed)
        fused_scores = pca_fuse.fit_transform(fused_all)

    elif fusion_method == "cca":
        if len(per_view_scores) != 2:
            raise ValueError(
                "CCA fusion requires exactly two views (value_cols must have length 2)."
            )
        comp = (
            n_fused_components
            if n_fused_components is not None
            else n_components_per_view
        )
        cca = CCA(n_components=comp)
        U, V = cca.fit_transform(
            per_view_scores[0], per_view_scores[1]
        )  # each is (n_series, comp)
        if cca_output == "average":
            fused_scores = (U + V) / 2
        elif cca_output == "concat":
            fused_scores = np.hstack([U, V])
        else:
            raise ValueError(f"Unsupported cca_output: {cca_output}")

    elif fusion_method == "cka":
        # compute CKA similarities to weight each view
        n_views = len(per_view_scores)
        if n_views == 1:
            fused = per_view_scores[0]
            fused_scores = (
                fused
                if n_fused_components is None
                else PCA(
                    n_components=n_fused_components, random_state=rng_seed
                ).fit_transform(fused)
            )
        else:
            # pairwise CKA matrix
            sim_matrix = np.zeros((n_views, n_views))
            for i in range(n_views):
                for j in range(i, n_views):
                    sim = _linear_cka(per_view_scores[i], per_view_scores[j])
                    sim_matrix[i, j] = sim_matrix[j, i] = sim
            # weight for view i is sum of similarities to others (including self), normalized
            weights = sim_matrix.sum(axis=1)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()  # normalize to sum to 1
            # scale each view's scores by sqrt(weight) to combine
            scaled_views = []
            for w, view in zip(weights, per_view_scores):
                scaled_views.append(np.sqrt(w) * view)
            fused = np.hstack(scaled_views)
            if n_fused_components is not None:
                pca_fuse = PCA(n_components=n_fused_components, random_state=rng_seed)
                fused_scores = pca_fuse.fit_transform(fused)
            else:
                fused_scores = fused
    else:
        raise ValueError(f"Unsupported fusion_method: {fusion_method}")

    # Clustering
    if cluster_method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=rng_seed)
        labels = model.fit_predict(fused_scores)
    elif cluster_method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=rng_seed)
        labels = model.fit_predict(fused_scores)
    else:
        raise ValueError(f"Unsupported cluster_method: {cluster_method}")

    # Build clusters
    clusters = [[] for _ in range(n_clusters)]
    for sid, lab in zip(series_list.tolist(), labels):
        clusters[lab].append(sid)

    return clusters


def clustering_feats(
    data: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    series_col: str,
    interpolate: bool,
    **interpolation_kwargs,
):
    if interpolate:
        df_interp = preprocessing.get_interpolated(
            df,
            id_col=series_col,
            time_col=time_col,
            val_col="eGFR_int",
            keep_t0_value=True,
            time_res=30,
            days_before=0,
            max_days=max_days,
            df_out=True,
        ).dropna()

    df_interp = df_interp.loc[df_interp.ID.isin(index_1) & df_interp.ID.isin(index_2)]
