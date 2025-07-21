import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_feasible, init_to_uniform, init_to_median, Predictive
from numpyro.ops.indexing import Vindex
from patsy import dmatrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score


def build_spline_basis(time, df=5, degree=3):
    dm = dmatrix(f"bs(x, df={df}, degree={degree}, include_intercept=True)", {"x": time},
                 return_type="dataframe")
    return jnp.array(dm)


# -----------------------------------------------------------------------------
#  Mixture LMM model definition
# -----------------------------------------------------------------------------

def mixture_lmm_model(
    spline_basis: jnp.ndarray,  # (N, P)
    y: jnp.ndarray,            # (N,)
    series_idx: jnp.ndarray,   # (N,)
    n_series: int,
    K: int = 3,
):
    """Hierarchical spline LMM with a mixture over *series*.

    Parameters
    ----------
    spline_basis : (N, P) JAX array
        B‑spline (or other) design matrix evaluated at each observation.
    y : (N,) JAX array
        Observed responses.
    series_idx : (N,) int array
        Maps every observation to a 0‑based series index (length ``n_series``).
    n_series : int
        Number of independent time‑series.
    K : int, optional
        Number of mixture components / clusters (default 3).
    """

    P = spline_basis.shape[1]
    N = spline_basis.shape[0]

    # 1) Mixture weights (softmax of unconstrained logits)
    logits = numpyro.sample("logits", dist.Normal(0, 1).expand([K]).to_event(1))  # (K,)
    mix_probs = jax.nn.softmax(logits)

    # 2) Cluster assignment per *series* (enumerated)
    with numpyro.plate("series", n_series):
        z = numpyro.sample(
            "z",
            dist.Categorical(probs=mix_probs),
            infer={"enumerate": "parallel"},
        )  # shape: (..., n_series)

    # 3) Cluster‑specific spline coefficients and random‑effect SDs
    beta = numpyro.sample(
        "beta", dist.Normal(0, 1).expand([K, P]).to_event(2)  # (K, P)
    )
    sigma_re = numpyro.sample(
        "sigma_re", dist.HalfNormal(1.0).expand([K]).to_event(1)  # (K,) # LogNormal(0, 1)
    )

    # 4) Random intercept for each series (depends on latent cluster)
    with numpyro.plate("series_re", n_series):
        re = numpyro.sample("re", dist.Normal(0.0, sigma_re[z]))  # (..., n_series)

    # ------------------------------------------------------------------
    # Broadcast cluster‑specific parameters to the observation level
    # ------------------------------------------------------------------
    beta_series = Vindex(beta)[z, :]  # (..., n_series, P)
    re_series = re                    # (..., n_series)

    # Index along the n_series axis (−2 for beta_series, −1 for re_series)
    beta_obs = jnp.take(beta_series, series_idx, axis=-2)  # (..., N, P)
    re_obs = jnp.take(re_series, series_idx, axis=-1)      # (..., N)

    # 5) Linear predictor and likelihood
    mu = jnp.sum(spline_basis * beta_obs, axis=-1) + re_obs  # (..., N)

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    with numpyro.plate("obs_plate", N):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def lmm_mixture_model(spline_basis, y, series_idx, n_series):
    """
    Linear mixed model: global spline fixed effects + per-series random spline offsets.
    spline_basis: (N_obs, P)
    y: (N_obs,)
    series_idx: (N_obs,) integers in [0, n_series)
    """
    P = spline_basis.shape[1]
    # Random-intercept-on-coefficients sd
    sigma_re = numpyro.sample("sigma_re", dist.Exponential(1.0))
    # Global spline coefficients
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), 1.0).to_event(1))
    # Per-series deviations in spline coefficients
    with numpyro.plate("series", n_series):
        re = numpyro.sample("re", dist.Normal(jnp.zeros(P), sigma_re).to_event(1))
    # Expected value per observation
    mu = jnp.sum(spline_basis * (beta + re[series_idx]), axis=1)
    # Observation noise
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)



def fit_lmm_and_cluster(data, time_col, value_col, series_col,
                        spline_df=5, spline_degree=3,
                        n_clusters=3, rng_seed=0,
                        num_warmup=500, num_samples=1000):
    """
    1. Fit one LMM across all series.
    2. Extract per-series random effects (re) posterior means.
    3. Cluster those random-effects vectors via KMeans.
    Returns: dict mapping series_id -> cluster label
    """
    # Prepare series indices
    series_ids = data[series_col].unique()
    id_map = {sid: i for i, sid in enumerate(series_ids)}
    data = data.assign(series_idx=data[series_col].map(id_map))
    n_series = len(series_ids)

    # Build spline basis on unique time grid
    time_grid = onp.sort(data[time_col].unique())
    B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree)
    # Map each observation to its row in the basis
    t2i = {t: i for i, t in enumerate(time_grid)}
    idxs = data[time_col].map(t2i).values
    B_obs = B_grid[idxs]

    # Fit via MCMC
    kernel = NUTS(lmm_mixture_model, init_strategy=init_to_feasible)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key,
             spline_basis=B_obs,
             y=data[value_col].to_numpy(),
             series_idx=data['series_idx'].to_numpy(),
             n_series=n_series)
    post = mcmc.get_samples()

    # Compute posterior means
    re_mean = jnp.mean(post['re'], axis=0)  # shape (n_series, P)
    features = onp.array(re_mean)

    # KMeans clustering in feature space
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed)
    labels = kmeans.fit_predict(features)

    return {sid: int(labels[id_map[sid]]) for sid in series_ids}



def fit_mixture_lmm(
    data,
    time_col: str,
    value_col: str,
    series_col: str,
    n_clusters: int = 3,
    spline_df: int = 5,
    spline_degree: int = 3,
    rng_seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 1_000,
    num_chains: int = 1,
):
    """Fits the mixture LMM and returns modal cluster labels per series."""

    # Build a B‑spline basis with patsy
    from patsy import dmatrix

    time_grid = onp.sort(onp.unique(data[time_col].values))
    spline_formula = (
        f"bs(x, df={spline_df}, degree={spline_degree}, "
        "include_intercept=True) - 1"
    )
    B_grid = dmatrix(spline_formula, {"x": time_grid}).astype(float)

    # Design matrix rows corresponding to observed times
    t2i = {t: i for i, t in enumerate(time_grid)}
    idxs = data[time_col].map(t2i).values
    B_obs = B_grid[idxs]

    # Map series IDs to integer indices
    series_ids = data[series_col].unique()
    id_map = {sid: i for i, sid in enumerate(series_ids)}
    series_idx = data[series_col].map(id_map).values

    # Run NUTS
    kernel = NUTS(mixture_lmm_model, init_strategy=init_to_median)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
    )
    rng_key = jax.random.PRNGKey(rng_seed)

    mcmc.run(
        rng_key,
        spline_basis=jnp.asarray(B_obs),
        y=jnp.asarray(data[value_col].values),
        series_idx=jnp.asarray(series_idx),
        n_series=len(series_ids),
        K=n_clusters,
    )

    # Draw posterior discrete labels
    predictive = Predictive(
        mixture_lmm_model,
        posterior_samples=mcmc.get_samples(),
        infer_discrete=True,
        parallel=True,
    )
    z_draws = predictive(
        rng_key,
        spline_basis=jnp.asarray(B_obs),
        y=None,  # only latents
        series_idx=jnp.asarray(series_idx),
        n_series=len(series_ids),
        K=n_clusters,
    )["z"]  # (draws, n_series)

    # Modal label per series
    z_mode = onp.apply_along_axis(
        lambda arr: onp.bincount(arr, minlength=n_clusters).argmax(),
        0,
        onp.asarray(z_draws),
    )

    return {sid: int(z_mode[i]) for sid, i in id_map.items()}


def fpca_clustering(data, time_col, value_col, series_col,
                    n_components=5, n_clusters=3):
    pivot = data.pivot(index=series_col, columns=time_col, values=value_col)
    pivot = pivot.sort_index(axis=1)
    X = pivot.values

    X_centered = X - X.mean(axis=1, keepdims=True)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_centered)

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(scores)

    return dict(zip(pivot.index.tolist(), labels)), pca, scores


if __name__ == "__main__":
    onp.random.seed(0)
    # Parameters
    n_groups = 3
    series_per_group = 100
    n_time = 50
    noise_std = 0.5
    times = onp.linspace(0, 250, n_time)
    records = []    
    spline_params = [
        {'df': 2, 'degree': 1},
        {'df': 3, 'degree': 2},
        {'df': 4, 'degree': 3},
        {'df': 6, 'degree': 3}
    ]

    series_list = []
    true_labels = []

    for g in range(n_groups):
        params = spline_params[g]
        # group-specific spline coefficients        
        B = build_spline_basis(times, df=params['df'], degree=params['degree'])
        coefs = onp.random.randn(params['df']+1)
        
        for s in range(series_per_group):
            vals = B @ coefs + noise_std * onp.random.randn(n_time)
            series_id = f"g{g}_s{s}"
            for t, v in zip(times, vals):
                records.append({'time': float(t), 'value': float(v), 'id': series_id, 'group': g})
            true_labels.append(g)
            series_list.append(series_id)

    df = pd.DataFrame(records)
    
    mix_labels = fit_mixture_lmm(df, 'time', 'value', 'id', n_clusters=n_groups)
    pred_mix = [mix_cluster_labels[s] for s in sorted(mix_labels)]
    ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    print(f"AMI (mixture LMM):    {ami_mix:.3f}")

    mix_cluster_labels = fit_lmm_and_cluster(df, 'time', 'value', 'id', n_clusters=n_groups)
    pred_mix = [mix_cluster_labels[s] for s in sorted(mix_cluster_labels)]
    ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    print(f"AMI (LMM + KMeans):    {ami_mix:.3f}")
    
    fpc_labels, pca, scores = fpca_clustering(df, 'time', 'value', 'id',
                                            n_components=8, n_clusters=n_groups)       
    pred_fpca = [fpc_labels[s] for s in sorted(fpc_labels)]
    ami_fpca = adjusted_mutual_info_score(true_labels, pred_fpca)
    print(f"AMI (FPCA + KMeans): {ami_fpca:.3f}")
    
