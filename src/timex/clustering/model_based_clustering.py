# Using latent analysis to find the best latent dimension for the model
# Python
# https://pypi.org/project/stepmix/
# https://www.statsmodels.org/dev/examples/notebooks/generated/mixed_lm_example.html
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

# R-wrapper
# lcmm: https://cran.r-project.org/web/packages/lcmm/index.html
# flexmix: https://cran.r-project.org/web/packages/flexmix/index.html

# LCMM
# LCGA
# Growing MM

import optax
import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_feasible, init_to_uniform, init_to_median, Predictive
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_value
from numpyro.ops.indexing import Vindex
from numpyro.infer.util import find_valid_initial_params
from patsy import dmatrix
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklego.meta import GroupedTransformer

from typing import Literal

numpyro.set_host_device_count(1)

tau0      = 10.0        # start temperature
tau_min   = 0.1        # end temperature
T_max     = 100       # total VI iterations

def get_tau(step):
    # linear decay:
    frac = min(1.0, step / T_max)
    return tau0 - frac * (tau0 - tau_min)


def build_spline_basis(time, df=5, degree=3, adaptive=False):
    """
    time: (T,) array of timestamps (numeric)
    df: when adaptive=False behaves as before (passed to patsy.bs);
        when adaptive=True, target number of basis functions is df + 1
        and interior knots are placed at quantiles of `time`.
    degree: spline degree
    adaptive: if True, place knots at empirical quantiles (more knots in dense time regions)
    returns: (T, P) design matrix
    """
    if not adaptive:
        dm = dmatrix(f"bs(x, df={df}, degree={degree}, include_intercept=True) -1",
                     {"x": time}, return_type="dataframe")
    else:
        # target number of output columns to mirror original behavior: bs(..., df=df) gave df+1 columns
        n_basis = df + 1  # desired # of basis functions
        # For include_intercept=True, number of columns = #interior_knots + degree + 2
        n_interior = n_basis - degree - 2
        if n_interior < 0:
            raise ValueError(f"df={df} too small for degree={degree} in adaptive mode")
        if n_interior > 0:
            probs = onp.linspace(0, 1, n_interior + 2)[1:-1]  # skip 0 and 1
            knots = onp.quantile(time, probs)
            dm = dmatrix(f"bs(x, degree={degree}, include_intercept=True, knots={list(knots)}) -1",
                         {"x": time}, return_type="dataframe")
        else:
            # no interior knots case: fall back to simple bs with df to get n_basis columns
            dm = dmatrix(f"bs(x, df={df}, degree={degree}, include_intercept=True) -1",
                         {"x": time}, return_type="dataframe")
    return jnp.array(dm)

# 2. Define linear mixed model in NumPyro
def lmm_model(spline_basis, y, series_idx, n_series):
    P = spline_basis.shape[1]
    sigma_re = numpyro.sample("sigma_re", dist.Exponential(1.0))
    with numpyro.plate("series", n_series):
        re = numpyro.sample("re", dist.Normal(0, sigma_re))

    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    mu = jnp.dot(spline_basis, beta) + re[series_idx]
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def lmm_model_sps(spline_basis, y, series_idx, n_series):
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

def lmm_model_random_slopes(spline_basis, y, series_idx, n_series):
    # spline_basis: (N, P), series_idx: (N,), y: (N,)
    N, P = spline_basis.shape

    # global scale for series-level deviations
    sigma_b = numpyro.sample("sigma_b", dist.Exponential(1.0))

    # per-series deviation vector b[i] of length P
    with numpyro.plate("series", n_series):
        b = numpyro.sample("b", dist.Normal(jnp.zeros(P), sigma_b).to_event(1))  # (n_series, P)

    # global shape
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)).to_event(1))  # (P,)

    # observation noise
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # per-observation coefficient: beta + series-specific deviation
    coeffs = beta + b[series_idx]           # (N, P)
    mu = jnp.sum(spline_basis * coeffs, axis=1)  # (N,)

    with numpyro.plate("obs_plate", N):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    

def cluster_from_random_slopes(b_hat, n_clusters, pca_components=5):
    # to numpy
    X = onp.array(b_hat)  # shape (n_series, P)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if pca_components is not None and pca_components < Xs.shape[1]:
        pca = PCA(n_components=pca_components)
        Z = pca.fit_transform(Xs)
    else:
        Z = Xs
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Z)
    return labels  # array of cluster assignments per series index


def mixture_lmm_model_gumbel(
    spline_basis: jnp.ndarray,    # (N, P)
    y: jnp.ndarray,               # (N,)
    series_idx: jnp.ndarray,      # (N,)
    n_series: int,
    K: int = 3,
    tau: float = 0.5,             # Gumbel‑Softmax temperature
):
    P = spline_basis.shape[1]
    N = spline_basis.shape[0]

    # ——— mixture weights ———
    # you can choose Dirichlet or logistic‑normal; here we use Dirichlet
    mix_probs = numpyro.sample("mix_probs", dist.Dirichlet(2.0 * jnp.ones(K)))

    # convert to logits for Gumbel‑Softmax
    logits = jnp.log(mix_probs + 1e-8)  # (K,)

    # ——— cluster‑specific coefficients & random‑effect SDs ———
    beta    = numpyro.sample("beta",    dist.Normal(0, .1)
                                         .expand([K, P]).to_event(2))  # (K, P)
    sigma_re= numpyro.sample("sigma_re",dist.HalfNormal(0.5)
                                         .expand([K]).to_event(1)) + 0.1  # (K,)

    # ——— cluster‑and‑series random intercepts ———
    # we sample a matrix re[k, i] ~ N(0, sigma_re[k]) for k=1..K, i=1..n_series
    re = numpyro.sample(
        "re",
        dist.Normal(0.0, sigma_re[:, None])
            .expand([K, n_series])
            .to_event(2)
    )  # shape: (K, n_series)

    # ——— Gumbel‑Softmax relaxation ———
    # draw Uniform noise U ~ Uniform(0,1) for each series & cluster
    u = numpyro.sample(
        "u",
        dist.Uniform(0, 1)
            .expand([n_series, K])
            .to_event(2)
    )
    gumbel = -jnp.log(-jnp.log(u + 1e-8) + 1e-8)                # (n_series, K)
    z_soft = jax.nn.softmax((logits + gumbel) / tau, axis=-1)  # (n_series, K)

    numpyro.deterministic("z_soft", z_soft)

    # ——— mix cluster parameters per series ———
    # get per‑series spline coeffs and random intercepts
    beta_series  = z_soft @ beta            # (n_series, P)
    re_series    = jnp.sum(z_soft * re.T, axis=-1)  # (n_series,)

    # ——— broadcast to observations ———
    beta_obs = beta_series[series_idx]      # (N, P)
    re_obs   = re_series[series_idx]        # (N,)

    # ——— observation noise ———
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5)) + 0.1

    # ——— linear predictor & likelihood ———
    mu = jnp.sum(spline_basis * beta_obs, axis=-1) + re_obs
    with numpyro.plate("obs_plate", N):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


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
    #logits = numpyro.sample("logits", dist.Normal(0, 0.5).expand([K]).to_event(1))  # (K,)
    #mix_probs = jax.nn.softmax(logits)
    mix_probs = numpyro.sample("mix_probs", dist.Dirichlet(5.0 * jnp.ones(K)))
    # 2) Cluster assignment per *series* (enumerated)
    with numpyro.plate("series", n_series):
        z = numpyro.sample(
            "z",
            dist.Categorical(probs=mix_probs),
            infer={"enumerate": "parallel"},
        )  # shape: (..., n_series)

    # 3) Cluster‑specific spline coefficients and random‑effect SDs
    beta = numpyro.sample( "beta", dist.Normal(0, .1).expand([K, P]).to_event(2))
    sigma_re = numpyro.sample( "sigma_re", dist.HalfNormal(0.5).expand([K]).to_event(1))+0.1

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

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))+0.1

    with numpyro.plate("obs_plate", N):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)



def random_intercept_lmm_model(spline_basis, y, series_idx, n_series):
    P = spline_basis.shape[1]
    N = spline_basis.shape[0]

    beta = numpyro.sample("beta", dist.Normal(0, 0.1).expand([P]).to_event(1))
    
    # Random intercept scale (safe)
    sigma_re = numpyro.sample("sigma_re", dist.HalfNormal(0.5)) + 0.1
    
    # Per-series random intercept
    with numpyro.plate("series", n_series):
        re = numpyro.sample("re", dist.Normal(0, sigma_re))
    
    mu = jnp.sum(spline_basis * beta, axis=-1) + jnp.take(re, series_idx, axis=-1)
    
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5)) + 0.1
    
    with numpyro.plate("obs_plate", N):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


# 3. Fit model and compute per-series RMSE
def lmm_split(spline_basis, y, series_idx, n_series, rng_key,
            num_warmup=500, num_samples=1000):
    kernel = NUTS(lmm_model, init_strategy=init_to_feasible)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, spline_basis=spline_basis, y=y, series_idx=series_idx, n_series=n_series)
    post = mcmc.get_samples()
    beta_hat = jnp.mean(post['beta'], axis=0)
    re_hat = jnp.mean(post['re'], axis=0)
    mu_hat = (spline_basis @ beta_hat) + re_hat[series_idx]
    resid = y - mu_hat
    df = pd.DataFrame({'series': series_idx, 'resid2_plus': jnp.square(resid)*(resid > 0), 
                       'resid3_neg': jnp.square(resid)*(resid < 0)})
    rmse_pos = df.groupby('series')['resid2_plus'].mean().pow(0.5)
    rmse_neg = df.groupby('series')['resid3_neg'].mean().pow(0.5)

    return rmse_pos, rmse_neg

def lmm_slopes_res(spline_basis, y, series_idx, n_series, rng_key,
                          num_warmup=500, num_samples=1000):
    kernel = NUTS(lmm_model_random_slopes, init_strategy=init_to_feasible)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, spline_basis=spline_basis, y=y,
             series_idx=series_idx, n_series=n_series)
    post = mcmc.get_samples()
    b_hat = jnp.mean(post['b'], axis=0)  # (n_series, P): per-series shape deviations
    beta_hat = jnp.mean(post['beta'], axis=0)

    return b_hat, beta_hat, post


###########################################################################################################
###########################################################################################################


def clustering_mm(data, time_col, value_col, series_col,
                    n_clusters=4, spline_df=5, spline_degree=3, rng_seed=0, 
                    adaptive_spline=True, how: Literal['normal', 'slope']='normal',
                    num_warmup=500, num_samples=1500, normalize=True, direction='outwards', 
                    cluster_method='gmm', pca_components=10):
    # check if n_clusters is even number
    # TODO: timegrid is now set to use all unique timepoints, add warning if this number is high and suggest use
    #  of interpolation and homogeneous grid

    assert (n_clusters % 2 == 0), "n_clusters should be even"

    clusters = []

    if normalize:
        remaining = data.copy()
        remaining.loc[:, value_col] = GroupedTransformer(StandardScaler(), groups=series_col).fit_transform(data[[series_col, value_col]])[:,0]
        data = remaining
    else:
        remaining = data.copy()
    rng_key = jax.random.PRNGKey(rng_seed)
    k = 0
    if how=='normal':
        num_size_cluster = int(remaining[series_col].nunique()/n_clusters)
        while True:
            series_ids = remaining[series_col].unique()
            n_series = len(series_ids)
            
            if n_series == 0:
                break
            
            if len(clusters)>=n_clusters:
                # put remaining ids in remainder_cluster
                print(f"Assigned {n_clusters} clusters, assigning remaning {n_series} series to remainder clusters")
                clusters.append(list(series_ids))
                return clusters 

            id_map = {sid: i for i, sid in enumerate(series_ids)}
            inv_id_map = {i: sid for i, sid in enumerate(series_ids)}
            remaining['series_idx'] = remaining[series_col].map(id_map)
            time_grid = onp.sort(remaining[time_col].unique())
            B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)
            # map each obs to basis row
            time_to_idx = {t: i for i, t in enumerate(time_grid)}
            idxs = remaining[time_col].map(time_to_idx).values
            B_obs = B_grid[idxs]

            print(f'Running LMM splitter..', flush=True)
            rmse_p, rmse_n = lmm_split(B_obs, remaining[value_col].to_numpy(),
                        remaining['series_idx'].to_numpy(), n_series, rng_key, num_samples=num_samples)
            
            cut = num_size_cluster
            
            if rmse_p.shape[0]>0:
                if direction == 'outwards':
                    best_series_p = rmse_p.nsmallest(cut).index.tolist()
                elif direction == 'inwards':
                    best_series_p = rmse_p.nlargest(cut).index.tolist()
                best_rmse_p = rmse_p.loc[best_series_p].min()
                res_p = [*map(inv_id_map.get, best_series_p)]      
                print(f"cl pos: {len(res_p)}")
                clusters.append(res_p)
            else:
                best_series_p = []

            if rmse_n.shape[0]>0:
                if direction == 'outwards':
                    best_series_n = rmse_n.nsmallest(cut).index.tolist()
                elif direction == 'inwards':
                    best_series_n = rmse_n.nlargest(cut).index.tolist()
                best_rmse_n = rmse_n.loc[best_series_n].min()
                res_n = [*map(inv_id_map.get, best_series_n)]      
                print(f"cl neg: {len(res_n)}")
                clusters.append(res_n)
            else:
                best_series_n = []

            remaining = remaining[(~remaining['series_idx'].isin(best_series_n)) 
                                & (~remaining['series_idx'].isin(best_series_p))]
            print(f"# series: {n_series}, remaining: {remaining[series_col].nunique()}, RMSE_p: {best_rmse_p}, RMSE_n: {best_rmse_n}")

            k += 1
    elif how=='slope':
        series_ids = data[series_col].unique()
        n_series = len(series_ids)
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        inv_id_map = {i: sid for i, sid in enumerate(series_ids)}
        data['series_idx'] = data[series_col].map(id_map)
        
        time_grid = onp.sort(data[time_col].unique())
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)
        # map each obs to basis row
        time_to_idx = {t: i for i, t in enumerate(time_grid)}
        idxs = data[time_col].map(time_to_idx).values
        B_obs = B_grid[idxs]

        b_hat, _, _ = lmm_slopes_res(B_obs, data[value_col].to_numpy(),
                        data['series_idx'].to_numpy(), n_series, rng_key, num_samples=1500)
        print(b_hat.shape, flush=True)
        # to numpy
        X = onp.array(b_hat)  # shape (n_series, P)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if pca_components is not None and pca_components < Xs.shape[1]:
            pca = PCA(n_components=pca_components)
            Z = pca.fit_transform(Xs)
        else:
            Z = Xs
        
        if cluster_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        elif cluster_method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=0)

        _clusters = clusterer.fit_predict(Z)

        cluster_d = dict(zip([*map(inv_id_map.get, data['series_idx'].to_numpy())] , _clusters))

        clusters = [[] for _ in range(n_clusters)]
        for _id, _clust in cluster_d.items():
            clusters[_clust] =  clusters[_clust] + [_id]
    elif how=='random_effect_clustering':
        """
        1. Fit one LMM across all series.
        2. Extract per-series random effects (re) posterior means.
        3. Cluster those random-effects vectors via KMeans.
        Returns: dict mapping series_id -> cluster label
        """
        series_ids = data[series_col].unique()
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        data = data.assign(series_idx=data[series_col].map(id_map))
        n_series = len(series_ids)

        # Build spline basis on unique time grid
        time_grid = onp.sort(data[time_col].unique())
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)
        # Map each observation to its row in the basis
        t2i = {t: i for i, t in enumerate(time_grid)}
        idxs = data[time_col].map(t2i).values
        B_obs = B_grid[idxs]
        
        kernel = NUTS(lmm_model_sps, init_strategy=init_to_feasible)
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

        if cluster_method=='kmeans':
            # KMeans clustering in feature space
            kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed)
            labels = kmeans.fit_predict(features)
        elif cluster_method=='gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=0)
            labels = gmm.fit_predict(features)

        cluster_d =  {sid: int(labels[id_map[sid]]) for sid in series_ids}
        clusters = [[] for _ in range(n_clusters)]
        for _id, _clust in cluster_d.items():
            clusters[_clust] =  clusters[_clust] + [_id]
    elif how=='random_intercept_clustering':
        # === Prepare design matrices ===
        series_ids = data[series_col].unique()
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        data = data.assign(series_idx=data[series_col].map(id_map))
        n_series = len(series_ids)

        time_grid = onp.sort(data[time_col].unique())
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)
        t2i = {t: i for i, t in enumerate(time_grid)}
        idxs = data[time_col].map(t2i).values
        B_obs = jnp.array(B_grid[idxs])
        y_obs = jnp.array(data[value_col].to_numpy())
        series_idx_arr = jnp.array(data["series_idx"].to_numpy())

        # === Stage 1: Fit hierarchical LMM ===
        P = B_obs.shape[1]
        init_vals = {
            "beta": 0.05 * jnp.ones(P),
            "sigma_re": jnp.array(0.5),
            "sigma": jnp.array(1.0),
        }
        init_strategy = init_to_value(values=init_vals)

        kernel = NUTS(random_intercept_lmm_model, init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        rng_key = jax.random.PRNGKey(rng_seed)
        mcmc.run(rng_key, B_obs, y_obs, series_idx_arr, n_series)
        posterior = mcmc.get_samples()

        # Extract per-series random effects mean & SD
        re_mean = onp.array(jnp.mean(posterior["re"], axis=0))
        re_sd   = onp.array(jnp.std(posterior["re"], axis=0))
        features = onp.column_stack([re_mean, re_sd])  # shape (n_series, 2)

        # === Stage 2: Cluster in feature space ===
        if cluster_method=='kmeans':
            # KMeans clustering in feature space
            kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed)
            labels = kmeans.fit_predict(features)
        elif cluster_method=='gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=0)
            labels = gmm.fit_predict(features)

        cluster_d = {sid: int(labels[idx]) for sid, idx in id_map.items()}
        clusters = [[] for _ in range(n_clusters)]
        for _id, _clust in cluster_d.items():
            clusters[_clust] =  clusters[_clust] + [_id]

    return clusters



def clustering_lmm(
    data,
    time_col: str,
    value_col: str,
    series_col: str,
    n_clusters: int = 3,
    spline_df: int = 5,
    spline_degree: int = 3,
    rng_seed: int = 0,
    num_steps: int = 2_000,
    num_warmup: int = 500,
    num_samples: int = 1_500,
    num_chains: int = 1,
    discrete: bool = False,
    normalize: bool = True,
    adaptive_spline: bool = True,
    tau_search: bool = True,
    tau: float = 0.5,
    vi: bool= False
):
    
    if normalize:
        data.loc[:, value_col] = GroupedTransformer(StandardScaler(), groups=series_col).fit_transform(data[[series_col, value_col]])[:,0]

    # TODO: timegrid is now set to use all unique timepoints, add warning if this number is high and suggest use
    #  of interpolation and homogeneous grid

    if vi == False:
        """Fits the mixture LMM and returns modal cluster labels per series."""

        # Build a B‑spline basis with patsy        

        time_grid = onp.sort(onp.unique(data[time_col].values))
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)

        # Design matrix rows corresponding to observed times
        t2i = {t: i for i, t in enumerate(time_grid)}
        idxs = data[time_col].map(t2i).values
        B_obs = B_grid[idxs]

        # Map series IDs to integer indices
        series_ids = data[series_col].unique()
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        series_idx = data[series_col].map(id_map).values

        print("Characteristics of B_obs")
        print("min", "max", "cond#")
        print(jnp.min(B_obs), jnp.max(B_obs), jnp.linalg.cond(B_obs.T @ B_obs))

        # Fit via MCMC
        ################################################
        val_dict = {
            "logits": 1/n_clusters*jnp.ones(n_clusters),        # balanced mixture
            "beta": 0.05*jnp.ones((n_clusters, spline_df)),     # near zero coefficients
            "sigma_re": jnp.ones(n_clusters)*0.5,   # moderate RE SD
            "sigma": jnp.array(1.0),       # reasonable obs noise
        }
        init_strategy = init_to_value(values=val_dict)

        ################################################
        # print("Performing SVI for initialisation...", flush=True)
        # guide = AutoDiagonalNormal(mixture_lmm_model)
        # optimizer = optax.adam(1e-4)  # or numpyro.optim.Adam(1e-2)

        # svi = SVI(mixture_lmm_model, guide, optimizer, loss=Trace_ELBO())

        # # Get the actual obs + series_idx
        # y_obs = data[value_col].to_numpy()
        # series_idx_arr = jnp.array(series_idx)

        # # Now run SVI
        # state = svi.run(
        #     jax.random.PRNGKey(rng_seed),
        #     500,                  # num_steps
        #     B_obs,                # spline_basis
        #     y_obs,                # y
        #     series_idx_arr,       # series_idx
        #     len(series_ids),      # n_series
        #     n_clusters            # K
        # )

        # print("Setting initial parameters...", flush=True)
        # init_params = guide.sample_posterior(
        #     jax.random.PRNGKey(rng_seed), 
        #     state.params
        # )
        # init_strategy = init_to_value(values=init_params)
        ###############################################
        # y_obs = data[value_col].to_numpy()
        # series_idx_arr = jnp.array(series_idx)
        # rng_key = jax.random.PRNGKey(0)
        # print("Searching for valid initial parameters...", flush=True)
        # init_params, pe = find_valid_initial_params(
        #     rng_key,
        #     mixture_lmm_model,
        #     init_strategy=init_to_median,
        #     model_args= [B_obs, y_obs,series_idx_arr, len(series_ids), n_clusters]  
        # )
        # print("Found valid init log_prob:", pe)
        # init_strategy = init_to_value(values=init_params)

        print("Defining kernel...", flush=True)
        if discrete:
            mod = mixture_lmm_model
            kernel = NUTS(mixture_lmm_model, init_strategy=init_strategy)
            kwargs = {}
        else:
            mod = mixture_lmm_model_gumbel
            kernel = NUTS(mixture_lmm_model_gumbel, init_strategy=init_strategy)
            kwargs = {'tau': tau}

        print("Init sampler...", flush=True)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="parallel",        
        )
        rng_key = jax.random.PRNGKey(rng_seed)

        print("Start sampler...", flush=True)
        mcmc.run(
            rng_key,
            spline_basis=jnp.asarray(B_obs),
            y=jnp.asarray(data[value_col].values),
            series_idx=jnp.asarray(series_idx),
            n_series=len(series_ids),
            K=n_clusters,
            **kwargs
        )

        # Draw posterior discrete labels
        print("Start posterior draws...", flush=True)
        predictive = Predictive(
            mod,
            posterior_samples=mcmc.get_samples(),
            return_sites = ['z_soft'],
            infer_discrete=discrete,
            parallel=True,
        )
        z_draws = predictive(
            rng_key,
            spline_basis=jnp.asarray(B_obs),
            y=None,  # only latents
            series_idx=jnp.asarray(series_idx),
            n_series=len(series_ids),
            K=n_clusters,
        )  # (draws, n_series)

        if discrete:
            # Modal label per series
            z_mode = onp.apply_along_axis(
                lambda arr: onp.bincount(arr, minlength=n_clusters).argmax(),
                0,
                onp.asarray(z_draws['z'])
            )
        else:
            z_soft_samples = z_draws["z_soft"] # shape: (num_samples, n_series, K)
            z_hard_samples = onp.argmax(z_soft_samples, axis=-1)
            z_mode =[
                        onp.bincount(z_hard_samples[:, i], minlength=n_clusters).argmax()
                        for i in range(len(series_ids))
                    ]

        cluster_d = {sid: int(z_mode[i]) for sid, i in id_map.items()}
        clusters = [[] for _ in range(n_clusters)]
        for _id, _clust in cluster_d.items():
            clusters[_clust] =  clusters[_clust] + [_id]
    else:
        # Build spline basis as before...
        time_grid = onp.sort(onp.unique(data[time_col].values))
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree, adaptive=adaptive_spline)

        t2i = {t: i for i, t in enumerate(time_grid)}
        idxs = data[time_col].map(t2i).values
        B_obs = B_grid[idxs]

        # Map series IDs to integer indices
        series_ids = data[series_col].unique()
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        series_idx = data[series_col].map(id_map).values

        # Variational inference
        scheduler = optax.linear_schedule(
            init_value=1e-2,
            end_value=1e-5,
            transition_steps=num_steps,   # match your num_steps
        )
        optimizer = optax.adam(scheduler)

        # optimizer = optax.chain(
        # optax.clip(1.0),        # clip global norm to 1.0
        # optax.adam(scheduler),
        # )

        if discrete:
            mod = mixture_lmm_model
            kwargs = {}
        else:
            mod = mixture_lmm_model_gumbel
            kwargs = {'tau': tau}

        guide = AutoDiagonalNormal(mod)
        svi = SVI(mod, guide, optimizer, loss=Trace_ELBO())
        
        rng_key = jax.random.PRNGKey(rng_seed)

        # Run VI
        if tau_search:
            svi_state = svi.init(rng_key,            
                spline_basis=jnp.asarray(B_obs),
                y=jnp.asarray(data[value_col].values),
                series_idx=jnp.asarray(series_idx),
                n_series=len(series_ids),
                K=n_clusters,
                **kwargs
            )
            for step in range(num_steps):
                current_tau = get_tau(step)      # if you’re annealing
                svi_state, loss = svi.update(
                    svi_state, spline_basis=B_obs, y=jnp.asarray(data[value_col].values),
                    series_idx=series_idx, n_series=len(series_ids), K=n_clusters, tau=current_tau
                )
                if step % 100 == 0:
                    print(f"[{step:4d}] ELBO = {-loss:.1f}, τ = {current_tau:.3f}")
        else:
            svi_result = svi.run(
                rng_key,
                num_steps=num_steps,
                spline_basis=jnp.asarray(B_obs),
                y=jnp.asarray(data[value_col].values),
                series_idx=jnp.asarray(series_idx),
                n_series=len(series_ids),
                K=n_clusters,
                **kwargs
            )

        # Approx posterior
        params = svi_result.params
        predictive = Predictive(mod, 
                                guide=guide, 
                                params=params, 
                                return_sites = ['z_soft'],
                                num_samples=500, 
                                infer_discrete=discrete)
        z_draws = predictive(
            rng_key,
            spline_basis=jnp.asarray(B_obs),
            y=None,
            series_idx=jnp.asarray(series_idx),
            n_series=len(series_ids),
            K=n_clusters,
        )

        if discrete:
            # Modal label per series
            z_mode = onp.apply_along_axis(
                lambda arr: onp.bincount(arr, minlength=n_clusters).argmax(),
                0,
                onp.asarray(z_draws['z'])
            )
        else:
            z_soft_samples = z_draws["z_soft"] # shape: (num_samples, n_series, K)
            print(z_soft_samples.shape)
            z_hard_samples = onp.argmax(z_soft_samples, axis=-1)
            z_mode =[
                        onp.bincount(z_hard_samples[:, i], minlength=n_clusters).argmax()
                        for i in range(len(series_ids))
                    ]

        cluster_d = {sid: int(z_mode[i]) for sid, i in id_map.items()}
        clusters = [[] for _ in range(n_clusters)]
        for _id, _clust in cluster_d.items():
            clusters[_clust] =  clusters[_clust] + [_id]

    return clusters


# 5. Generate synthetic data and evaluate clustering
if __name__ == "__main__":
    onp.random.seed(0)
    # Parameters
    n_groups = 4
    series_per_group = 25
    n_time = 50
    noise_std = 0.1
    times = onp.linspace(0, 250, n_time)
    print(times[:10])
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
    
    
    print("Testing MM clustering...")
    # Run clustering
    clusters = clustering_mm(df, 'time', 'value', 'id', n_clusters=n_groups, spline_df=6, spline_degree=3)
    # Map series to predicted cluster index
    pred_labels = {}
    for cluster_idx, group in enumerate(clusters):
        for sid in group:
            pred_labels[sid] = cluster_idx
    pred = []
    for sid in sorted(pred_labels):
        pred.append(pred_labels[sid])
    # Compute AMI
    ami = adjusted_mutual_info_score(true_labels, pred)
    print(f"Adjusted Mutual Information: {ami:.3f}")

    cmap_ = plt.get_cmap('tab10', n_groups)
    cluster_colors = {c: cmap_(c) for c in range(len(clusters))}

    fig, axes = plt.subplots(nrows=2, figsize=(17, 10))
    # Plot 1: colored by true series
    ax1 = axes[0]
    for k, sid in enumerate(series_list):
        sub = df[df['id'] == sid]
        ax1.plot(sub['time'], sub['value'], label=true_labels[k], color=cluster_colors[true_labels[k]])
    ax1.set_title('Synthetic Timeseries by Series')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right', ncol=3, fontsize='small')

    # Plot 2: colored by predicted cluster
    ax2 = axes[1]
    for sid in series_list:
        sub = df[df['id'] == sid]
        c = pred_labels[sid]
        ax2.plot(sub['time'], sub['value'], label=c, color=cluster_colors[c])
    ax2.set_title('Synthetic Timeseries by Predicted Cluster')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


    # mix_labels = fit_lcmm(df, 'time', 'value', 'id', n_clusters=n_groups)
    # pred_mix = [mix_cluster_labels[s] for s in sorted(mix_labels)]
    # ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    # print(f"AMI (LCMM):    {ami_mix:.3f}")
    
    mix_cluster_labels = fit_lcmm_vi(df, 'time', 'value', 'id', n_clusters=n_groups, discrete=False, tau_search=False)
    pred_mix = [mix_cluster_labels[s] for s in sorted(mix_cluster_labels)]
    ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    print(f"AMI (LCMM VI):    {ami_mix:.3f}")

    # mix_cluster_labels = fit_lcmm(df, 'time', 'value', 'id', n_clusters=n_groups, discrete=False)
    # pred_mix = [mix_cluster_labels[s] for s in sorted(mix_cluster_labels)]
    # ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    # print(f"AMI (LCMM):    {ami_mix:.3f}")

    # mix_cluster_labels = two_stage_growth_mixture(df, 'time', 'value', 'id', n_clusters=n_groups)
    # pred_mix = [mix_cluster_labels[s] for s in sorted(mix_cluster_labels)]
    # ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    # print(f"AMI (2sLMM + GMM):    {ami_mix:.3f}")

    # mix_cluster_labels = fit_lmm_and_cluster(df, 'time', 'value', 'id', n_clusters=n_groups)
    # pred_mix = [mix_cluster_labels[s] for s in sorted(mix_cluster_labels)]
    # ami_mix = adjusted_mutual_info_score(true_labels, pred_mix)
    # print(f"AMI (LMM + GMM):    {ami_mix:.3f}")
    
    # fpc_labels, pca, scores = fpca_clustering(df, 'time', 'value', 'id',
    #                                         n_components=10, n_clusters=n_groups)       
    # pred_fpca = [fpc_labels[s] for s in sorted(fpc_labels)]
    # ami_fpca = adjusted_mutual_info_score(true_labels, pred_fpca)
    # print(f"AMI (FPCA + GMM): {ami_fpca:.3f}")
    
