import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_feasible
from patsy import dmatrix
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import numpy as np


# 1. Build spline basis for time
def build_spline_basis(time, df=5, degree=3):
    """
    time: (T,) array of timestamps (numeric)
    df: number of spline basis functions
    degree: spline degree
    returns: (T, df) design matrix
    """
    dm = dmatrix(f"bs(x, df={df}, degree={degree}, include_intercept=True)", {"x": time}, return_type="dataframe")
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

# 3. Fit model and compute per-series RMSE
def fit_lmm(spline_basis, y, series_idx, n_series, rng_key,
            num_warmup=500, num_samples=1000):
    kernel = NUTS(lmm_model, init_strategy=init_to_feasible)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, spline_basis=spline_basis, y=y, series_idx=series_idx, n_series=n_series)
    post = mcmc.get_samples()
    beta_hat = jnp.mean(post['beta'], axis=0)
    re_hat = jnp.mean(post['re'], axis=0)
    mu_hat = (spline_basis @ beta_hat) + re_hat[series_idx]
    resid = y - mu_hat
    df = pd.DataFrame({'series': series_idx, 'resid2': jnp.square(resid)})
    rmse = df.groupby('series')['resid2'].mean().pow(0.5)
    return rmse

# 4. Iterative clustering by goodness-of-fit
def time_clustering(data, time_col, value_col, series_col,
                    n=3, spline_df=5, spline_degree=3, rng_seed=0):
    clusters = []
    remaining = data.copy()
    rng_key = jax.random.PRNGKey(rng_seed)
    k = 0
    num_size_cluster = int(remaining[series_col].nunique()/n)
    while True:
        series_ids = remaining[series_col].unique()
        n_series = len(series_ids)
        if n_series == 0:
            break
        id_map = {sid: i for i, sid in enumerate(series_ids)}
        inv_id_map = {i: sid for i, sid in enumerate(series_ids)}
        remaining['series_idx'] = remaining[series_col].map(id_map)
        time_grid = onp.sort(remaining[time_col].unique())
        B_grid = build_spline_basis(time_grid, df=spline_df, degree=spline_degree)
        # map each obs to basis row
        time_to_idx = {t: i for i, t in enumerate(time_grid)}
        idxs = remaining[time_col].map(time_to_idx).values
        B_obs = B_grid[idxs]
        rmse = fit_lmm(B_obs, remaining[value_col].to_numpy(),
                       remaining['series_idx'].to_numpy(), n_series, rng_key)
        cut = num_size_cluster
        best_series = rmse.nsmallest(cut).index.tolist()
        best_rmse = rmse.loc[best_series].max()

        res = [*map(inv_id_map.get, best_series)]
        clusters.append(res)
        remaining = remaining[~remaining['series_idx'].isin(best_series)]
        print(f"# series: {n_series}, RMSE: {best_rmse}")

        k += 1
    return clusters


# 5. Generate synthetic data and evaluate clustering
if __name__ == "__main__":
    onp.random.seed(0)
    # Parameters
    n_groups = 3
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
    
    
    # Run clustering
    clusters = time_clustering(df, 'time', 'value', 'id', n=3, spline_df=6, spline_degree=3)
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
    cluster_colors = {c: cmap_(c) for c in range(n_groups)}

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
