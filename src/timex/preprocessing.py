####################
## Pre-processing ##
####################

import gc
import os
import sys
from collections import defaultdict
from time import sleep
from typing import Dict, List, Literal, Optional

# from scipy.stats import skew, kurtosis
# from scipy.fft import rfft, rfftfreq
# from scipy.signal import cwt, ricker
import numpy as np
import pandas as pd
import tsfresh
from scipy import interpolate, ndimage, signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklego.meta.grouped_transformer import GroupedTransformer
from tqdm import tqdm
from tslearn import metrics
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax


def normalise_ts(
    ts_df,
    id_col="ID",
    time_col="Time_days",
    val_col="eGFR_CKDEpi2012",
    df_out=False,
    scaler: Literal["standard", "minmax"] = "standard",
):
    if scaler not in {"standard", "minmax"}:
        raise ValueError("scaler must be 'standard' or 'minmax'")

    res = {id_col: [], time_col: [], val_col: []}
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[id_col, time_col, val_col]].copy()

        vals = _df[val_col].to_numpy(dtype=float)
        finite_mask = np.isfinite(vals)
        scaled = np.full(vals.shape, np.nan, dtype=float)

        if finite_mask.any():
            x = vals[finite_mask].reshape(-1, 1)
            if scaler == "standard":
                scaler_model = StandardScaler()
            else:
                scaler_model = MinMaxScaler()
            scaled[finite_mask] = scaler_model.fit_transform(x).reshape(-1)

        _df[val_col] = scaled

        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)

    if df_out:
        return pd.DataFrame(res)
    else:
        return res


def get_filtered_df(
    ts_df, id_col="ID", time_col="Time_days", min_time=365, min_measurements=3
):
    maxts = ts_df.groupby(id_col)[time_col].max()
    ltids = maxts[maxts > min_time].index

    cnts = ts_df[ts_df[id_col].isin(ltids)].groupby(id_col).size()
    filtered_ids = cnts[cnts > min_measurements].index

    ts = ts_df.loc[ts_df[id_col].isin(filtered_ids)]
    return ts


def get_interpolated(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    time_before: int = 0,
    max_time: int | None = 365,
    time_res: int = 7,
    keep_t0_value: bool = False,
    df_out: bool = False,
) -> pd.DataFrame | Dict[str, List]:
    """
    Extract interpolated time series data.

    :param ts_df: pd.DataFrame, time series data
    :param id_col: str, default 'ID'
    :param time_col: str, time column
    :param val_col: str, default 'eGFR_CKDEpi2012'
    :param days_before: int, how many days before t0 to include
    :param max_time: int, maximum number of days to include
    :param time_res: int, time resolution of time series
    :param keep_t0_value: boolean, whether to keep t0 value
    :param df_out: boolean, whether to output dataframe
    """

    # TODO: implementation version where each ID has it's own max_time
    if max_time is None:
        finite_time = ts_df[time_col].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_time.empty:
            raise ValueError("No finite time values available for interpolation")
        max_time = int(np.ceil(finite_time.max())) + int(time_res)

    trange = np.arange(-time_before, max_time, time_res)
    if (keep_t0_value) & (time_before > 0):
        trange = np.insert(trange, 1, 0)

    res = {id_col: [], time_col: [], val_col: []}
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[time_col, val_col]].copy()

        # PCHIP requires finite values and strictly increasing x.
        _df = _df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, val_col])
        _df = _df.sort_values(by=time_col)

        # collapse duplicated time points to one value
        _df = _df.groupby(time_col, as_index=False)[val_col].mean()

        if _df.shape[0] < 2:
            interpolated = np.full(shape=len(trange), fill_value=np.nan, dtype=float)
        else:
            x = _df[time_col].to_numpy(dtype=float)
            y = _df[val_col].to_numpy(dtype=float)
            interpolator = interpolate.PchipInterpolator(x, y, extrapolate=False)
            interpolated = interpolator(trange)

        ids = len(trange) * [_id]

        res[id_col].extend(ids)
        res[time_col].extend(trange)
        res[val_col].extend(interpolated)

    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res


def get_smoothed(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    Nskip: int = 3,
    df_out: bool = False,
    smoothing_method: Literal[
        "gaussian_kernel", "gaussian_kernel_simple", "box_kernel", "rolling_mean"
    ] = "gaussian_kernel",
) -> pd.DataFrame | Dict[str, List]:
    if smoothing_method == "gaussian_kernel":
        return get_smoothed_gaussian_kernel(
            ts_df, id_col, time_col, val_col, window, Nskip, df_out
        )
    elif smoothing_method == "gaussian_kernel_simple":
        return get_smoothed_gaussian_kernel_simple(
            ts_df, id_col, time_col, val_col, window, Nskip, df_out
        )
    elif smoothing_method == "box_kernel":
        return get_smoothed_box_kernel(
            ts_df, id_col, time_col, val_col, window, Nskip, df_out
        )
    elif smoothing_method == "rolling_mean":
        return get_smoothed_rolling_mean(
            ts_df, id_col, time_col, val_col, window, Nskip, df_out
        )

    return


# TODO: ensure that first N-points are excluded from smoothing
def get_smoothed_rolling_mean(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    Nskip: int = 3,
    df_out: bool = False,
):
    res = {id_col: [], time_col: [], val_col: []}

    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)
        if Nskip > 0:
            r = range(Nskip, _df.shape[0] - Nskip)
        else:
            r = range(_df.shape[0])
        _df.iloc[r, 2] = (
            _df[val_col].rolling(window=window, min_periods=0).mean().values[r]
        )

        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)
    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res


# TODO: ensure that first N-points are excluded from smoothing
def get_smoothed_gaussian_kernel(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    Nskip: int = 3,
    df_out: bool = False,
):
    res = {id_col: [], time_col: [], val_col: []}

    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)

        # Handle NaN values
        values = _df[val_col].values.copy()
        nan_mask = np.isnan(values)

        if not nan_mask.all():  # Only smooth if there are non-NaN values
            # Convert window size to sigma (approximately window/6 for 99% coverage)
            # For large windows, use a smaller ratio to avoid excessive smoothing
            if window > 30:
                sigma = window / 10.0
            else:
                sigma = window / 6.0

            # Interpolate NaN values temporarily for smoothing
            if nan_mask.any():
                # Simple linear interpolation for NaN values
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) >= 2:
                    # Interpolate internal NaNs
                    interp_func = interpolate.interp1d(
                        valid_indices,
                        values[valid_indices],
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    all_indices = np.arange(len(values))
                    interpolated_values = interp_func(all_indices)

                    # Apply smoothing to interpolated data
                    smoothed_values = ndimage.gaussian_filter1d(
                        interpolated_values,
                        sigma,
                        mode="nearest",  # Handle edges better
                    )

                    # Restore NaN values where they originally were
                    smoothed_values[nan_mask] = np.nan
                else:
                    # Not enough valid points to interpolate
                    smoothed_values = values
            else:
                # No NaN values, smooth directly
                smoothed_values = ndimage.gaussian_filter1d(
                    values, sigma, mode="nearest"
                )
        else:
            # All values are NaN
            smoothed_values = values

        if Nskip > 0:
            r = range(Nskip, _df.shape[0] - Nskip)
        else:
            r = range(_df.shape[0])

        _df.iloc[r, 2] = smoothed_values[r]

        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)
    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res


def get_smoothed_gaussian_kernel_simple(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    Nskip: int = 3,
    df_out: bool = False,
):
    """
    Simple Gaussian kernel smoothing with direct window size control.

    :param ts_df: DataFrame with time series data
    :param id_col: ID column name
    :param time_col: Time column name
    :param val_col: Value column name
    :param window: Window size in timesteps (actual kernel width)
    :param Nskip: Number of points to skip at start/end
    :param df_out: Whether to return DataFrame
    """
    res = {id_col: [], time_col: [], val_col: []}

    # Create Gaussian kernel with specified window size
    # Window should be odd for symmetry
    if window % 2 == 0:
        window += 1

    # Create kernel - sigma chosen so 99% of weight is within window
    sigma = window / 6.0
    kernel_range = np.arange(-window // 2 + 1, window // 2 + 1)
    kernel = np.exp(-(kernel_range**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize

    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)

        values = _df[val_col].values.copy()
        smoothed_values = np.empty_like(values)
        smoothed_values[:] = np.nan

        # Apply convolution manually to handle edges and NaN properly
        half_window = window // 2

        for i in range(len(values)):
            # Define window boundaries
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)

            # Get values in window
            window_vals = values[start:end]

            # Get corresponding kernel weights (adjusted for edge cases)
            kernel_start = max(0, half_window - i)
            kernel_end = kernel_start + (end - start)
            window_kernel = kernel[kernel_start:kernel_end]

            # Handle NaN values - only use non-NaN values
            valid_mask = ~np.isnan(window_vals)

            if valid_mask.any():
                # Renormalize kernel for valid values only
                valid_kernel = window_kernel[valid_mask]
                valid_kernel = valid_kernel / valid_kernel.sum()

                # Compute weighted average
                smoothed_values[i] = np.sum(window_vals[valid_mask] * valid_kernel)
            else:
                smoothed_values[i] = np.nan

        # Apply Nskip
        if Nskip > 0:
            r = range(Nskip, _df.shape[0] - Nskip)
        else:
            r = range(_df.shape[0])

        # Keep original values outside skip range
        final_values = values.copy()
        final_values[r] = smoothed_values[r]

        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(final_values)

    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res


def get_smoothed_box_kernel(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    Nskip: int = 3,
    df_out: bool = False,
):
    """
    Simple box kernel (uniform) smoothing with direct window size control.
    This is equivalent to a moving average with equal weights.

    :param ts_df: DataFrame with time series data
    :param id_col: ID column name
    :param time_col: Time column name
    :param val_col: Value column name
    :param window: Window size in timesteps (number of points to average)
    :param Nskip: Number of points to skip at start/end
    :param df_out: Whether to return DataFrame
    """
    res = {id_col: [], time_col: [], val_col: []}

    # Window should be odd for symmetry
    if window % 2 == 0:
        window += 1

    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col] == _id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)

        values = _df[val_col].values.copy()
        smoothed_values = np.empty_like(values)
        smoothed_values[:] = np.nan

        half_window = window // 2

        for i in range(len(values)):
            # Define window boundaries
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)

            # Get values in window
            window_vals = values[start:end]

            # Handle NaN values - only use non-NaN values
            valid_vals = window_vals[~np.isnan(window_vals)]

            if len(valid_vals) > 0:
                # Simple average of valid values
                smoothed_values[i] = np.mean(valid_vals)
            else:
                smoothed_values[i] = np.nan

        # Apply Nskip
        if Nskip > 0:
            r = range(Nskip, _df.shape[0] - Nskip)
        else:
            r = range(_df.shape[0])

        # Keep original values outside skip range
        final_values = values.copy()
        final_values[r] = smoothed_values[r]

        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(final_values)

    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res


# function get_smoothed_savgol(ts, window=self.smoothing_window_size, polyorder=self.smoothing_polyorder, df_out=True, **id_kwargs)
def get_smoothed_savgol(
    ts_df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "Time_days",
    val_col: str = "eGFR_CKDEpi2012",
    window: int = 5,
    polyorder: int = 7,
    Nskip: int = 3,
    df_out: bool = True,
):
    smoothed_values = signal.savgol_filter(ts, window, polyorder)

    if Nskip > 0:
        r = range(Nskip, len(_df) - Nskip)
    else:
        r = range(len(_df))

    if df_out:
        return pd.DataFrame(smoothed_values, columns=["smoothed"])
    else:
        return smoothed_values


# perform low pass filtering
def get_low_pass_filtered(
    ts_dict: dict,
    id_col="ID",
    time_col="Time_days",
    val_col="eGFR_CKDEpi2012",
    cutoff=0.25,
    stationary=True,
    order=1,
    btype="low",
    window=3,
    df_out=False,
):
    res = {id_col: [], time_col: [], val_col: []}
    """
    Should normaly only be applied for uniformly sampled time series that
    are stationary and have a constant sampling rate.
    If stationary is set to False
    """

    if stationary:
        for _id in tqdm(ts_dict[id_col].unique()):
            _df = ts_dict.loc[ts_dict[id_col] == _id][[id_col, time_col, val_col]]
            _df = _df.sort_values(by=time_col)
            vals = _df[val_col].values
            filtered = signal.butter(
                order, cutoff, btype="low", analog=False, output="sos"
            )
            filtered = signal.sosfilt(filtered, vals)

            res[id_col].extend(_df[id_col].values)
            res[time_col].extend(_df[time_col].values)
            res[val_col].extend(filtered)
    else:
        for _id in tqdm(ts_dict[id_col].unique()):
            _df = ts_dict.loc[ts_dict[id_col] == _id][[id_col, time_col, val_col]]
            _df = _df.sort_values(by=time_col)

            vals = _df[val_col].values
            trend = ndimage.gaussian_filter1d(vals, window)
            trend[-window:] = vals[-window:]

            detrended = vals - trend
            passfilter = signal.butter(
                order, cutoff, btype="low", analog=False, output="sos"
            )
            filtered = signal.sosfilt(passfilter, detrended)
            retrended = filtered + trend

            res[id_col].extend(_df[id_col].values)
            res[time_col].extend(_df[time_col].values)
            res[val_col].extend(retrended)

    if df_out:
        return pd.DataFrame.from_dict(res, orient="columns")
    else:
        return res
