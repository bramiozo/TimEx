# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional
from numpy import ndarray, concatenate
import numpy as np
import neurokit2 as nk
from catch22 import catch22_all
from tsfresh.feature_extraction import extract_features
import pandas as pd

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class ECGxtract():
    def __init__(self,
                 sampling_rate: int = 1000,
                 smoothing: bool = True,
                 trimming: bool = True,
                 extractor: Literal['catch22', 'tsfresh', 'both'] = 'catch22',
                 aggregation: Literal['concatenate'] = 'concatenate',
                 trimming_kwargs: Dict = {},
                 smoothing_kwargs: Dict = {}):
        self.sampling_rate = sampling_rate
        self.smoothing = smoothing
        self.trimming = trimming
        self.extractor = extractor
        self.aggregation = aggregation
        self.trimming_kwargs = trimming_kwargs
        self.smoothing_kwargs = smoothing_kwargs

    def _smoothing(self, TimeSerie: ndarray) -> ndarray:
        # Default Neurokit smoothing (low-pass filtering)
        return nk.signal_filter(TimeSerie, sampling_rate=self.sampling_rate, **self.smoothing_kwargs)

    def _trimming(self, TimeSerie: ndarray) -> ndarray:
        # Trim ECG between the first and last R-peaks
        peaks, _ = nk.ecg_peaks(TimeSerie, sampling_rate=self.sampling_rate)
        peak_indices = peaks['ECG_R_Peaks']
        return TimeSerie[peak_indices[0]:peak_indices[-1]]

    def _wavelet_features(self, signal: ndarray):
        # Continuous Wavelet Transform (CWT) based features
        coefs, freqs = nk.signal_cwt(signal, sampling_rate=self.sampling_rate)
        return coefs.mean(axis=1)  # using mean power across frequencies

    def _extractor_features(self, signal: ndarray):
        features = []
        if self.extractor in ['catch22', 'both']:
            c22_features = catch22_all(signal)['values']
            features.extend(c22_features)

        if self.extractor in ['tsfresh', 'both']:
            tsfresh_df = pd.DataFrame({'signal': signal})
            tsfresh_features = extract_features(tsfresh_df, column_id=None, disable_progressbar=True).values.flatten()
            features.extend(tsfresh_features)

        return np.array(features)

    def _ecg_specific_features(self, signal: ndarray):
        # Extract ECG-specific features: RR intervals, QRS features
        try:
            signals, info = nk.ecg_process(signal, sampling_rate=self.sampling_rate)
            hr_features = nk.ecg_intervalrelated(info)
            return hr_features.values.flatten()
        except Exception:
            return np.array([])

    def _extract_features_for_single_channel(self, TimeSerie: ndarray) -> ndarray:
        if self.smoothing:
            TimeSerie = self._smoothing(TimeSerie)

        if self.trimming:
            TimeSerie = self._trimming(TimeSerie)

        wavelet_feats = self._wavelet_features(TimeSerie)
        extractor_feats = self._extractor_features(TimeSerie)
        ecg_feats = self._ecg_specific_features(TimeSerie)

        # Concatenate all features
        return concatenate([wavelet_feats, extractor_feats, ecg_feats])

    def _extract_single_multichannel(self, TimeSeries: NDArray2D) -> ndarray:
        assert TimeSeries.ndim == 2

        features = [self._extract_features_for_single_channel(TimeSeries[:, ch]) for ch in range(TimeSeries.shape[1])]

        # Concatenate all channel features
        return concatenate(features)

    def extract(self, TimeSeries: List[ndarray]) -> List[ndarray]:
        extracted_features = []
        for ts in TimeSeries:
            if ts.ndim == 1:
                feats = self._extract_features_for_single_channel(ts)
            elif ts.ndim == 2:
                feats = self._extract_single_multichannel(ts)
            else:
                raise ValueError(f"Unsupported ndarray dimension: {ts.ndim}")

            extracted_features.append(feats)

        return extracted_features
