# parse multi-lead ECG signals
# get wavelet features, CWT/SFT features
# get extractor features
# get ECG specific features like RR interval, QRS complex, etc
# Get features for all bands, then concatenate them
# use neurokit2, tsfresh22 and catch22, scipy.signal

from typing import Literal, Dict, List
from numpy import ndarray

class ECGxtract():
    def __init__(smoothing: bool=True, 
                 trimming: bool=True,
                 aggregation: Literal['concatenate']='concatenate',
                 trimming_kwargs: Dict={}, 
                 smoothing_kwargs: Dict={}):
        pass

    def _smoothing():
        # use defaults from neurokit

        pass

    def _trimming():
        # trim first and last peak

        pass

    def _extract_features_for_single_channel(TimeSerie: ndarray)->ndarray:
        

        pass

    def extract_single_multichannel(TimeSeries: List[ndarray])->List[ndarray]:
        pass

    def extract(TimeSeries: List[ndarray])->List[ndarray]:
        pass