
from typing import Literal, Dict, List, Annotated, Optional, Sequence
from numpy import ndarray, concatenate
import numpy as np
NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class TsTokenizer():
    '''
        Pre-defined canonical shapes -> transform requires scanning the timeseries and using some similarity metric
    '''

    def __init__(self, pattern_dict: Dict[int, List[float]], max_len: int|None, num_channels: int|None, stride: int=20, truncation: bool=True, truncation_side: str='right', padding: bool=True, padding_side: str='right', padding_token: int=-100):
        """
            pattern_dict: dictionary with match patterns
            stride: stride by which we move through the timeseries to match the pattern
            max_len: maximum number of tokens per channel
            num_channels: how many channels are in the timeseries
            truncation: all series are truncated to the max_len
            truncation_side: 'left', 'right', or 'center'
            padding: padding series until max_len
            padding_side: 'left', 'right', or 'center'
        """

        # add asserts
        self.pattern_dict = pattern_dict
        self.stride = stride
        self.num_channels = num_channels
        self.max_len = max_len
        self.num_channels = num_channels
        self.truncation = truncation
        self.truncation_side = truncation_side
        self.padding = padding
        self.padding_side = padding_side
        self.padding_token = padding_token
        self.similarity_type = 'cosine'

    def get_tokens(self, ts):
        """
            ts: 1D array
        """


        pass

    def transform(self, X: List[NDArray2D])->List[NDArray2D]:
        """
            Given a timeseries: x<-(ns, sz, d) we generate a tokenized timeseries xt<-(ns, st, d)
        """
        # do asserts ..

        tokenized_set = []
        for ts in X:
            Tarr = np.zeros(self.max_len, self.num_channels, dtype=np.int8)
            for channel in range(ts.shape[1]):
                toks = get_tokens(ts[:, channel])
                Tarr[:, channel] = toks
            tokenized_set.append(Tarr)
        return tokenized_set
