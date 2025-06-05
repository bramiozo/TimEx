from typing import Literal, Dict, List, Annotated, Optional, Sequence, Tuple
from numpy import ndarray, concatenate
import numpy as np
from numpy.linalg import norm
import numba
from transformers import PreTrainedTokenizer
from transformers.PreTrainedTokenizer import added_token as AddedToken
import fastdtw
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
from sklearn.cluster import BisectingKMeans, KMeans

import argparse
import json

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class CanonicalPatterns():
    """
    extract CanonicalPatterns
    0. select non-overlapping segments of varying size
    1. DTW distance followed by HDBSCAN
    2. Continue with the timeseries that were not clustered and apply KMeans.

    3 approaches for scale:
        - batch wise processing
        - sparse affinity matrix
        - combination

    num_of_patterns: effectively how many clusters are we looking for
    segment_range: how long are the ts segments we are clustering
    sample_rounds: how many time do we go over the timeseries to extract samples
    distance_metric: which distance metric is used for the DTW distance
    sparse_affinity: to enable clustering of large amounts of timeseries
    sparsity_distance_cutoff: percentile distance above which we ignore it
    burnin: how many rounds do we use to gather statistics
    """


    def __init__(self, num_of_patterns: int = 100, segment_range: Tuple[int,int] =(50,10), sample_rounds: int=5, bisecting_kmeans: bool=False, sparse_affinity: bool=True,
    sparsity_distance_cutoff: float=0.95, burnin: int=1000,
    distance_metric: Literal['euclidean', 'manhattan', 'cosine']='euclidean'):

        self.num_of_patterns = num_of_patterns
        self.segment_range = segment_range
        self.sample_rounds = sample_rounds
        self.bisecting_kmeans = bisecting_kmeans

        if distance_metric=='euclidean':
            self.dist_metric = d.euclidean
        elif distance_metric=='manhattan':
            self.dist_metric = d.cityblock
        elif distance_metric=='cosine':
            self.dist_metric = d.cosine

    def _distance_extraction(self, X,Y):
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        if len(Y.shape)==1:
            Y = Y.reshape(-1,1)
        dist = fastdtw.fastdtw(X, Y, dist=d.euclidean)
        return dist

    def get_distance_matrix(self):
        # if self.sparse_affinity is True, then use sparsity_distance_cutoff
        pass

    def fit(self, TS: List[NDArray2D]):

        return self

    def get_clusters_hdbscan(self):
        """
            HDBSCAN: allows sparsity?
        """
        pass

    def get_clusters_idbscan(self):
        """
            Incremental dbscan
        """
        pass

    def get_clusters_bkmeans(self):
        """
            Mini-batch bisecting k-means
        """
        pass

    def get_clusters_dpmeans(self):
        """https://github.com/BGU-CS-VIL/pdc-dp-means"""
        pass

    def get_clusters_SAP(self):
        """
            Sparse Affinity Propagation
        """
        pass

    def get_canonicals(self)-> Dict[int, np.ndarray]:
        """
            Given the cluster assignments we would like to extract one statistical
            average of the timeseries. We do this with LOWESS
        """
        self.canonicals = {1:np.array([1,2,3,4,5])}
        return self.canonicals

    def write_canonicals_to_json(self, Output: str="canonical_pdcs.json"):
        """
            Given a dictionary Dict[int, np.ndarray], write a json to disk
        """

        fw = open(Output, 'w', encoding='utf-8')
        json.dump(self.canonicals, fw)


class TsTokenizer(PreTrainedTokenizer):
    '''
        Pre-Defined Canonical Shapes -> transform requires scanning the timeseries and using some similarity metric
    '''

    def __init__(self, pattern_dict: Dict[int, np.ndarray], max_len: int|None, num_channels: int|None, stride: int=20, window: int=50, truncation: bool=True, truncation_side: str='right', padding: bool=True, padding_side: str='right', padding_token_id: int=-100,
    ):
        """
            pattern_dict: dictionary with match patterns
            stride: stride by which we move through the timeseries to match the pattern
            window: the window size that we move through the timeseries to match the pattern
            max_len: maximum number of tokens per channel
            num_channels: how many channels are in the timeseries
            truncation: all series are truncated to the max_len
            truncation_side: 'left', 'right', or 'center'
            padding: padding series until max_len
            padding_side: 'left', 'right', or 'center'
        """

        # initialize base tokenizer (no vocab files)
        super().__init__(pad_token=str(padding_token),
            truncation_side=truncation_side, padding_side=padding_side)
        # assert required parameters
        if max_len is None or num_channels is None:
            raise ValueError("Both max_len and num_channels must be provided")
        # add asserts
        self.pattern_dict = pattern_dict
        self.stride = stride
        self.window = window
        self.num_channels = num_channels
        self.max_len = max_len
        self.num_channels = num_channels
        self.truncation = truncation
        self.truncation_side = truncation_side
        self.padding = padding
        self.padding_side = padding_side
        self.padding_token = padding_token
        # token id space: include all pattern ids and padding
        self.token_ids = set(pattern_dict.keys()) | {padding_token}
        # set attributes
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        self.cls_token_id = 4
        self.sep_token_id = 5
        self.similarity_type = 'cosine'

    @staticmethod
    def distance(s1: np.ndarray, s2: np.ndarray)-> float:
        return fastdtw.fastdtw(s1, s2, dist=None)

    @staticmethod
    def sim(s1: np.ndarray, s2: np.ndarray)-> float:
        # cosine similarity
        #
        s1L=len(s1)
        s2L=len(s2)
        L = min(s1L, s2L)
        D = max(s1L, s2L)-L

        res = []
        for k in range(D):
            _s1 = s1 if s1L==L else s1+k
            _s2 = s2 if s2L==L else s2+k
            res.append((_s1 @ _s2.T) / (norm(_s1)*norm(_s2)))
        return sum(res)/D # COULD BE MAX, AVERAGE, MIN?

    def closest_match(self, ts: np.ndarray)-> int:
        max_similarity = -1000
        max_idx = -1
        for k,v in self.pattern_dict.items():
            max_idx = k if self.sim(ts, v)> max_similarity else max_idx
        return max_idx

    def get_tokens(self, ts: np.ndarray)->np.ndarray:
        """
            ts: 1D array
        """
        # Calculate the potential number of tokens from the input timeseries
        num_possible_tokens = len(ts) // self.stride
        # Determine the target length for the output tokens array
        # If self.max_len is None, use the number of possible tokens from the current timeseries
        target_len = self.max_len if self.max_len is not None else num_possible_tokens
        # Initialize the tokens array with padding tokens.
        # The shape is determined by the target_len, which is guaranteed to be an integer.
        tokens = np.full((target_len,), self.padding_token, dtype=np.int16)
        for i in range(num_possible_tokens):
            move = i * self.stride
            segment = ts[move : move + self.window]
            # check similarity with pattern_dict, select closest match id as token
            token = self.closest_match(segment)
            tokens[i] = token
        return tokens

    def _transform(self, X: List[NDArray2D]) -> List[np.ndarray]:
        """
            Given a timeseries: x<-(ns, sz, d) we generate a tokenized timeseries xt<-(ns, st, d)
        """
        tokenized_set: List[np.ndarray] = []
        for ts in X:
            # generate token sequence per channel
            target_len = self.max_len if self.max_len is not None else (ts.shape[0] // self.stride)
            Tarr = np.full((target_len, int(self.num_channels)), self.padding_token, dtype=np.int16)
            for channel in range(ts.shape[1]):
                toks = self.get_tokens(ts[:, channel])
                length = min(len(toks), target_len)
                Tarr[:length, channel] = toks[:length]
            tokenized_set.append(Tarr)
        return tokenized_set

    # Transformers compatibility methods
    @property
    def model_input_names(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    def _tokenize(self, series: np.ndarray) -> List[int]:
        # single-channel or multi-channel series
        # flatten to 2D array (n_samples, n_channels)
        if series.ndim == 1:
            arr = series.reshape(-1, 1)
        else:
            arr = series
        tokens = self._transform([arr])[0]
        return tokens.flatten().tolist()

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def encode_plus(
        self,
        series: np.ndarray,
        return_tensors: Optional[str] = None,
        **kwargs) -> Dict[str, List[int]]:
        input_ids = self._tokenize(series)
        attention_mask = [1 if tok != self.padding_token else 0 for tok in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(
        self,
        token_ids: List[int],
        **kwargs,
    ) -> str:
        return ' '.join(self._convert_id_to_token(i) for i in token_ids)

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as mapping from token strings to ids."""
        return {str(i): i for i in sorted(self.token_ids)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_ids)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        import os
        # ensure directory exists
        os.makedirs(save_directory, exist_ok=True)
        file_name = (filename_prefix or "vocab") + ".json"
        path = os.path.join(save_directory, file_name)
        # write token-to-id mapping
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab(), f)
        return (path,)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_series', type=str, default=None)
    argparser.add_argument('--input_canonical', type=str, default=None)
    argparser.add_argument('--save_folder', type=str)
    argparser.add_argument('--test', action='store_true', default=False)

    args = argparser.parse_args()

    if args.test:
    # load test series to test CanonicalPatterns
    #

    # load test patterns to test tokenizers
    #

    # inverse transform the tokenizer to check output validity
    # Make line plot with test timeseries (noisy sinus)
