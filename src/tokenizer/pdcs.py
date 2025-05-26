from typing import Literal, Dict, List, Annotated, Optional, Sequence, Tuple
from numpy import ndarray, concatenate
import numpy as np
from numpy.linalg import norm
import numba
from transformers import PreTrainedTokenizer
import fastdtw
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
from sklearn.cluster import BisectingKMeans, KMeans


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
        pass

    def write_canonicals_to_json(self):
        """
            Given a dictionary Dict[int, np.ndarray], write a json to disk
        """
        pass



class TsTokenizer(PreTrainedTokenizer):
    '''
        Pre-Defined Canonical Shapes -> transform requires scanning the timeseries and using some similarity metric
    '''

    def __init__(self, pattern_dict: Dict[int, np.ndarray], max_len: int|None, num_channels: int|None, stride: int=20, window: int=50, truncation: bool=True, truncation_side: str='right', padding: bool=True, padding_side: str='right', padding_token: int=-100):
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
        max_len = len(ts)
        # create list of range tuples
        prev = 0
        # Calculate the potential number of tokens from the input timeseries
        num_possible_tokens = len(ts) // self.stride
        # Determine the target length for the output tokens array
        # If self.max_len is None, use the number of possible tokens from the current timeseries
        target_len = self.max_len if self.max_len is not None else num_possible_tokens
        # Initialize the tokens array with padding tokens.
        # The shape is determined by the target_len, which is guaranteed to be an integer.
        tokens = np.full((target_len,), self.padding_token, dtype=np.int16)
        for i, s in enumerate(range(max_len//self.stride)):
            move = s*self.stride
            segment = ts[prev + move : prev+self.window + move]
            # check similarity with pattern_dict, select closest match id as token
            token = closest_match(segment)
            tokens[i] = token
        return tokens

    def _transform(self, X: List[NDArray2D])->List[NDArray2D]:
        """
            Given a timeseries: x<-(ns, sz, d) we generate a tokenized timeseries xt<-(ns, st, d)
        """
        # do asserts ..

        tokenized_set = []
        for ts in X:
            #TODO Need to deal with self.max_len is None !
            Tarr = np.zeros(self.max_len, self.num_channels, dtype=np.int8)
            for channel in range(ts.shape[1]):
                toks = self.get_tokens(ts[:, channel])
                # truncation and padding
                #
                Tarr[:, channel] = toks
            tokenized_set.append(Tarr)
        return tokenized_set
