# This modules
# https://arxiv.org/abs/2308.01578
# the beta-VAE model and the training process
# the Time-MAE model

# 1D VAE: https://towardsdatascience.com/vae-for-time-series-1dc0fef4bffa
# 2D VAE

# https://github.com/mqwfrog/ULTS
# TS2Vec model (requires classes): https://github.com/zhihanyue/ts2vec
# TimeCLR https://www.sciencedirect.com/science/article/pii/S0950705122002726
# TimeNet: https://arxiv.org/abs/1706.08838, https://github.com/paudan/TimeNet
# TimeVAE  https://arxiv.org/abs/2111.08095, https://github.com/abudesai/timeVAE
# Wave2Vec https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
# Wave2Vec2 https://huggingface.co/docs/transformers/model_doc/wav2vec2
# TST https://arxiv.org/abs/2010.02803, https://github.com/gzerveas/mvts_transformer
# TS-TCC https://arxiv.org/abs/2208.06616, https://github.com/emadeldeen24/TS-TCC

import torch
from transformers

class ECGxAI:
    '''
        Wrapper around ECGxAI
        Paper: https://www.ahajournals.org/doi/full/10.1161/JAHA.119.015138
        Github: https://github.com/bramiozo/ecgxai
        Pre-trained model:

        pre-trained beta-VAE for 12-lead ECG signals. Good basis for miniECG
    '''
    def __init__(self):
        pass

class TimeMAE:
    '''
        Time-MAE model
        Paper: https://arxiv.org/abs/2111.08095
        Github:
    '''

    def __init__(self):
        pass

class TimeCLR:
    '''
        TimeCLR model
        Paper: https://www.sciencedirect.com/science/article/pii/S0950705122002726
        Github:
    '''
    def __init__(self):
        pass

class TimeNet:
    '''
        TimeNet model
        Paper: https://arxiv.org/abs/1706.08838
        Github:
    '''
    def __init__(self):
        pass

class InceptionTime:
    '''
        InceptionTime model
        Paper: https://arxiv.org/abs/1909.04939
        Github:
    '''
    def __init__(self):
        pass


class Time2Vec:
    '''
        Time2Vec model
        Paper: https://arxiv.org/abs/1907.05321
        Github:
    '''
    def __init__(self):
        pass

class Wav2Vec2:
    '''
        Wav2Vec2 model
        Paper: https://arxiv.org/abs/2006.11477

        ECG-FM: https://arxiv.org/abs/2408.05178
        ECG-FM Github: https://github.com/bowang-lab/ECG-FM
        Github:
    '''
    def __init__(self, model_name='wanglab/ecg-fm-preprint'):
        '''
            model_name: str - model name or path
        '''
