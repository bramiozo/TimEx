# partitional
# hierarchical
# density-based
#
class TSKMeans:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans
    def __init__(self, backend="tslearn", n_clusters=3, **kwargs):
        if backend == "tslearn":
            self.clusterer = tslearn_clustering.TimeSeriesKMeans(
                n_clusters=n_clusters, **kwargs
            )

    def fit(self, ts):
        pass


# distance based clustering
class TSKernelKMeans:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html#tslearn.clustering.KernelKMeans
    def __init__(self):
        pass

    def fit(self, ts):
        pass


# distance based clustering
class TSKShape:
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html#tslearn.clustering.KShape
    def __init__(self):
        pass

    def fit(self, ts):
        pass
