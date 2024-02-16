import os

import eta.core.utils as etau

from fiftyone.core.utils import add_sys_path

curr_dir = os.path.dirname(os.path.abspath(__file__))
cluster_dir = os.path.dirname(curr_dir)

with add_sys_path(cluster_dir):
    # pylint: disable=no-name-in-module,import-error
    from clustering import ClusteringConfig, Clustering, ClusteringResults


class KMeansClusteringConfig(ClusteringConfig):
    """Configuration for the K-means clustering.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        n_clusters (8): the number of clusters to create
        init (``'k-means++'``): the method for initializing the cluster
            centroids. Supported values are ``('k-means++', 'random')``
        n_init (10): the number of times to run the k-means algorithm with
            different centroid seeds. The final results will be the best output
            of ``n_init`` consecutive runs in terms of inertia
        max_iter (300): the maximum number of iterations to perform
        tol (0.0001): the relative tolerance with regards to inertia to
            declare convergence
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=None,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        super().__init__(**kwargs)

    @property
    def method(self):
        return "kmeans"


class KMeansClusteringResults(ClusteringResults):
    """K-Means clustering Results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`KMeansClusteringConfig` used
        brain_key: the brain key
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
            applicable
        method (None): a :class:`KMeansClustering` instance
    """

    def __init__(
        self, samples, config, brain_key, embeddings=None, method=None
    ):
        ClusteringResults.__init__(
            self,
            samples,
            config,
            brain_key,
            embeddings=embeddings,
            method=method,
        )

    @property
    def method(self):
        return "kmeans"

    def _compute_clusters(self):
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.init,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
        )
        kmeans.fit(self.embeddings)
        self._clusters = kmeans.labels_
        self._kmeans = kmeans


class KMeansClustering(Clustering):
    """K-Means clustering factory.

    Args:
        config: an :class:`KMeansClusteringConfig`
    """

    def initialize(self, samples, brain_key):
        return KMeansClusteringResults(
            samples, self.config, brain_key, method=self
        )
