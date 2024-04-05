import os

import eta.core.utils as etau

from fiftyone.core.utils import add_sys_path

curr_dir = os.path.dirname(os.path.abspath(__file__))
cluster_dir = os.path.dirname(curr_dir)

with add_sys_path(cluster_dir):
    # pylint: disable=no-name-in-module,import-error
    from clustering import ClusteringConfig, Clustering, ClusteringResults


class HDBSCANClusteringConfig(ClusteringConfig):
    """Configuration for the HDBSCAN clustering.

    Args:
        embeddings_field (None): The sample field containing the embeddings,
            if one was provided
        model (None): The :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        min_cluster_size (5): The minimum number of samples in a group for that
            group to be considered a cluster; groupings smaller than this size
            will be left as noise.
        min_samples (None): The number of samples in a neighborhood for a point
            to be considered a core point. This includes the point itself. When
            ``None``, defaults to ``min_cluster_size``.
        cluster_selection_epsilon (0.0): A distance threshold. Clusters below
            this value will be merged.
        max_cluster_size (None): The maximum number of samples in a cluster.
            There is no limit when max_cluster_size=None. Has no effect if
            ``cluster_selection_method="leaf"``.
        metric (``'euclidean'``): The metric used to compute the linkage. Can
            be ``'euclidean'``, ``'l1'``, ``'l2'``, or ``'manhattan'``.
        alpha (1.0): A distance scaling parameter as used in robust single
            linkage.
        algorithm (`"auto"`): The algorithm to use. Can be ``"brute"``,
            ``"kd_tree"``, or ``"ball_tree"``.
        leaf_size (40): Leaf size for trees responsible for fast nearest
            neighbor queries when a KDTree or a BallTree are used as
            core-distance algorithms. A large dataset size and small leaf_size
            may induce excessive memory usage. If you are running out of memory
            consider increasing the leaf_size parameter. Ignored for
            ``algorithm="brute"``.
        cluster_selection_method (`"eom"`): The method used to select clusters
            from the condensed tree. The standard approach for HDBSCAN* is to
            use an Excess of Mass ("eom") algorithm to find the most persistent
            clusters. Alternatively you can instead select the clusters at the
            leaves of the tree – this provides the most fine grained and
            homogeneous clusters by setting this to ``"leaf"``.
        allow_single_cluster (False): Whether or not to allow single cluster.
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        alpha=1.0,
        algorithm="auto",
        leaf_size=40,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        **kwargs,
    ):

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

        super().__init__(**kwargs)

    @property
    def method(self):
        return "hdbscan"


class HDBSCANClusteringResults(ClusteringResults):
    """HDBSCAN clustering Results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`HDBSCANClusteringConfig` used
        run_key: the run key
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
            applicable
        method (None): a :class:`HDBSCANClustering` instance
    """

    def __init__(self, samples, config, run_key, embeddings=None, method=None):
        ClusteringResults.__init__(
            self,
            samples,
            config,
            run_key,
            embeddings=embeddings,
            method=method,
        )

    @property
    def method(self):
        return "hdbscan"

    def _compute_clusters(self):
        from sklearn.cluster import HDBSCAN

        hdbscan = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            max_cluster_size=self.config.max_cluster_size,
            metric=self.config.metric,
            alpha=self.config.alpha,
            algorithm=self.config.algorithm,
            leaf_size=self.config.leaf_size,
            cluster_selection_method=self.config.cluster_selection_method,
            allow_single_cluster=self.config.allow_single_cluster,
        )
        hdbscan.fit(self.embeddings)
        self._clusters = hdbscan.labels_
        self._hdbscan = hdbscan


class HDBSCANClustering(Clustering):
    """HDBSCAN clustering factory.

    Args:
        config: an :class:`HDBSCANClusteringConfig`
    """

    def initialize(self, samples, run_key):
        return HDBSCANClusteringResults(
            samples, self.config, run_key, method=self
        )
