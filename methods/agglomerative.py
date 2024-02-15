import os

import eta.core.utils as etau

from fiftyone.core.utils import add_sys_path

curr_dir = os.path.dirname(os.path.abspath(__file__))
cluster_dir = os.path.dirname(curr_dir)

with add_sys_path(cluster_dir):
    # pylint: disable=no-name-in-module,import-error
    from clustering import ClusteringConfig, Clustering, ClusteringResults


class AgglomerativeClusteringConfig(ClusteringConfig):
    """Configuration for the Agglomerative clustering.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        n_clusters (2): the number of clusters to find
        affinity (``'euclidean'``): the metric used to compute the linkage.
            Can be ``'euclidean'``, ``'l1'``, ``'l2'``, ``'manhattan'``,
            ``'cosine'``, or ``'precomputed'``
        linkage (``'ward'``): the linkage criterion to use. Can be ``'ward'``,
            ``'complete'``, ``'average'``, or ``'single'``
        distance_threshold (None): the linkage distance threshold above which,
            clusters will not be merged. If not ``None``, ``n_clusters`` must
            be ``None`` and ``compute_full_tree`` must be ``True``
        compute_full_tree (False): whether to compute the full tree of
            subclusters during the fit. If ``True``, ``distance_threshold``
            must be ``None`` and ``n_clusters`` must be ``None``
        affinity_params (None): optional keyword arguments for the affinity
            matrix computation
        memory (None): the memory used to cache the distance matrices. If a
            string is given, it is the path to the caching directory
        connectivity (None): the connectivity matrix used for hierarchical
            clustering. If ``None``, the hierarchical clustering algorithm is
            unstructured
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.n_clusters = n_clusters
        self.metric = metric
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            **kwargs,
        )

    @property
    def method(self):
        return "agglomerative"


class AgglomerativeClusteringResults(ClusteringResults):
    """Agglomerative clustering Results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`AgglomerativeClusteringConfig` used
        brain_key: the brain key
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
            applicable
        method (None): a :class:`AgglomerativeClustering` instance
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
        return "agglomerative"

    def _compute_clusters(self):
        from sklearn.cluster import AgglomerativeClustering

        agglomerative = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            metric=self.config.metric,
            memory=self.config.memory,
            connectivity=self.config.connectivity,
            compute_full_tree=self.config.compute_full_tree,
            linkage=self.config.linkage,
            distance_threshold=self.config.distance_threshold,
            compute_distances=self.config.compute_distances,
        )
        agglomerative.fit(self.embeddings)
        self._clusters = agglomerative.labels_
        self._agglomerative = agglomerative


class AgglomerativeClustering(Clustering):
    """Agglomerative clustering factory.

    Args:
        config: an :class:`AgglomerativeClusteringConfig`
    """

    def initialize(self, samples, brain_key):
        return AgglomerativeClusteringResults(
            samples, self.config, brain_key, method=self
        )
