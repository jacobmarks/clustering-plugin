import os

import eta.core.utils as etau

from fiftyone.core.utils import add_sys_path

curr_dir = os.path.dirname(os.path.abspath(__file__))
cluster_dir = os.path.dirname(curr_dir)

with add_sys_path(cluster_dir):
    # pylint: disable=no-name-in-module,import-error
    from clustering import ClusteringConfig, Clustering, ClusteringResults


class BirchClusteringConfig(ClusteringConfig):
    """Configuration for the Birch clustering.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        threshold (0.5): the radius of the subcluster obtained by merging a
            new sample and the closest subcluster should be lesser than the
            threshold. Otherwise a new subcluster is started
        branching_factor (50): the maximum number of CF subclusters in each
            node
        n_clusters (3): the number of clusters after the final clustering step,
            which treats the subclusters from the leaves as new samples
        compute_labels (True): whether to compute labels for each fit
            subcluster and major cluster. If ``False``, the ``labels_`` and
            ``labels_samples_`` attributes are not available
        copy (True): whether to make copies of the subclusters. If ``False``,
            Birch will store references to the subclusters
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        threshold=0.5,
        branching_factor=50,
        n_clusters=3,
        compute_labels=True,
        copy=True,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            **kwargs,
        )

    @property
    def method(self):
        return "birch"


class BirchClusteringResults(ClusteringResults):
    """Birch clustering Results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`BirchClusteringConfig` used
        brain_key: the brain key
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
            applicable
        method (None): a :class:`BirchClustering` instance
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
        return "birch"

    def _compute_clusters(self):
        from sklearn.cluster import Birch

        birch = Birch(
            threshold=self.config.threshold,
            branching_factor=self.config.branching_factor,
            n_clusters=self.config.n_clusters,
            compute_labels=self.config.compute_labels,
            copy=self.config.copy,
        )
        birch.fit(self.embeddings)
        self._clusters = birch.labels_
        self._birch = birch


class BirchClustering(Clustering):
    """Birch clustering factory.

    Args:
        config: an :class:`BirchClusteringConfig`
    """

    def initialize(self, samples, brain_key):
        return BirchClusteringResults(
            samples, self.config, brain_key, method=self
        )
