import eta.core.utils as etau

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import fiftyone.core.brain as fcb
import fiftyone.core.validation as fov
import fiftyone.core.utils as fou

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")


class ClusteringConfig(fcb.BrainMethodConfig):
    """Clustering configuration.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        cluster_field=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.cluster_field = cluster_field
        super().__init__(**kwargs)

    @property
    def type(self):
        return "clustering"

    @property
    def method(self):
        """The name of the clustering method."""
        raise NotImplementedError("subclass must implement method")


class Clustering(fcb.BrainMethod):
    """Base class for clustering factories.

    Args:
        config: a :class:`ClusteringConfig`
    """

    def initialize(self, samples, brain_key):
        """Initializes a clustering run.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleColllection`
            brain_key: the brain key

        Returns:
            a :class:`ClusteringResults`
        """
        raise NotImplementedError("subclass must implement initialize()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields


class ClusteringResults(fcb.BrainResults):
    """Base class for clustering results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`ClusteringConfig` used
        brain_key: the brain key
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
        method (None): a :class:`Clustering` method
    """

    def __init__(
        self,
        samples,
        config,
        brain_key,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
        method=None,
    ):
        super().__init__(
            samples,
            config,
            brain_key,
        )

        embeddings, sample_ids, label_ids = self._parse_data(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )

        has_sample_ids = sample_ids is not None and len(sample_ids) > 0
        has_label_ids = label_ids is not None and len(label_ids) > 0
        has_patches = config.patches_field is not None

        if not has_sample_ids and not has_patches:
            sample_ids = samples.values("id")

        if not has_label_ids and has_patches:
            label_ids = samples.values(
                f"{config.patches_field}.detections.id", unwind=True
            )

        self._brain_key = brain_key
        self._embeddings = embeddings
        self._sample_ids = sample_ids
        self._label_ids = label_ids
        self._model = None
        self._method = method
        self._clusters = None

    @property
    def config(self):
        """The :class:`ClusteringConfig` for these results."""
        return self._config

    @staticmethod
    def _parse_data(
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
    ):
        if embeddings is None:
            embeddings, sample_ids, label_ids = fbu.get_embeddings(
                samples._dataset,
                patches_field=config.patches_field,
                embeddings_field=config.embeddings_field,
            )
        elif sample_ids is None:
            sample_ids, label_ids = fbu.get_ids(
                samples,
                patches_field=config.patches_field,
                data=embeddings,
                data_type="embeddings",
            )

        return embeddings, sample_ids, label_ids

    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is None:
            attrs.extend(["embeddings"])

        return attrs

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def sample_ids(self):
        return self._sample_ids

    @property
    def label_ids(self):
        return self._label_ids

    def get_model(self):
        """Returns the stored model for this run.

        Returns:
            a :class:`fiftyone.core.models.Model`
        """
        if self._model is None:
            model = self.config.model
            if model is None:
                raise ValueError("These results don't have a stored model")

            if etau.is_str(model):
                model = foz.load_zoo_model(model)

            self._model = model

        return self._model

    def _assign_sample_cluster_labels(self):
        samples = self.samples

        if self.config.cluster_field is None:
            self.config.cluster_field = self._brain_key + "_cluster"

        label_strs = [str(c) for c in self._clusters]

        samples.set_values(
            self.config.cluster_field,
            label_strs,
        )
        samples.save()

    def _assign_patch_cluster_labels(self):
        pass

    # def _assign_patch_cluster_labels(self):
    #     samples = self.samples
    #     patches_field = self.config.patches_field
    #     samples.set_values(
    #         patches_field + ".detections." + self.config.cluster_field,
    #         self._clusters,
    #         label_ids=self._label_ids,
    #     )
    #     samples.save()

    def _assign_cluster_labels(self):
        if self._clusters is None:
            raise ValueError("Clusters have not been computed")

        if self.config.patches_field is None:
            self._assign_sample_cluster_labels()
        else:
            self._assign_patch_cluster_labels()

    def get_clusters(self):
        return self._clusters

    def get_cluster_field(self):
        return self.config.cluster_field

    def get_cluster(self, cluster_id):
        return self._clusters[cluster_id]

    def get_cluster_ids(self):
        return list(self._clusters.keys())

    def _compute_clusters(self):
        raise NotImplementedError(
            "subclass must implement _compute_clusters()"
        )

    def compute_clusters(self):
        print("Computing clusters")
        self._compute_clusters()
        print("Clusters computed")
        self._assign_cluster_labels()

    def compute_centroids(self):
        raise NotImplementedError(
            "subclass must implement compute_centroids()"
        )

    def compute_distance_to_centroids(self):
        raise NotImplementedError(
            "subclass must implement compute_distance_to_centroids()"
        )
