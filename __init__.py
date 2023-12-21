"""Unsupervised clustering plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os

import eta.core.utils as etau

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.zoo.models as fozm
from fiftyone import ViewField as F

import fiftyone.core.brain as fcb
import fiftyone.core.validation as fov
import fiftyone.core.utils as fou
from fiftyone.core.utils import add_sys_path

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")


_DEFAULT_MODEL = "clip-vit-base32-torch"
_DEFAULT_BATCH_SIZE = None
_DEFAULT_METHOD = "kmeans"

curr_dir = os.path.dirname(os.path.abspath(__file__))
method_dirs = os.path.join(curr_dir, "methods")

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    # pylint: disable=no-name-in-module,import-error
    from methods.kmeans import KMeansClusteringConfig

    # pylint: disable=no-name-in-module,import-error
    from methods.birch import BirchClusteringConfig

    # pylint: disable=no-name-in-module,import-error
    from methods.agglomerative import AgglomerativeClusteringConfig


def _parse_config(method, **kwargs):
    if method is None:
        method = _DEFAULT_METHOD

    if method == "kmeans":
        config_cls = KMeansClusteringConfig
    elif method == "birch":
        config_cls = BirchClusteringConfig
    elif method == "agglomerative":
        config_cls = AgglomerativeClusteringConfig
    else:
        raise ValueError("Unsupported method '%s'" % method)

    return config_cls(**kwargs)


def _save_embeddings_to_dataset(
    samples,
    embeddings,
    embeddings_field,
    patches_field=None,
):
    if patches_field is None:
        samples.set_values(embeddings_field, embeddings)
    else:
        samples.set_values(
            patches_field + ".detections." + embeddings_field,
            embeddings,
        )


def compute_clusters(
    samples,
    patches_field=None,
    embeddings=None,
    embeddings_field=None,
    brain_key=None,
    model=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    skip_failures=True,
    method=None,
    **kwargs,
):
    fov.validate_collection(samples)

    if embeddings and etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None

    if embeddings_field is not None and embeddings is False:
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings_field,
            patches_field=patches_field,
        )
    elif embeddings is not None:
        embeddings_exist = True
    else:
        embeddings_exist = False

    if model is None and embeddings is None and not embeddings_exist:
        model = _DEFAULT_MODEL
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    if etau.is_str(model):
        _model = foz.load_zoo_model(model)
    else:
        _model = model

    config = _parse_config(
        method,
        embeddings_field=embeddings_field,
        patches_field=patches_field,
        model=model,
        force_square=force_square,
        alpha=alpha,
        skip_failures=skip_failures,
        **kwargs,
    )
    clustering = config.build()
    clustering.ensure_requirements()

    if brain_key is not None:
        clustering.register_run(samples, brain_key, overwrite=False)

    results = clustering.initialize(samples, brain_key)

    get_embeddings = embeddings is not False
    if get_embeddings:
        embeddings, _, _ = fbu.get_embeddings(
            samples,
            model=_model,
            patches_field=patches_field,
            embeddings=embeddings,
            embeddings_field=embeddings_field,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            skip_failures=skip_failures,
        )
    else:
        embeddings = None

    results._embeddings = embeddings.tolist()

    if not embeddings_exist and embeddings_field is not None:
        _save_embeddings_to_dataset(
            samples,
            embeddings,
            embeddings_field,
            patches_field=patches_field,
        )

    results.compute_clusters()

    clustering.save_run_results(samples, brain_key, results)

    return results


#### The Operator ####


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def _list_target_views(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = "DATASET"
    if has_view or has_selected:
        target_choices = types.RadioGroup()
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Run clustering on the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Run clustering on the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Run clustering on the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
        )
    else:
        ctx.params["target"] = "DATASET"


def get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


AVAILABLE_METHODS = (
    "kmeans",
    "birch",
    "agglomerative",
)


def get_embeddings(ctx, inputs, view, patches_field):
    if patches_field is not None:
        root, _ = view._get_label_field_root(patches_field)
        field = view.get_field(root, leaf=True)
        schema = field.get_field_schema(ftype=fo.VectorField)
        embeddings_fields = set(root + "." + k for k in schema.keys())
    else:
        schema = view.get_field_schema(ftype=fo.VectorField)
        embeddings_fields = set(schema.keys())

    embeddings_choices = types.AutocompleteView()
    for field_name in sorted(embeddings_fields):
        embeddings_choices.add_choice(field_name, label=field_name)

    inputs.str(
        "embeddings",
        label="Embeddings",
        description=(
            "An optional sample field containing pre-computed embeddings to "
            "use. Or when a model is provided, a new field in which to store "
            "the embeddings"
        ),
        view=embeddings_choices,
    )

    embeddings = ctx.params.get("embeddings", None)

    if embeddings not in embeddings_fields:
        model_choices = types.DropdownView()
        for name in sorted(_get_zoo_models()):
            model_choices.add_choice(name, label=name)

        inputs.str(
            "model",
            label="Model",
            description=(
                "An optional name of a model from the FiftyOne Model Zoo to "
                "use to generate embeddings"
            ),
            view=model_choices,
        )

        model = ctx.params.get("model", None)

        if model is not None:
            inputs.int(
                "batch_size",
                label="Batch size",
                description=(
                    "A batch size to use when computing embeddings. Some "
                    "models may not support batching"
                ),
            )

            inputs.int(
                "num_workers",
                label="Num workers",
                description=(
                    "The number of workers to use for Torch data loaders"
                ),
            )

            inputs.bool(
                "skip_failures",
                default=True,
                label="Skip failures",
                description=(
                    "Whether to gracefully continue without raising an error "
                    "if embeddings cannot be generated for a sample"
                ),
            )


def _get_label_fields(sample_collection, label_types):
    schema = sample_collection.get_field_schema(embedded_doc_type=label_types)
    return list(schema.keys())


def _get_zoo_models():
    available_models = set()
    for model in fozm._load_zoo_models_manifest():
        if model.has_tag("embeddings"):
            available_models.add(model.name)

    return available_models


def get_new_brain_key(
    ctx,
    inputs,
    name="brain_key",
    label="Brain key",
    description="Provide a brain key for this run",
):
    prop = inputs.str(
        name,
        required=True,
        label=label,
        description=description,
    )

    brain_key = ctx.params.get(name, None)
    if brain_key is not None and brain_key in ctx.dataset.list_brain_runs():
        prop.invalid = True
        prop.error_message = "Brain key already exists"
        brain_key = None

    return brain_key


def brain_init(ctx, inputs):
    target_view = get_target_view(ctx, inputs)

    brain_key = get_new_brain_key(ctx, inputs)
    if brain_key is None and brain_key != "None":
        return False

    patches_fields = _get_label_fields(
        target_view,
        (fo.Detection, fo.Detections, fo.Polyline, fo.Polylines),
    )

    if patches_fields:
        patches_field_choices = types.DropdownView()
        for field_name in sorted(patches_fields):
            patches_field_choices.add_choice(field_name, label=field_name)

        inputs.str(
            "patches_field",
            label="Patches field",
            description=(
                "An optional sample field defining the image patches in each "
                "sample that have been/will be embedded. If omitted, the "
                "full images are processed"
            ),
            view=patches_field_choices,
        )

    patches_field = ctx.params.get("patches_field", None)

    get_embeddings(ctx, inputs, target_view, patches_field)

    return True


def _handle_basic_inputs(ctx, inputs):
    method_choices = types.DropdownView()
    for method in AVAILABLE_METHODS:
        method_choices.add_choice(
            method,
            label=method,
        )

    brain_key = ctx.params.get("brain_key", None)

    if brain_key:
        inputs.enum(
            "method",
            method_choices.values(),
            default=method_choices.choices[0].value,
            label="Clustering method",
            description="The clustering method to use",
            view=method_choices,
        )

        inputs.str(
            "cluster_field",
            label="Cluster field",
            description=(
                "The name of the sample field in which to store the cluster "
                "labels. If omitted, a default field name will be used"
            ),
        )


def _handle_kmeans_inputs(ctx, inputs):
    inputs.view(
        "kmeans_header",
        types.Header(
            label="KMeans Config",
            description="Specify the hyperparameters for the k-means clustering algorithm",
            divider=True,
        ),
    )

    inputs.int(
        "n_clusters",
        label="Number of clusters",
        description="The number of clusters to create",
        default=8,
        required=True,
    )

    init_choices = ("k-means++", "random")
    init_group = types.RadioGroup()

    for choice in init_choices:
        init_group.add_choice(choice, label=choice)

    inputs.enum(
        "init",
        init_group.values(),
        label="Initialization method",
        description="The method for initializing the cluster centroids",
        view=types.RadioGroup(),
        default=init_choices[0],
        required=False,
    )

    inputs.int(
        "n_init",
        label="Number of initializations",
        description=(
            "The number of times to run the k-means algorithm with different "
            "centroid seeds. The final results will be the best output of "
            "``n_init`` consecutive runs in terms of inertia"
        ),
        default=10,
    )

    inputs.int(
        "max_iter",
        label="Maximum iterations",
        description="The maximum number of iterations to perform",
        default=300,
    )

    inputs.float(
        "tol",
        label="Tolerance",
        description=(
            "The relative tolerance with regards to inertia to declare "
            "convergence"
        ),
        default=0.0001,
    )

    inputs.int(
        "random_state",
        label="Random state",
        description="The random state to use for the algorithm",
    )


def _handle_birch_inputs(ctx, inputs):
    inputs.int(
        "n_clusters",
        label="Number of clusters",
        description="The number of clusters to create",
        default=3,
        required=True,
    )

    inputs.float(
        "threshold",
        label="Threshold",
        description="The threshold to stop splitting clusters",
        default=0.5,
    )

    inputs.int(
        "branching_factor",
        label="Branching factor",
        description="The branching factor of the tree",
        default=50,
    )

    inputs.bool(
        "compute_labels",
        default=True,
        label="Compute labels",
        description="Whether to compute labels for each cluster",
    )

    inputs.bool(
        "copy",
        default=True,
        label="Copy",
        description="Whether to copy the input data",
    )


def _handle_agglomerative_inputs(ctx, inputs):
    inputs.int(
        "n_clusters",
        label="Number of clusters",
        description="The number of clusters to create",
        default=2,
        required=True,
    )

    linkage_choices = ("ward", "complete", "average", "single")
    linkage_group = types.DropdownView()

    for choice in linkage_choices:
        linkage_group.add_choice(choice, label=choice)

    inputs.enum(
        "linkage",
        linkage_group.values(),
        label="Linkage",
        description="The linkage criterion to use",
        view=linkage_group,
        default=linkage_choices[0],
    )

    metric_choices = ("euclidean", "l1", "l2", "manhattan", "cosine")
    metric_group = types.DropdownView()

    for choice in metric_choices:
        metric_group.add_choice(choice, label=choice)

    inputs.enum(
        "metric",
        metric_group.values(),
        label="Metric",
        description="The metric to use",
        view=metric_group,
        default=metric_choices[0],
    )

    inputs.int(
        "connectivity",
        label="Connectivity",
        description="The connectivity to use",
    )

    inputs.float(
        "distance_threshold",
        label="Distance threshold",
        description="The distance threshold to use",
    )

    inputs.bool(
        "compute_full_tree",
        label="Compute full tree",
        description="Whether to compute the full tree",
    )

    inputs.bool(
        "compute_distances",
        label="Compute distances",
        description="Whether to compute distances",
        default=False,
    )


def _handle_method_input_routing(ctx, inputs):
    method = ctx.params.get("method", None)

    if method == "kmeans":
        _handle_kmeans_inputs(ctx, inputs)
    elif method == "birch":
        _handle_birch_inputs(ctx, inputs)
    elif method == "agglomerative":
        _handle_agglomerative_inputs(ctx, inputs)


class ComputeClusters(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_clusters",
            label="Clustering: create clusters",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Clustering", description="Create clusters from embeddings"
        )

        ready = brain_init(ctx, inputs)
        if not ready:
            return types.Property(inputs, view=form_view)

        _handle_basic_inputs(ctx, inputs)
        _handle_method_input_routing(ctx, inputs)
        _list_target_views(ctx, inputs)
        _execution_mode(ctx, inputs)

        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        kwargs = ctx.params.copy()
        target = kwargs.pop("target", None)
        target_view = get_target_view(ctx, target)

        method = kwargs.pop("method", None)
        patches_field = kwargs.pop("patches_field", None)
        embeddings = kwargs.pop("embeddings", None)
        embeddings_field = kwargs.pop("embeddings_field", None)
        brain_key = kwargs.pop("brain_key")
        model = kwargs.pop("model", None)
        cluster_field = kwargs.pop("cluster_field", None)

        compute_clusters(
            target_view,
            brain_key=brain_key,
            method=method,
            model=model,
            patches_field=patches_field,
            cluster_field=cluster_field,
            embeddings_field=embeddings_field,
            embeddings=embeddings,
            **kwargs,
        )

        ctx.trigger("reload_dataset")


def register(plugin):
    plugin.register(ComputeClusters)
