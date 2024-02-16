# Clustering Plugin for FiftyOne

This plugin provides a FiftyOne App that allows you to cluster your dataset using a variety of algorithms:

- K-Means
- Birch
- Agglomerative

It also serves as a proof of concept for adding new "types" of runs to FiftyOne!!!

## Installation

```bash
fiftyone plugins download https://github.com/jacobmarks/clustering-runs-plugin
```

You will also need to have `scikit-learn` installed:

```bash
pip install -U scikit-learn
```

## Usage

Once you have the plugin installed, you can generate clusters for your dataset using the
`compute_clusters` operator:

INSERT GIF

The specific arguments depend on the `method` you choose — `kmeans`, `birch`, or `agglomerative`.

Here, we are generating clusters at the same time as we are generating the embeddings, but you can also generate clusters from existing embeddings:

INSERT GIF

You can generate clusters for:

- Your entire dataset
- A view of your dataset
- Currently selected samples in the App

Additionally, you can run the operator in:

- Real-time, or
- In the background, as a delegated operation

Once you have generated clusters, you can view information about the clusters in the App with the `get_clustering_run_info` operator:

INSERT GIF

Finally, it is insightful to use clustering in conjunction with `compute_visualization` to visualize the clusters:

INSERT GIF
