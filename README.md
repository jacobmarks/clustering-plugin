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

![compute_clusters_from_scratch](https://github.com/jacobmarks/clustering-runs-plugin/assets/12500356/c701d40a-ddf4-47a7-bb5d-8f026a54bb6e)


The specific arguments depend on the `method` you choose — `kmeans`, `birch`, or `agglomerative`.

Here, we are generating clusters at the same time as we are generating the embeddings, but you can also generate clusters from existing embeddings:

![compute_clusters_from_embeddings](https://github.com/jacobmarks/clustering-runs-plugin/assets/12500356/950c10d7-9d7e-4876-a2ea-66574e594607)


You can generate clusters for:

- Your entire dataset
- A view of your dataset
- Currently selected samples in the App

Additionally, you can run the operator in:

- Real-time, or
- In the background, as a delegated operation

Once you have generated clusters, you can view information about the clusters in the App with the `get_clustering_run_info` operator:

![get_cluster_info](https://github.com/jacobmarks/clustering-runs-plugin/assets/12500356/63660858-091f-4a94-865e-a3fb41c2c2c6)


Finally, it is insightful to use clustering in conjunction with `compute_visualization` to visualize the clusters:

![visualize_clusters](https://github.com/jacobmarks/clustering-runs-plugin/assets/12500356/2c48fdcb-c59c-4b46-a27f-a248a6974d4c)
