# Clustering Plugin for FiftyOne

This plugin provides a FiftyOne App that allows you to cluster your dataset using a variety of algorithms:

- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)
- [Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch)
- [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)

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

### Clustering

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

### Visualizing Clusters

It can be insightful to use clustering in conjunction with `compute_visualization` to visualize the clusters:

![visualize_clusters](https://github.com/jacobmarks/clustering-runs-plugin/assets/12500356/2c48fdcb-c59c-4b46-a27f-a248a6974d4c)

### Labeling Clusters

Once you have generated clusters, you can also use the magic of multimodal AI to automatically
assign short descriptions, or labels to each cluster!

This is achieved by randomly selecting a few samples from each cluster, and prompting
GPT-4V to generate a description for the cluster from the samples.

To use this functionality, you must have an API key for OpenAI's GPT-4V API, and you must set it in your environment as `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=your-api-key
```

Then, you can label the clusters using the `label_clusters_with_gpt4v` operator.
This might take a minute or so, depending on the number of clusters, but it is worth it!
It is recommended to delegate the execution of this operation, and then launch it via

```bash
fiftyone delegated launch
```

Then you can view the labels in the App!
