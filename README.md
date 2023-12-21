# Clustering Plugin for FiftyOne

This plugin provides a FiftyOne App that allows you to cluster your dataset using a variety of algorithms:

- K-Means
- Birch
- Agglomerative

It also serves as a proof of concept for adding new "types" of runs to FiftyOne, or the FiftyOne Brain!!!

## Installation

```bash
fiftyone plugins download https://github.com/jacobmarks/clustering-runs-plugin
```

## Operators

This plugin adds a `compute_clusters` operator, in direct analogy with `compute_similarity`, `compute_visualization`, and other Brain methods.

## Usage

Here are some of the ways you can use this plugin:

You can generate clusters for:

A. Your entire dataset
B. A view of your dataset
C. Currently selected samples in the App

You can run the operator in:

A. Real-time
B. In the background, as a delegated operation

## To Do

- [ ] Validate edge cases with embeddings and `embeddings_field`
- [ ] Fully extend support to patches
- [ ] Add more clustering algorithms
- [ ] Add `compute_centroids` and other useful utility operators
