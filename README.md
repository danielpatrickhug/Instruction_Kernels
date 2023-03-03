# Instruction Kernels

- This is a repo of some of the work I've done while working on Open Assistant. mainly for topic modeling.

TODO:
Use max instead of sum for gnn

## Main Idea:
Kernels and message passing and feature aggregation
```
Message passing and feature aggregation are effective techniques for improving the quality of topic clusters in a graph-based topic modeling system. 
Message passing involves propagating information through the edges of a graph using matrix exponentials, which allows information to be shared between 
nodes and helps to capture the relationships between them. This allows for more accurate modeling of topic clusters and helps to identify hidden themes 
that may not be apparent in the raw data. Feature aggregation involves summarizing the information contained in the neighboring nodes and using this 
summary to update the features of the current node. The information passing in the local neighborhood is `multiplicative`, the node "communicate" with each other. This helps to capture the shared characteristics of the neighboring nodes and leads to better 
representation of the topics in the graph. By combining these two techniques, the topic model is able to identify more coherent and meaningful topic 
clusters, and produces results that are more informative and useful for downstream analysis.
```

## What in this repo?
`cos_sim(a, b)`: 
```
Computes the cosine similarity between all pairs of vectors in two tensors a and b, and returns the resulting similarity matrix.
```
`cos_sim_torch(embs_a, embs_b)`:
```
Computes the cosine similarity between all pairs of vectors in two tensors embs_a and embs_b using PyTorch's cosine_similarity function, and returns the resulting similarity matrix.
```
`gaussian_kernel_torch(embs_a, embs_b, sigma=1.0)`: 
```
Computes the Gaussian kernel matrix between two sets of embeddings embs_a and embs_b using PyTorch, with a given kernel width sigma, and returns the resulting kernel matrix.
```
`compute_cos_sim_kernel(embs, threshold=0.65, kernel_type="cosine", sigma=1.0)`: 
```
Computes a similarity or kernel matrix between a set of embeddings embs, using either cosine similarity or Gaussian kernel, and applies a threshold to convert the matrix into a binary adjacency matrix.
```
`k_hop_message_passing(A, node_features, k)`: 
```
Computes the k-hop adjacency matrix and aggregated features using message passing, given an adjacency matrix A and feature matrix node_features for a graph, and the number of hops k for message passing.
```
`k_hop_message_passing_sparse(A, node_features, k)`: 
```
A sparse version of k_hop_message_passing that uses sparse matrices.
```
`prune_ref_docs(qa_embs, ref_embs, ref_docs, threshold=0.1)`: 
```
Drops unnecessary documents from the reference embeddings ref_embs and updates the list of reference documents ref_docs, and then recomputes the adjacency matrix between the QA embeddings qa_embs and the pruned reference embeddings, using a threshold to identify unnecessary documents.
```
`load_topic_model(args)`: 
```
Loads a topic model with specified arguments, including a CountVectorizer model for text preprocessing, a ClassTfidfTransformer model for calculating TF-IDF values, a pre-trained SentenceTransformer model for generating sentence embeddings, and a MaximalMarginalRelevance model for selecting diverse topic representatives. UMAP and HDBSCAN
```
`fit_topic_model(topic_model, data, embeddings, key="query")`: 
```
Fits the topic model to a dataset data with precomputed embeddings embeddings, and returns the resulting topics and their probabilities.
```
`get_topic_info(topic_model)`: 
```
Retrieves information about the topics in the topic model, including their IDs, frequency, and representative documents.
```
`reduce_topics(topic_model, data, nr_topics, key="query")`: 
```
Reduces the number of topics in the topic model to a specified number nr_topics, and returns the updated topic model.
```
`get_representative_docs(topic_model)`: 
```
Retrieves the representative documents for each topic in the topic model.
```
`reduce_outliers(topic_model, data, topics, probs, key="query", strategy="c-tf-idf")`: 
```
Reduces the number of topics in the topic model by merging topics that have low similarity, using a specified outlier reduction strategy (c-tf-idf, embeddings, or distributions), and returns the updated topics.
```
`compute_hierarchical_topic_tree(topic_model, data, key="query")`: 
```
Computes a hierarchical topic tree for the topics in the topic model, and returns the tree structure and the topic hierarchy.
```
