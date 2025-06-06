# **Paper Index of XAI for GNNs in KDD**

## Year 2024

**1. Unveiling Global Interactive Patterns across Graphs: Towards Interpretable Graph Neural Networks.**(1) *Yuwen Wang, Shunyu Liu, Tongya Zheng, Kaixuan Chen, Mingli Song.* [[paper]](https://dl.acm.org/doi/10.1145/3637528.3671838)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) have emerged as a prominent framework for graph mining, leading to significant advances across various domains. Stemmed from the node-wise representations of GNNs, existing explanation studies have embraced the subgraph-specific viewpoint that attributes the decision results to the salient features and local structures of nodes. However, graph-level tasks necessitate long-range dependencies and global interactions for advanced GNNs, deviating significantly from subgraph-specific explanations. To bridge this gap, this paper proposes a novel intrinsically interpretable scheme for graph classification, termed as Global Interactive Pattern (GIP) learning, which introduces learnable global interactive patterns to explicitly interpret decisions. GIP first tackles the complexity of interpretation by clustering numerous nodes using a constrained graph clustering module. Then, it matches the coarsened global interactive instance with a batch of self-interpretable graph prototypes, thereby facilitating a transparent graph-level reasoning process. Extensive experiments conducted on both synthetic and real-world benchmarks demonstrate that the proposed GIP yields significantly superior interpretability and competitive performance to the state-of-the-art counterparts. Our code will be made publicly available.

## Year 2023


**1. Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks.**(1) *Gaotang Li, Marlena Duda, Xiang Zhang, Danai Koutra, Yujun Yan.* [[paper]](https://dl.acm.org/doi/10.1145/3580305.3599394)
{ .annotate }

1.  **Abstract**
   
    Brain graphs, which model the structural and functional relationships between brain regions, are crucial in neuroscientific and clinical applications that can be formulated as graph classification tasks. However, dense brain graphs pose computational challenges such as large time and memory consumption and poor model interpretability. In this paper, we investigate effective designs in Graph Neural Networks (GNNs) to sparsify brain graphs by eliminating noisy edges. Many prior works select noisy edges based on explainability or task-irrelevant properties, but this does not guarantee performance improvement when using the sparsified graphs. Additionally, the selection of noisy edges is often tailored to each individual graph, making it challenging to sparsify multiple graphs collectively using the same approach.

    To address the issues above, we first introduce an iterative framework to analyze the effectiveness of different sparsification models. By utilizing this framework, we find that (i) methods that prioritize interpretability may not be suitable for graph sparsification, as the sparsified graphs may degenerate the performance of GNN models; (ii) it is beneficial to learn the edge selection during the training of the GNN, rather than after the GNN has converged; (iii) learning a joint edge selection shared across all graphs achieves higher performance than generating separate edge selection for each graph; and (iv) gradient information, which is task-relevant, helps with edge selection. Based on these insights, we propose a new model, Interpretable Graph Sparsification (IGS), which improves the graph classification performance by up to 5.1% with 55.0% fewer edges than the original graphs. The retained edges identified by IGS provide neuroscientific interpretations and are supported by well-established literature.