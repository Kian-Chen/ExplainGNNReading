# **Paper Index of XAI for GNNs in ICML**

## Year 2024

**1. How Interpretable Are Interpretable Graph Neural Networks?**(1) *Yongqiang Chen, Yatao Bian, Bo Han, James Cheng.* [[paper]](https://openreview.net/forum?id=F3G2udCF3Q)
{ .annotate }

1.  **Abstract**
   
    Interpretable graph neural networks (XGNNs ) are widely adopted in various scientific applications involving graph-structured data. Existing XGNNs predominantly adopt the attention-based mechanism to learn edge or node importance for extracting and making predictions with the interpretable subgraph. However, the representational properties and limitations of these methods remain inadequately explored. In this work, we present a theoretical framework that formulates interpretable subgraph learning with the multilinear extension of the subgraph distribution, coined as subgraph multilinear extension (SubMT). Extracting the desired interpretable subgraph requires an accurate approximation of SubMT, yet we find that the existing XGNNs can have a huge gap in fitting SubMT. Consequently, the SubMT approximation failure will lead to the degenerated interpretability of the extracted subgraphs. To mitigate the issue, we design a new XGNN architecture called Graph Multilinear neT (GMT), which is provably more powerful in approximating SubMT. We empirically validate our theoretical findings on a number of graph classification benchmarks. The results demonstrate that GMT outperforms the state-of-the-art up to 10% in terms of both interpretability and generalizability across 12 regular and geometric graph benchmarks.

**2. Explaining Graph Neural Networks via Structure-aware Interaction Index.**(1) *Ngoc Bui, Hieu Trung Nguyen, Viet Anh Nguyen, Rex Ying.* [[paper]](https://openreview.net/forum?id=2T00oYk54P)
{ .annotate }

1.  **Abstract**
   
    The Shapley value is a prominent tool for interpreting black-box machine learning models thanks to its strong theoretical foundation. However, for models with structured inputs, such as graph neural networks, existing Shapley-based explainability approaches either focus solely on node-wise importance or neglect the graph structure when perturbing the input instance. This paper introduces the Myerson-Taylor interaction index that internalizes the graph structure into attributing the node values and the interaction values among nodes. Unlike the Shapley-based methods, the Myerson-Taylor index decomposes coalitions into components satisfying a pre-chosen connectivity criterion. We prove that the Myerson-Taylor index is the unique one that satisfies a system of five natural axioms accounting for graph structure and high-order interaction among nodes. Leveraging these properties, we propose Myerson-Taylor Structure-Aware Graph Explainer (MAGE), a novel explainer that uses the second-order Myerson-Taylor index to identify the most important motifs influencing the model prediction, both positively and negatively. Extensive experiments on various graph datasets and models demonstrate that our method consistently provides superior subgraph explanations compared to state-of-the-art methods.

**3. Generating In-Distribution Proxy Graphs for Explaining Graph Neural Networks.**(1) *Zhuomin Chen, Jiaxing Zhang, Jingchao Ni, Xiaoting Li, Yuchen Bian, Md Mezbahul Islam, Ananda Mondal, Hua Wei, Dongsheng Luo.* [[paper]](https://openreview.net/forum?id=ohG9bVMs5j)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) have become a building block in graph data processing, with wide applications in critical domains. The growing needs to deploy GNNs in high-stakes applications necessitate explainability for users in the decision-making processes. A popular paradigm for the explainability of GNNs is to identify explainable subgraphs by comparing their labels with the ones of original graphs. This task is challenging due to the substantial distributional shift from the original graphs in the training set to the set of explainable subgraphs, which prevents accurate prediction of labels with the subgraphs. To address it, in this paper, we propose a novel method that generates proxy graphs for explainable subgraphs that are in the distribution of training data. We introduce a parametric method that employs graph generators to produce proxy graphs. A new training objective based on information theory is designed to ensure that proxy graphs not only adhere to the distribution of training data but also preserve explanatory factors. Such generated proxy graphs can be reliably used to approximate the predictions of the labels of explainable subgraphs. Empirical evaluations across various datasets demonstrate our method achieves more accurate explanations for GNNs.


**4. Predicting and Interpreting Energy Barriers of Metallic Glasses with Graph Neural Networks.**(1) *Haoyu Li, Shichang Zhang, Longwen Tang, Mathieu Bauchy, Yizhou Sun.* [[paper]](https://openreview.net/forum?id=7rTbqkKvA6)
{ .annotate }

1.  **Abstract**
   
    Metallic Glasses (MGs) are widely used materials that are stronger than steel while being shapeable as plastic. While understanding the structure-property relationship of MGs remains a challenge in materials science, studying their energy barriers (EBs) as an intermediary step shows promise. In this work, we utilize Graph Neural Networks (GNNs) to model MGs and study EBs. We contribute a new dataset for EB prediction and a novel Symmetrized GNN (SymGNN) model that is E(3)-invariant in expectation. SymGNN handles invariance by aggregating over orthogonal transformations of the graph structure. When applied to EB prediction, SymGNN are more accurate than molecular dynamics (MD) local-sampling methods and other machine-learning models. Compared to precise MD simulations, SymGNN reduces the inference time on new MGs from roughly 41 days to less than one second. We apply explanation algorithms to reveal the relationship between structures and EBs. The structures that we identify through explanations match the medium-range order (MRO) hypothesis and possess unique topological properties. Our work enables effective prediction and interpretation of MG EBs, bolstering material science research.

## Year 2023


**1. Rethinking Explaining Graph Neural Networks via Non-parametric Subgraph Matching.**(1) *Fang Wu, Siyuan Li, Xurui Jin, Yinghui Jiang, Dragomir Radev, Zhangming Niu, Stan Z. Li.* [[paper]](https://proceedings.mlr.press/v202/wu23j.html)
{ .annotate }

1.  **Abstract**
   
    The success of graph neural networks (GNNs) provokes the question about explainability: “Which fraction of the input graph is the most determinant of the prediction?” Particularly, parametric explainers prevail in existing approaches because of their more robust capability to decipher the black-box (i.e., target GNNs). In this paper, based on the observation that graphs typically share some common motif patterns, we propose a novel non-parametric subgraph matching framework, dubbed MatchExplainer, to explore explanatory subgraphs. It couples the target graph with other counterpart instances and identifies the most crucial joint substructure by minimizing the node corresponding-based distance. Moreover, we note that present graph sampling or node-dropping methods usually suffer from the false positive sampling problem. To alleviate this issue, we design a new augmentation paradigm named MatchDrop. It takes advantage of MatchExplainer to fix the most informative portion of the graph and merely operates graph augmentations on the rest less informative part. Extensive experiments on synthetic and real-world datasets show the effectiveness of our MatchExplainer by outperforming all state-of-the-art parametric baselines with significant margins. Results also demonstrate that MatchDrop is a general scheme to be equipped with GNNs for enhanced performance. The code is available at https://github.com/smiles724/MatchExplainer.


**2. Relevant Walk Search for Explaining Graph Neural Networks.**(1)  *Ping Xiong, Thomas Schnake, Michael Gastegger, Grégoire Montavon, Klaus-Robert Müller, Shinichi Nakajima.* [[paper]](https://proceedings.mlr.press/v202/xiong23b.html)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) have become important machine learning tools for graph analysis, and its explainability is crucial for safety, fairness, and robustness. Layer-wise relevance propagation for GNNs (GNN-LRP) evaluates the relevance of walks to reveal important information flows in the network, and provides higher-order explanations, which have been shown to be superior to the lower-order, i.e., node-/edge-level, explanations. However, identifying relevant walks by GNN-LRP requires exponential computational complexity with respect to the network depth, which we will remedy in this paper. Specifically, we propose polynomial-time algorithms for finding top-$K$ relevant walks, which drastically reduces the computation and thus increases the applicability of GNN-LRP to large-scale problems. Our proposed algorithms are based on the max-product algorithm—a common tool for finding the maximum likelihood configurations in probabilistic graphical models—and can find the most relevant walks exactly at the neuron level and approximately at the node level. Our experiments demonstrate the performance of our algorithms at scale and their utility across application domains, i.e., on epidemiology, molecular, and natural language benchmarks. We provide our codes under https://github.com/xiong-ping/rel_walk_gnnlrp.




## Year 2021


**1. Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity.**(1) *Ryan Henderson, Djork-Arné Clevert, Floriane Montanari.* [[paper]](https://proceedings.mlr.press/v139/henderson21a.html)
{ .annotate }

1.  **Abstract**
   
    Rationalizing which parts of a molecule drive the predictions of a molecular graph convolutional neural network (GCNN) can be difficult. To help, we propose two simple regularization techniques to apply during the training of GCNNs: Batch Representation Orthonormalization (BRO) and Gini regularization. BRO, inspired by molecular orbital theory, encourages graph convolution operations to generate orthonormal node embeddings. Gini regularization is applied to the weights of the output layer and constrains the number of dimensions the model can use to make predictions. We show that Gini and BRO regularization can improve the accuracy of state-of-the-art GCNN attribution methods on artificial benchmark datasets. In a real-world setting, we demonstrate that medicinal chemists significantly prefer explanations extracted from regularized models. While we only study these regularizers in the context of GCNNs, both can be applied to other types of neural networks.

**2. On Explainability of Graph Neural Networks via Subgraph Explorations.**(1) *Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, Shuiwang Ji.* [[paper]](https://proceedings.mlr.press/v139/yuan21c.html)
{ .annotate }

1.  **Abstract**
   
    We consider the problem of explaining the predictions of graph neural networks (GNNs), which otherwise are considered as black boxes. Existing methods invariably focus on explaining the importance of graph nodes or edges but ignore the substructures of graphs, which are more intuitive and human-intelligible. In this work, we propose a novel method, known as SubgraphX, to explain GNNs by identifying important subgraphs. Given a trained GNN model and an input graph, our SubgraphX explains its predictions by efficiently exploring different subgraphs with Monte Carlo tree search. To make the tree search more effective, we propose to use Shapley values as a measure of subgraph importance, which can also capture the interactions among different subgraphs. To expedite computations, we propose efficient approximation schemes to compute Shapley values for graph data. Our work represents the first attempt to explain GNNs via identifying subgraphs explicitly and directly. Experimental results show that our SubgraphX achieves significantly improved explanations, while keeping computations at a reasonable level.