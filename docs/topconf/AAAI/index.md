# **Paper Index of XAI for GNNs in AAAI**

## Year 2024

**1. Factorized Explainer for Graph Neural Networks.**(1) *Rundong Huang, Farhad Shirani, Dongsheng Luo.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29157)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) have received increasing attention due to their ability to learn from graph-structured data. To open the black-box of these deep learning models, post-hoc instance-level explanation methods have been proposed to understand GNN predictions. These methods seek to discover substructures that explain the prediction behavior of a trained GNN. In this paper, we show analytically that for a large class of explanation tasks, conventional approaches, which are based on the principle of graph information bottleneck (GIB), admit trivial solutions that do not align with the notion of explainability. Instead, we argue that a modified GIB principle may be used to avoid the aforementioned trivial solutions. We further introduce a novel factorized explanation model with theoretical performance guarantees. The modified GIB is used to analyze the structural properties of the proposed factorized explainer. We conduct extensive experiments on both synthetic and real-world datasets to validate the effectiveness of our proposed factorized explainer.

## Year 2023

**1. Interpreting Unfairness in Graph Neural Networks via Training Node Attribution.**(1) *Yushun Dong, Song Wang, Jing Ma, Ninghao Liu, Jundong Li.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25905)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) have emerged as the leading paradigm for solving graph analytical problems in various real-world applications. Nevertheless, GNNs could potentially render biased predictions towards certain demographic subgroups. Understanding how the bias in predictions arises is critical, as it guides the design of GNN debiasing mechanisms. However, most existing works overwhelmingly focus on GNN debiasing, but fall short on explaining how such bias is induced. In this paper, we study a novel problem of interpreting GNN unfairness through attributing it to the influence of training nodes. Specifically, we propose a novel strategy named Probabilistic Distribution Disparity (PDD) to measure the bias exhibited in GNNs, and develop an algorithm to efficiently estimate the influence of each training node on such bias. We verify the validity of PDD and the effectiveness of influence estimation through experiments on real-world datasets. Finally, we also demonstrate how the proposed framework could be used for debiasing GNNs. Open-source code can be found at https://github.com/yushundong/BIND.

**2. Towards Fine-Grained Explainability for Heterogeneous Graph Neural Network.**(1) *Tong Li, Jiale Deng, Yanyan Shen, Luyu Qiu, Yongxiang Huang, Caleb Chen Cao.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26040)
{ .annotate }

1.  **Abstract**
   
    Heterogeneous graph neural networks (HGNs) are prominent approaches to node classification tasks on heterogeneous graphs. Despite the superior performance, insights about the predictions made from HGNs are obscure to humans. Existing explainability techniques are mainly proposed for GNNs on homogeneous graphs. They focus on highlighting salient graph objects to the predictions whereas the problem of how these objects affect the predictions remains unsolved. Given heterogeneous graphs with complex structures and rich semantics, it is imperative that salient objects can be accompanied with their influence paths to the predictions, unveiling the reasoning process of HGNs. In this paper, we develop xPath, a new framework that provides fine-grained explanations for black-box HGNs specifying a cause node with its influence path to the target node. In xPath, we differentiate the influence of a node on the prediction w.r.t. every individual influence path, and measure the influence by perturbing graph structure via a novel graph rewiring algorithm. Furthermore, we introduce a greedy search algorithm to find the most influential fine-grained explanations efficiently. Empirical results on various HGNs and heterogeneous graphs show that xPath yields faithful explanations efficiently, outperforming the adaptations of advanced GNN explanation approaches.

**3. Interpretable Chirality-Aware Graph Neural Network for Quantitative Structure Activity Relationship Modeling in Drug Discovery.**(1) *Yunchao Liu, Yu Wang, Oanh Vu, Rocco Moretti, Bobby Bodenheimer, Jens Meiler, Tyler Derr.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26679)
{ .annotate }

1.  **Abstract**
   
    In computer-aided drug discovery, quantitative structure activity relation models are trained to predict biological activity from chemical structure. Despite the recent success of applying graph neural network to this task, important chemical information such as molecular chirality is ignored. To fill this crucial gap, we propose Molecular-Kernel Graph NeuralNetwork (MolKGNN) for molecular representation learning, which features SE(3)-/conformation invariance, chirality-awareness, and interpretability. For our MolKGNN, we first design a molecular graph convolution to capture the chemical pattern by comparing the atom's similarity with the learnable molecular kernels. Furthermore, we propagate the similarity score to capture the higher-order chemical pattern. To assess the method, we conduct a comprehensive evaluation with nine well-curated datasets spanning numerous important drug targets that feature realistic high class imbalance and it demonstrates the superiority of MolKGNN over other graph neural networks in computer-aided drug discovery. Meanwhile, the learned kernels identify patterns that agree with domain knowledge, confirming the pragmatic interpretability of this approach. Our code and supplementary material are publicly available at https://github.com/meilerlab/MolKGNN.

**4. Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis.**(1) *Han Xuanyuan, Pietro Barbiero, Dobrik Georgiev, Lucie Charlotte Magister, Pietro Liò.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26267)
{ .annotate }

1.  **Abstract**
   
    Graph neural networks (GNNs) are highly effective on a variety of graph-related tasks; however, they lack interpretability and transparency. Current explainability approaches are typically local and treat GNNs as black-boxes. They do not look inside the model, inhibiting human trust in the model and explanations. Motivated by the ability of neurons to detect high-level semantic concepts in vision models, we perform a novel analysis on the behaviour of individual GNN neurons to answer questions about GNN interpretability. We propose a novel approach for producing global explanations for GNNs using neuron-level concepts to enable practitioners to have a high-level view of the model. Specifically, (i) to the best of our knowledge, this is the first work which shows that GNN neurons act as concept detectors and have strong alignment with concepts formulated as logical compositions of node degree and neighbourhood properties; (ii) we quantitatively assess the importance of detected concepts, and identify a trade-off between training duration and neuron-level interpretability; (iii) we demonstrate that our global explainability approach has advantages over the current state-of-the-art -- we can disentangle the explanation into individual interpretable concepts backed by logical descriptions, which reduces potential for bias and improves user-friendliness.

## Year 2022

**1. KerGNNs: Interpretable Graph Neural Networks with Graph Kernels.**(1) *Aosong Feng, Chenyu You, Shiqiang Wang, Leandros Tassiulas.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20615)
{ .annotate }

1.  **Abstract**
   
    Graph kernels are historically the most widely-used technique for graph classification tasks. However, these methods suffer from limited performance because of the hand-crafted combinatorial features of graphs. In recent years, graph neural networks (GNNs) have become the state-of-the-art method in downstream graph-related tasks due to their superior performance. Most GNNs are based on Message Passing Neural Network (MPNN) frameworks. However, recent studies show that MPNNs can not exceed the power of the Weisfeiler-Lehman (WL) algorithm in graph isomorphism test. To address the limitations of existing graph kernel and GNN methods, in this paper, we propose a novel GNN framework, termed Kernel Graph Neural Networks (KerGNNs), which integrates graph kernels into the message passing process of GNNs. Inspired by convolution filters in convolutional neural networks (CNNs), KerGNNs adopt trainable hidden graphs as graph filters which are combined with subgraphs to update node embeddings using graph kernels. In addition, we show that MPNNs can be viewed as special cases of KerGNNs. We apply KerGNNs to multiple graph-related tasks and use cross-validation to make fair comparisons with benchmarks. We show that our method achieves competitive performance compared with existing state-of-the-art methods, demonstrating the potential to increase the representation ability of GNNs. We also show that the trained graph filters in KerGNNs can reveal the local graph structures of the dataset, which significantly improves the model interpretability compared with conventional GNN models.


**2. Interpretable Neural Subgraph Matching for Graph Retrieval.**(1) *Indradyumna Roy, Venkata Sai Baba Reddy Velugoti, Soumen Chakrabarti, Abir De.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20784)
{ .annotate }

1.  **Abstract**
   
    Given a query graph and a database of corpus graphs, a graph retrieval system aims to deliver the most relevant corpus graphs. Graph retrieval based on subgraph matching has a wide variety of applications, e.g., molecular fingerprint detection, circuit design, software analysis, and question answering. In such applications, a corpus graph is relevant to a query graph, if the query graph is (perfectly or approximately) a subgraph of the corpus graph. Existing neural graph retrieval models compare the node or graph embeddings of the query-corpus pairs, to compute the relevance scores between them. However, such models may not provide edge consistency between the query and corpus graphs. Moreover, they predominantly use symmetric relevance scores, which are not appropriate in the context of subgraph matching, since the underlying relevance score in subgraph search should be measured using the partial order induced by subgraph-supergraph relationship. Consequently, they show poor retrieval performance in the context of subgraph matching. In response, we propose ISONET, a novel interpretable neural edge alignment formulation, which is better able to learn the edge-consistent mapping necessary for subgraph matching. ISONET incorporates a new scoring mechanism which enforces an asymmetric relevance score, specifically tailored to subgraph matching. ISONET’s design enables it to directly identify the underlying subgraph in a corpus graph, which is relevant to the given query graph. Our experiments on diverse datasets show that ISONET outperforms recent graph retrieval formulations and systems. Additionally, ISONET can provide interpretable alignments between query-corpus graph pairs during inference, despite being trained only using binary relevance labels of whole graphs during training, without any fine-grained ground truth information about node or edge alignments.

**3. ProtGNN: Towards Self-Explaining Graph Neural Networks.**(1) *Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, Cheekong Lee.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20898)
{ .annotate }

1.  **Abstract**
   
    Despite the recent progress in Graph Neural Networks (GNNs), it remains challenging to explain the predictions made by GNNs. Existing explanation methods mainly focus on post-hoc explanations where another explanatory model is employed to provide explanations for a trained GNN. The fact that post-hoc methods fail to reveal the original reasoning process of GNNs raises the need of building GNNs with built-in interpretability. In this work, we propose Prototype Graph Neural Network (ProtGNN), which combines prototype learning with GNNs and provides a new perspective on the explanations of GNNs. In ProtGNN, the explanations are naturally derived from the case-based reasoning process and are actually used during classification. The prediction of ProtGNN is obtained by comparing the inputs to a few learned prototypes in the latent space. Furthermore, for better interpretability and higher efficiency, a novel conditional subgraph sampling module is incorporated to indicate which part of the input graph is most similar to each prototype in ProtGNN+. Finally, we evaluate our method on a wide range of datasets and perform concrete case studies. Extensive results show that ProtGNN and ProtGNN+ can provide inherent interpretability while achieving accuracy on par with the non-interpretable counterparts.


## Year 2021

**1. Interpretable Embedding Procedure Knowledge Transfer via Stacked Principal Component Analysis and Graph Neural Network.**(1) *Seunghyun Lee, Byung Cheol Song.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17009)
{ .annotate }

1.  **Abstract**
   
    Knowledge distillation (KD) is one of the most useful techniques for light-weight neural networks. Although neural networks have a clear purpose of embedding datasets into the low-dimensional space, the existing knowledge was quite far from this purpose and provided only limited information. We argue that good knowledge should be able to interpret the embedding procedure. This paper proposes a method of generating interpretable embedding procedure (IEP) knowledge based on principal component analysis, and distilling it based on a message passing neural network. Experimental results show that the student network trained by the proposed KD method improves 2.28% in the CIFAR100 dataset, which is a higher performance than the state-of-the-art (SOTA) method. We also demonstrate that the embedding procedure knowledge is interpretable via visualization of the proposed KD process. The implemented code is available at https://github.com/sseung0703/IEPKT.

**2. Interpretable Clustering on Dynamic Graphs with Recurrent Graph Neural Networks.**(1) *Yuhang Yao, Carlee Joe-Wong.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16590)
{ .annotate }

1.  **Abstract**
   
    We study the problem of clustering nodes in a dynamic graph, where the connections between nodes and nodes' cluster memberships may change over time, e.g., due to community migration. We first propose a dynamic stochastic block model that captures these changes, and a simple decay-based clustering algorithm that clusters nodes based on weighted connections between them, where the weight decreases at a fixed rate over time. This decay rate can then be interpreted as signifying the importance of including historical connection information in the clustering. However, the optimal decay rate may differ for clusters with different rates of turnover. We characterize the optimal decay rate for each cluster and propose a clustering method that achieves almost exact recovery of the true clusters. We then demonstrate the efficacy of our clustering algorithm with optimized decay rates on simulated graph data. Recurrent neural networks (RNNs), a popular algorithm for sequence learning, use a similar decay-based method, and we use this insight to propose two new RNN-GCN (graph convolutional network) architectures for semi-supervised graph clustering. We finally demonstrate that the proposed architectures perform well on real data compared to state-of-the-art graph clustering algorithms.