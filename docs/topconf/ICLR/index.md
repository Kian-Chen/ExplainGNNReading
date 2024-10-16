# **Paper Index of XAI for GNNs in ICLR**

## Year 2024

**1. GOAt: Explaining Graph Neural Networks via Graph Output Attribution.**(1) *Shengyao Lu, Keith G. Mills, Jiao He, Bang Liu, Di Niu.* [[paper]](https://openreview.net/forum?id=2Q8TZWAHv4)
{ .annotate }

1.  **Abstract**
   
    Understanding the decision-making process of Graph Neural Networks (GNNs) is crucial to their interpretability. Most existing methods for explaining GNNs typically rely on training auxiliary models, resulting in the explanations remain black-boxed. This paper introduces Graph Output Attribution (GOAt), a novel method to attribute graph outputs to input graph features, creating GNN explanations that are faithful, discriminative, as well as stable across similar samples. By expanding the GNN as a sum of scalar products involving node features, edge features and activation patterns, we propose an efficient analytical method to compute contribution of each node or edge feature to each scalar product and aggregate the contributions from all scalar products in the expansion form to derive the importance of each node and edge. Through extensive experiments on synthetic and real-world data, we show that our method not only outperforms various state-of-the-art GNN explainers in terms of the commonly used fidelity metric, but also exhibits stronger discriminability, and stability by a remarkable margin.

**2. GraphChef: Decision-Tree Recipes to Explain Graph Neural Networks.**(1) *Peter Müller, Lukas Faber, Karolis Martinkus, Roger Wattenhofer.* [[paper]](https://openreview.net/forum?id=IjMUGuUmBI)
{ .annotate }

1.  **Abstract**
   
    We propose a new self-explainable Graph Neural Network (GNN) model: GraphChef. GraphChef integrates decision trees into the GNN message passing framework. Given a dataset, GraphChef returns a set of rules (a recipe) that explains each class in the dataset unlike existing GNNs and explanation methods that reason on individual graphs. Thanks to the decision trees, GraphChef recipes are human understandable. We also present a new pruning method to produce small and easy to digest trees. Experiments demonstrate that GraphChef reaches comparable accuracy to not self-explainable GNNs and produced decision trees are indeed small. We further validate the correctness of the discovered recipes on datasets where explanation ground truth is available: Reddit-Binary, MUTAG, BA-2Motifs, BA-Shapes, Tree-Cycle, and Tree-Grid.

**3. GNNBoundary: Towards Explaining Graph Neural Networks through the Lens of Decision Boundaries.**(1) *Xiaoqi Wang, Han-Wei Shen.* [[paper]](https://openreview.net/forum?id=WIzzXCVYiH)
{ .annotate }

1.  **Abstract**
   
    While Graph Neural Networks (GNNs) have achieved remarkable performance on various machine learning tasks on graph data, they also raised questions regarding their transparency and interpretability. Recently, there have been extensive research efforts to explain the decision-making process of GNNs. These efforts often focus on explaining why a certain prediction is made for a particular instance, or what discriminative features the GNNs try to detect for each class. However, to the best of our knowledge, there is no existing study on understanding the decision boundaries of GNNs, even though the decision-making process of GNNs is directly determined by the decision boundaries. To bridge this research gap, we propose a model-level explainability method called GNNBoundary, which attempts to gain deeper insights into the decision boundaries of graph classifiers. Specifically, we first develop an algorithm to identify the pairs of classes whose decision regions are adjacent. For an adjacent class pair, the near-boundary graphs between them are effectively generated by optimizing a novel objective function specifically designed for boundary graph generation. Thus, by analyzing the nearboundary graphs, the important characteristics of decision boundaries can be uncovered. To evaluate the efficacy of GNNBoundary, we conduct experiments on both synthetic and public real-world datasets. The results demonstrate that, via the analysis of faithful near-boundary graphs generated by GNNBoundary, we can thoroughly assess the robustness and generalizability of the explained GNNs. The official implementation can be found at https://github.com/yolandalalala/GNNBoundary.


**4. Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks.**(1) *Xu Zheng, Farhad Shirani, Tianchun Wang, Wei Cheng, Zhuomin Chen, Haifeng Chen, Hua Wei, Dongsheng Luo.* [[paper]](https://openreview.net/forum?id=up6hr4hIQH)
{ .annotate }

1.  **Abstract**
   
    Graph Neural Networks (GNNs) are neural models that leverage the dependency structure in graphical data via message passing among the graph nodes. GNNs have emerged as pivotal architectures in analyzing graph-structured data, and their expansive application in sensitive domains requires a comprehensive understanding of their decision-making processes --- necessitating a framework for GNN explainability. An explanation function for GNNs takes a pre-trained GNN along with a graph as input, to produce a `sufficient statistic' subgraph with respect to the graph label. A main challenge in studying GNN explainability is to provide fidelity measures that evaluate the performance of these explanation functions. This paper studies this foundational challenge, spotlighting the inherent limitations of prevailing fidelity metrics, including $Fid_+$, $Fid_-$, and $Fid_\Delta$. Specifically, a formal, information-theoretic definition of explainability is introduced and it is shown that existing metrics often fail to align with this definition across various statistical scenarios. The reason is due to potential distribution shifts when subgraphs are removed in computing these fidelity measures. Subsequently, a robust class of fidelity measures are introduced, and it is shown analytically that they are resilient to distribution shift issues and are applicable in a wide range of scenarios. Extensive empirical analysis on both synthetic and real datasets are provided to illustrate that the proposed metrics are more coherent with gold standard metrics.

## Year 2023


**1. DAG Matters! GFlowNets Enhanced Explainer for Graph Neural Networks.**(1) *Wenqian Li, Yinchuan Li, Zhigang Li, Jianye Hao, Yan Pang.* [[paper]](https://openreview.net/forum?id=jgmuRzM-sb6)
{ .annotate }

1.  **Abstract**
   
    Uncovering rationales behind predictions of graph neural networks (GNNs) has received increasing attention over the years. Existing literature mainly focus on selecting a subgraph, through combinatorial optimization, to provide faithful explanations. However, the exponential size of candidate subgraphs limits the applicability of state-of-the-art methods to large-scale GNNs. We enhance on this through a different approach: by proposing a generative structure – GFlowNets-based GNN Explainer (GFlowExplainer), we turn the optimization problem into a step-by-step generative problem. Our GFlowExplainer aims to learn a policy that generates a distribution of subgraphs for which the probability of a subgraph is proportional to its’ reward. The proposed approach eliminates the influence of node sequence and thus does not need any pre-training strategies. We also propose a new cut vertex matrix to efficiently explore parent states for GFlowNets structure, thus making our approach applicable in a large-scale setting. We conduct extensive experiments on both synthetic and real datasets, and both qualitative and quantitative results show the superiority of our GFlowExplainer.


**2. GraphEx: A User-Centric Model-Level Explainer for Graph Neural Networks.**(1)  *Sayan Saha, Monidipa Das, Sanghamitra Bandyopadhyay.* [[paper]](https://openreview.net/forum?id=CuE1F1M0_yR)
{ .annotate }

1.  **Abstract**
   
    With the increasing application of Graph Neural Networks (GNNs) in real-world domains, there is a growing need to understand the decision-making process of these models. To address this, we propose GraphEx, a model-level explainer that learns a graph generative model to approximate the distribution of graphs classified into a target class by the GNN model. Unlike existing methods, GraphEx does not require another black box deep model to explain the GNN and can generate a diverse set of explanation graphs with different node and edge features in one shot. Moreover, GraphEx does not need white box access to the GNN model, making it more accessible to end-users. Experiments on both synthetic and real datasets demonstrate that GraphEx can consistently produce explanations aligned with the class identity and can also identify potential limitations of the GNN model.




## Year 2021


**1. Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking.**(1) *Michael Sejr Schlichtkrull, Nicola De Cao, Ivan Titov.* [[paper]](https://openreview.net/forum?id=WznmQa42ZAx)
{ .annotate }

1.  **Abstract**
   
    Graph neural networks (GNNs) have become a popular approach to integrating structural inductive biases into NLP models. However, there has been little work on interpreting them, and specifically on understanding which parts of the graphs (e.g. syntactic trees or co-reference structures) contribute to a prediction. In this work, we introduce a post-hoc method for interpreting the predictions of GNNs which identifies unnecessary edges. Given a trained GNN model, we learn a simple classifier that, for every edge in every layer, predicts if that edge can be dropped. We demonstrate that such a classifier can be trained in a fully differentiable fashion, employing stochastic gates and encouraging sparsity through the expected $L_0$ norm. We use our technique as an attribution method to analyze GNN models for two tasks -- question answering and semantic role labeling -- providing insights into the information flow in these models. We show that we can drop a large proportion of edges without deteriorating the performance of the model, while we can analyse the remaining edges for interpreting model predictions.


**2. Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering.**(1) *Yuxuan Zhang, Wenzheng Chen, Huan Ling, Jun Gao, Yinan Zhang, Antonio Torralba, Sanja Fidler.* [[paper]](https://openreview.net/forum?id=yWkP7JuHX1)
{ .annotate }

1.  **Abstract**
   
    Differentiable rendering has paved the way to training neural networks to perform “inverse graphics” tasks such as predicting 3D geometry from monocular photographs. To train high performing models, most of the current approaches rely on multi-view imagery which are not readily available in practice.  Recent Generative Adversarial Networks (GANs) that synthesize images, in contrast, seem to acquire 3D knowledge implicitly during training: object viewpoints can be manipulated by simply manipulating the latent codes. However, these latent codes often lack further physical interpretation and thus GANs cannot easily be inverted to perform explicit 3D reasoning. In this paper, we aim to extract and disentangle 3D knowledge learned by generative models by utilizing differentiable renderers. Key to our approach is to exploit GANs as a multi-view data generator to train an inverse graphics network using an off-the-shelf differentiable renderer, and the trained inverse graphics network as a teacher to disentangle the GAN's latent code into interpretable 3D properties. The entire architecture is trained iteratively using cycle consistency losses. We show that our approach significantly outperforms state-of-the-art inverse graphics networks trained on existing datasets, both quantitatively and via user studies. We further showcase the disentangled GAN as a controllable 3D “neural renderer", complementing traditional graphics renderers.