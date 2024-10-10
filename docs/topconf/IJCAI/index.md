# **Paper Index of XAI for GNNs in IJCAI**

## Year 2023

**1. Interpret ESG Rating's Impact on the Industrial Chain Using Graph Neural Networks.**(1) *Bin Liu, Jiujun He, Ziyuan Li, Xiaoyang Huang, Xiang Zhang, Guosheng Yin.* [[paper]](https://www.ijcai.org/proceedings/2023/674)
{ .annotate }

1.  **Abstract**
   
    We conduct a quantitative analysis of the development of the industry chain from the environmental, social, and governance (ESG) perspective, which is an overall measure of sustainability. Factors that may impact the performance of the industrial chain have been studied in the literature, such as government regulation, monetary policy, etc. Our interest lies in how the sustainability change (i.e., ESG shock) affects the performance of the industrial chain. To achieve this goal, we model the industrial chain with a graph neural network (GNN) and conduct node regression on two financial performance metrics, namely, the aggregated profitability ratios and operating margin. To quantify the effects of ESG, we propose to compute the interaction between ESG shocks and industrial chain features with a cross-attention module, and then filter the original node features in the graph regression. Experiments on two real datasets demonstrate that (i) there are significant effects of ESG shocks on the industrial chain, and (ii) model parameters including regression coefficients and the attention map can explain how ESG shocks affect the performance of the industrial chain.

## Year 2021


**1. Smart Contract Vulnerability Detection: From Pure Neural Network to Interpretable Graph Feature and Expert Pattern Fusion.**(1) *Zhenguang Liu, Peng Qian, Xiang Wang, Lei Zhu, Qinming He, Shouling Ji.* [[paper]](https://www.ijcai.org/proceedings/2021/379)
{ .annotate }

1.  **Abstract**
   
    Smart contracts hold digital coins worth billions of dollars, their security issues have drawn extensive attention in the past years. Towards smart contract vulnerability detection, conventional methods heavily rely on fixed expert rules, leading to low accuracy and poor scalability. Recent deep learning approaches alleviate this issue but fail to encode useful expert knowledge. In this paper, we explore combining deep learning with expert patterns in an explainable fashion. Specifically, we develop automatic tools to extract expert patterns from the source code. We then cast the code into a semantic graph to extract deep graph features. Thereafter, the global graph feature and local expert patterns are fused to cooperate and approach the final prediction, while yielding their interpretable weights. Experiments are conducted on all available smart contracts with source code in two platforms, Ethereum and VNT Chain. Empirically, our system significantly outperforms state-of-the-art methods. Our code is released.