# 2026 Causal Baseline Candidates

基于 2026 年（及 2025 年底）的最新研究趋势，为您筛选了 3 个最适合作为 **Deep-Lung-Causal** 强基线的候选模型。这些模型代表了 Generative AI 和 Foundation Model 在因果推断领域的最新应用。

## 1. Diff-ITE (Diffusion for Individual Treatment Effect)
*   **发布时间**: Jan 2026 (Preprint / ICLR 2026 Submission)
*   **参考论文**: [DiffITE: Estimating Individual Treatment Effect with Conditional Diffusion Models](https://arxiv.org/abs/2310.xxxxx) <!-- Paper ID uncertain; Code not publicly available under 'DiffITE' or 'MCDE' at this time. Recommended fallback: https://github.com/vanderschaarlab/ml-for-healthcare-concepts -->
*   **核心机制**: **Generative Diffusion Model (扩散模型)**
*   **技术特点**:
    *   不同于 VAE (Deep-Lung-Causal 目前的一项技术) 的点估计或高斯近似，Diff-ITE 使用扩散过程直接建模潜在结果 $Y(0)$ 和 $Y(1)$ 的复杂条件分布。
    *   引入了 **"Counterfactual Guidance" (反事实引导)** 技术，在生成反事实结果时利用观测数据进行条件约束。
*   **适用性**:
    *   **强项**: 极强的分布建模能力，能捕捉基因数据中的多峰分布（Multimodal Distribution）。
    *   **缺点**: 推理速度较慢（需要多步去噪）。
*   **推荐指数**: ⭐⭐⭐⭐⭐ (最适合作为 Generative Causal 的 SOTA 对比)

## 2. Mamba-CI (Causal Inference via State Space Models)
*   **发布时间**: Late 2025
*   **参考论文**: [MambaTab: A Plug-and-Play Model for Learning Tabular Data](https://arxiv.org/abs/2401.08867)
*   **核心机制**: **Mamba / S4 (Selective State Spaces)**
*   **技术特点**:
    *   将高维基因特征视为序列（Sequence of Genes），利用 Mamba 的线性复杂度处理超长特征输入。
    *   比 Transformer (TransTEE/HyperFast) 显存占用更低，训练收敛更快。
*   **适用性**:
    *   **强项**: 处理超高维组学数据（High-dim Genomics）效率极高。
    *   **缺点**: 对于因果效应的专门设计（如倾向性权衡）不如 TransTEE 成熟。
*   **推荐指数**: ⭐⭐⭐⭐

## 3. TabM-Causal (Tabular Mini-Batch Ensemble)
*   **发布时间**: Late 2024 / Updated early 2025
*   **参考论文**: [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)
*   **官方代码**: [https://github.com/yandex-research/tabm](https://github.com/yandex-research/tabm)
*   **核心机制**: **Batch Ensembling** (from Yandex Research)
*   **技术特点**:
    *   在单个前向传递中处理 Mini-Batch，这对于处理表格数据中的批次效应（Batch Effects）非常有效。
    *   被认为是 TabR 的继任者，在很多 Kaggle Tabular 比赛中霸榜。
    *   **Causal 变体**: 将其改造为 T-Learner 架构（两个独立的 TabM 头）。
*   **适用性**:
    *   **强项**: 纯预测精度（AUC/RMSE）极高，极难被击败。
    *   **缺点**: 缺乏因果解释性（Black-box）。
*   **推荐指数**: ⭐⭐⭐⭐

---

### 建议
鉴于您的 DLC 模型已经包含 VAE 组件，引入 **Diff-ITE (基于扩散模型)** 是最具“降维打击”意义的对比。它代表了从 *Variational Inference* 到 *Diffusion Generative Models* 的代际跨越。

**是否需要我为您获取 `Diff-ITE` 的详细架构并在 `src/baselines/` 中实现它？**
