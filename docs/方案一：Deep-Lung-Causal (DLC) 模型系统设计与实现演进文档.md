# **方案一：Deep-Lung-Causal (DLC) 模型系统设计与实现演进文档**

版本： V4.3 (路径修正版)  
适用对象： 编程助手 (Kiro) / 项目开发人员  
任务目标： 采用“双源并行”策略，分别处理 PANCAN (泛癌) 和 LUAD (肺腺癌) 两套原始数据，构建两套独立的半合成验证数据集。

## **第一章：数据源定义与读取规范 (Data Sources)**

本项目涉及两套独立的数据源，需编写通用清洗逻辑分别进行处理。

### **🚨 核心原则：ID 对齐标准 (Critical)**

**所有数据合并必须基于 “长格式样本 ID” (例如 TCGA-05-4244-01)。**

* **禁止使用** 短格式病人 ID (如 TCGA-05-4244)。  
* **理由：** 同一个病人可能包含正常组织样本 (如 \-11)，混用会导致将癌症标签错误匹配给正常基因组数据，造成严重的标签泄漏。

### **1.1 数据源 A：泛癌数据集 (用于预训练)**

* **存放路径:** data/PANCAN/  
* **包含文件:**  
  1. **基因文件:** PANCAN\_mutation.txt  
     * **用途:** 获取 $X\_{gene}$ (基因突变特征)。  
     * **操作指令:**  
       * **ID确认:** 该文件的**列名 (Columns)** 即为长格式样本 ID，**行名 (Index)** 为基因 Symbol。  
       * **转置 (Transpose):** **必须转置** (df.T)，变为“行=样本，列=基因”。  
  2. **临床文件:** PANCAN\_clinical.txt  
     * **用途:** 获取 $X\_{conf}$ (Age, Gender)。  
     * **关键字段映射 (必须精确匹配):**  
       * sample: **主键** (使用此列作为长格式 ID)  
       * age\_at\_initial\_pathologic\_diagnosis $\\rightarrow$ 重命名为 Age  
       * gender $\\rightarrow$ 重命名为 Gender  
* **合并逻辑:** 基于 **sample** 列进行 Inner Join。

### **1.2 数据源 B：肺腺癌数据集 (用于微调与评估)**

* **存放路径:** data/LUAD/  
* **包含文件:**  
  1. **基因文件:** LUAD\_mc3\_gene\_level.txt  
     * **用途:** 获取 $X\_{gene}$。  
     * **操作指令:**  
       * **ID确认:** 列名即为样本 ID。  
       * **转置 (Transpose):** **必须转置** (df.T)，变为“行=样本，列=基因”。  
       * **索引对齐:** 转置后，将行索引重命名为 sampleID。  
  2. **临床文件:** TCGA.LUAD.sampleMap\_LUAD\_clinicalMatrix.txt  
     * **用途:** 获取 $X\_{conf}$。  
     * **关键字段映射:**  
       * sampleID: **主键** (严禁使用 \_PATIENT)  
       * age\_at\_initial\_pathologic\_diagnosis $\\rightarrow$ 重命名为 Age  
       * gender $\\rightarrow$ 重命名为 Gender  
  3. **生存文件:** survival\_LUAD\_survival.txt  
     * **用途:** 仅用于样本筛选（确保样本质量），**不读取**生存时间。  
     * **操作指令:** 仅读取 sample 列 (长格式)，重命名为 sampleID，忽略 \_PATIENT 列。  
* **合并逻辑:** 基于 **sampleID** 列进行 Inner Join。

### **1.3 通用清洗逻辑 (Common Cleaning Rules)**

无论处理哪个数据源，必须执行以下统一标准：

#### **1.3.1 变量排除标准 (Exclusion Criteria)**

在代码注释中必须注明以下变量的**排除原因**：

* **排除所有非核心变量:** 分期 (Stage)、病理 (Pathology)、种族 (Race)、吸烟史等均不读取。  
* **理论依据:** 依据因果图 (DAG) 理论，Stage 和 Pathology 为中介变量 (Mediator)，控制它们会导致**过度调整偏倚 (Over-adjustment Bias)**。

#### **1.3.2 具体处理步骤**

1. **数据合并 (Merge):**  
   * 统一使用 **长格式 ID** 作为主键取交集 (Inner Join)。  
   * *Check point:* 打印合并后的样本量。  
2. **基因特征筛选 (**$X\_{gene}$**):**  
   * 筛选 **Top 20 高频突变基因**。  
   * **强制包含检查:** 检查 Top 20 中是否包含 EGFR, KRAS, TP53。若未包含，必须强制替换频次最低的基因加入。  
   * **数值化:** 确保数据为 0 (无突变) / 1 (有突变)。  
3. **临床特征清洗 (**$X\_{conf}$**):**  
   * **Age:** 转换为数值型，填补缺失值 (均值填充)。  
   * **Gender:** 转换为二值型 (Male=0, Female=1)。

## **第二章：半合成数据生成 (Semi-synthetic Generation)**

编写一个通用的生成器类 SemiSyntheticGenerator，它可以接受任意一个清洗后的 DataFrame，并生成虚拟列。

### **2.1 任务定义**

生成两套数据集，分别对应不同的流行病学假设场景。

* **输入:** 清洗后的基因 \+ Age/Gender 数据。  
* **输出:** Virtual\_PM2.5 (环境) \+ Outcome\_Label (结局)。

### **2.2 生成规则与公式**

#### **步骤 A: 生成虚拟环境暴露 Virtual\_PM2.5 ($X\_{env}$)**

* **目的:** 模拟“混杂效应”，即年龄影响居住环境。  
* **生成公式:**  
  np.random.seed(42)  
  \# 均值30，标准差10，叠加年龄带来的偏移（模拟混杂）  
  \# 注意：需先对 Age 进行 z-score 标准化  
  df\['Virtual\_PM2.5'\] \= 30 \+ 0.5 \* z\_score(df\['Age'\]) \+ np.random.normal(0, 10, len(df))

* **验证点:** DLC 模型必须能切断 Age 和 PM2.5 的关联。

#### **步骤 B: 生成结局标签 ($Y\_{syn}$) —— 核心步骤**

计算每个样本的 **Logit (对数几率, 记为** $L$**)**。

**名词解释：L (Logit)**

* $L$ 是线性预测算子 (Linear Predictor)。  
* 它代表了在转化为概率之前的\*\*“原始风险评分”\*\*，取值范围为 $(-\\infty, \+\\infty)$。  
* $L$ 越大，患癌概率越高。

**核心变量定义:**

* **EGFR:** 二值变量 (0/1)。  
* **KRAS:** 二值变量 (0/1)。

**核心权重参数 (基于文献):**

* $w\_{base} \\approx 0.086$**:** 基准环境权重 (Hamra et al., 2014\)  
* $w\_{int} \\approx 0.69$**:** 交互环境权重 (Hill et al., 2023, 模拟 Nature 机制)  
* $w\_{gene} \= 0.5$**:** 基因主效应

**场景 1：交互机制 (Scenario='interaction') —— 验证集**

* **假设:** PM2.5 显著促进 EGFR 突变者的癌症风险，对 KRAS 无效。  

* 公式:  
  $$L \= \-3.0 \+ 0.086 \\cdot \\text{PM2.5}^\* \+ \\mathbf{0.69 \\cdot (\\text{PM2.5}^\* \\times 1 \\text{ if EGFR else } 0)} \+ 0.0 \\cdot (\\text{PM2.5}^\* \\times 1 \\text{ if KRAS else } 0\) \+ \\text{Genetics}$$

  (注：$\\text{PM2.5}^$ 为标准化后的 PM2.5，Genetics 为 Top20 基因加权和)\*

**场景 2：线性机制 (Scenario='linear') —— 对照集**

* **假设:** PM2.5 对所有人的风险贡献一致。  
* 公式:  
  $$L \= \-3.0 \+ 0.086 \\cdot \\text{PM2.5}^\* \+ \\text{Genetics}$$

通用项 Genetics 计算:

$$\\text{Genetics} \= 0.5 \\cdot \\sum\_{i=1}^{20} \\text{Gene}\_i$$

(即：每个 Top20 基因若突变为 1，未突变为 0；将这 20 个值相加后乘以 0.5)

#### **步骤 C: 概率转化与采样**

1. **Sigmoid:** $P \= 1 / (1 \+ \\exp(-L))$  
2. **Bernoulli Sampling:** $Y \\sim \\text{Bernoulli}(P)$

## **第三章：输出产物规范**

代码运行结束后，需输出 **4 个 CSV 文件**。

### **3.1 泛癌产物 (基于数据源 A)**

* **用途:** 用于模型预训练 (Pre-training)。  
* **文件名:**  
  * data/pancan\_synthetic\_interaction.csv  
  * data/pancan\_synthetic\_linear.csv

### **3.2 肺癌产物 (基于数据源 B)**

* **用途:** 用于模型微调 (Fine-tuning) 与最终评估。  
* **文件名:**  
  * data/luad\_synthetic\_interaction.csv  
  * data/luad\_synthetic\_linear.csv

**文件字段列表 (Columns):**

* sampleID (长格式)  
* Age, Gender  
* EGFR, KRAS, TP53, ... (Top 20 基因)  
* **Virtual\_PM2.5**  
* **True\_Prob** (生成的真实概率 $P$，用于调试)  
* **Outcome\_Label** (生成的二分类标签 $Y$)

## **第四章：工作过程记录要求**

请在代码执行完毕后，向 docs/工作过程.md 追加如下记录：

### **\[YYYY-MM-DD\] 双源数据清洗与生成**

1. **PANCAN 处理:** 读取 PANCAN 源，清洗并生成 \[N\] 条预训练数据。  
2. **LUAD 处理:** 读取 LUAD 源，清洗并生成 \[N\] 条微调数据。  
3. **一致性检查:** 确认两套数据的列名 (Columns) 完全一致，确保模型可迁移。  
4. **半合成分布验证:**  
   * Age 与 PM2.5 相关系数: \[r\]  
   * 交互场景下 EGFR 突变组的阳性率: \[X\]%  
5. **产出文件:** 4 个 CSV 文件已保存。