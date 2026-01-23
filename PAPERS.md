# Baseline 论文解读

本文档整理了三个baseline方法对应的核心论文，包括论文链接、核心思想、方法细节和与AER任务的关联。

---

## 目录

- [Baseline 1: LLM Prompting](#baseline-1-llm-prompting)
- [Baseline 2: UnifiedQA + RoBERTa](#baseline-2-unifiedqa--roberta)
- [Baseline 3: KG Embedding + LLM](#baseline-3-kg-embedding--llm)
- [综合对比](#综合对比)

---

## Baseline 1: LLM Prompting

### 核心论文

#### 1.1 GPT-4 Technical Report

| 项目 | 内容 |
|------|------|
| **标题** | GPT-4 Technical Report |
| **作者** | OpenAI |
| **发表** | arXiv 2023 |
| **链接** | https://arxiv.org/abs/2303.08774 |

**摘要解读**:
GPT-4是一个大规模多模态模型，在各种专业和学术基准测试中展现出人类水平的性能。论文展示了GPT-4在推理任务上的强大能力，包括因果推理、逻辑推理等。

**核心贡献**:
1. 展示了规模扩展对推理能力的提升
2. 引入了多模态能力
3. 在复杂推理任务上达到SOTA

**与AER任务的关联**:
- GPT-4的零样本推理能力可直接用于因果推理
- 通过精心设计的prompt可以引导模型进行溯因推理
- 上下文学习(In-Context Learning)可提供少样本示例

---

#### 1.2 Chain-of-Thought Prompting

| 项目 | 内容 |
|------|------|
| **标题** | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models |
| **作者** | Jason Wei, Xuezhi Wang, et al. (Google) |
| **发表** | NeurIPS 2022 |
| **链接** | https://arxiv.org/abs/2201.11903 |

**摘要解读**:
该论文提出了Chain-of-Thought (CoT)提示方法，通过在prompt中加入推理步骤的示例，显著提升LLM在复杂推理任务上的表现。

**核心方法**:
```
标准Prompt:
Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?
A: 7

CoT Prompt:
Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?
A: Roger started with 5 balls. 2 more makes 5 + 2 = 7. The answer is 7.
```

**关键发现**:
1. CoT在算术、常识、符号推理上都有显著提升
2. 模型规模越大，CoT效果越明显（涌现能力）
3. 推理链的质量比数量更重要

**与AER任务的应用**:
```
标准Prompt:
Event: Stock market crashes
Options: A) Economic recession B) Good weather C) ...
Answer: A

CoT Prompt:
Event: Stock market crashes
Let me think step by step:
1. Stock market crashes are usually caused by economic factors
2. Economic recession leads to reduced corporate profits
3. Reduced profits cause stock prices to fall
4. Therefore, A (Economic recession) is the most direct cause
Answer: A
```

---

#### 1.3 Self-Consistency

| 项目 | 内容 |
|------|------|
| **标题** | Self-Consistency Improves Chain of Thought Reasoning in Language Models |
| **作者** | Xuezhi Wang, Jason Wei, et al. (Google) |
| **发表** | ICLR 2023 |
| **链接** | https://arxiv.org/abs/2203.11171 |

**核心思想**:
对同一问题生成多个推理路径，然后通过投票选择最一致的答案。

**方法流程**:
```
问题 → [路径1: A]
     → [路径2: A]  → 投票 → 最终答案: A
     → [路径3: B]
```

**与AER任务的应用**:
- 生成多个因果推理链
- 对可能的原因进行投票
- 提高预测的稳定性

---

## Baseline 2: UnifiedQA + RoBERTa

### 核心论文

#### 2.1 UnifiedQA

| 项目 | 内容 |
|------|------|
| **标题** | UnifiedQA: Crossing Format Boundaries with a Single QA System |
| **作者** | Daniel Khashabi, Sewon Min, et al. (AllenAI) |
| **发表** | EMNLP 2020 (Findings) |
| **链接** | https://arxiv.org/abs/2005.00700 |
| **代码** | https://github.com/allenai/unifiedqa |

**摘要解读**:
UnifiedQA提出了一个统一的问答框架，可以处理多种QA格式（抽取式、多选题、是否题等），而无需针对每种格式单独训练。

**核心创新**:

1. **统一输入格式**:
```
抽取式: context \n question
多选题: question \n (A) opt1 (B) opt2 (C) opt3 (D) opt4
是否题: question \n yes or no
```

2. **Text-to-Text框架**: 基于T5，将所有QA任务统一为文本生成任务

3. **多任务学习**: 在多种QA数据集上联合训练

**模型架构**:
```
输入: "What causes rain? \n (A) Sun (B) Clouds (C) Wind (D) Moon"
      ↓
   [T5 Encoder]
      ↓
   [T5 Decoder]
      ↓
输出: "B"
```

**训练数据集**:
- SQuAD (抽取式)
- NarrativeQA (抽取式)
- RACE (多选题)
- ARC (多选题)
- BoolQ (是否题)
- 等20+数据集

**实验结果** (原论文):

| 数据集 | UnifiedQA-base | UnifiedQA-large |
|--------|---------------|-----------------|
| ARC-Easy | 68.9 | 81.4 |
| ARC-Challenge | 47.4 | 54.9 |
| OBQA | 60.4 | 67.0 |
| RACE | 64.5 | 73.2 |

**与AER任务的应用**:
- 直接将AER转换为多选题格式
- 零样本推理，无需额外训练
- 利用预训练中学到的常识知识

**代码示例**:
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-base")
model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

input_text = "What causes stock market crash? \\n (A) Economic recession (B) Good weather"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0])  # "A"
```

---

#### 2.2 RoBERTa

| 项目 | 内容 |
|------|------|
| **标题** | RoBERTa: A Robustly Optimized BERT Pretraining Approach |
| **作者** | Yinhan Liu, Myle Ott, et al. (Facebook AI) |
| **发表** | arXiv 2019 |
| **链接** | https://arxiv.org/abs/1907.11692 |
| **代码** | https://github.com/facebookresearch/fairseq |

**摘要解读**:
RoBERTa通过改进BERT的预训练策略，在不改变模型架构的情况下显著提升了性能。

**核心改进** (相比BERT):

| 改进点 | BERT | RoBERTa |
|--------|------|---------|
| 训练数据 | 16GB | 160GB |
| Batch Size | 256 | 8K |
| 训练步数 | 1M | 500K |
| NSP任务 | 有 | **移除** |
| 动态Masking | 静态 | **动态** |

**动态Masking解释**:
- BERT: 预处理时确定mask位置，训练时固定
- RoBERTa: 每个epoch重新随机mask

**多选题分类头**:
```
输入: [CLS] question [SEP] option_i [SEP]
      ↓
   [RoBERTa Encoder]
      ↓
   [CLS] embedding → Linear → score_i

对4个选项分别计算score，softmax得到概率
```

**实验结果** (原论文):

| 任务 | BERT-large | RoBERTa-large |
|------|-----------|---------------|
| MNLI | 86.6 | 90.2 |
| QNLI | 92.3 | 94.7 |
| SST-2 | 93.2 | 96.4 |
| RACE | 72.0 | 83.2 |

**与AER任务的应用**:
- 将因果推理转化为多选题分类
- 在训练集上微调
- 利用预训练的语言理解能力

---

#### 2.3 DeBERTa (推荐替代)

| 项目 | 内容 |
|------|------|
| **标题** | DeBERTa: Decoding-enhanced BERT with Disentangled Attention |
| **作者** | Pengcheng He, et al. (Microsoft) |
| **发表** | ICLR 2021 |
| **链接** | https://arxiv.org/abs/2006.03654 |

**核心创新**:
1. **Disentangled Attention**: 将内容和位置分开编码
2. **Enhanced Mask Decoder**: 改进的mask预测
3. **Virtual Adversarial Training**: 虚拟对抗训练提升泛化

**为什么推荐DeBERTa**:
- 在多选题任务上通常优于RoBERTa
- 参数效率更高
- SuperGLUE榜单上表现优异

---

## Baseline 3: KG Embedding + LLM

### 核心论文

#### 3.1 TransE

| 项目 | 内容 |
|------|------|
| **标题** | Translating Embeddings for Modeling Multi-relational Data |
| **作者** | Antoine Bordes, et al. (Facebook AI) |
| **发表** | NeurIPS 2013 |
| **链接** | https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html |

**核心思想**:
将关系建模为向量空间中的**平移操作**：
```
h + r ≈ t
```
其中 h 是头实体嵌入，r 是关系嵌入，t 是尾实体嵌入。

**直观理解**:
```
"北京" + "是...的首都" ≈ "中国"
"巴黎" + "是...的首都" ≈ "法国"
```

**损失函数** (Margin Ranking Loss):
```
L = Σ max(0, γ + d(h+r, t) - d(h'+r, t'))
```
- γ: margin超参数
- d: 距离函数 (L1或L2)
- (h, r, t): 正样本
- (h', r, t'): 负样本 (随机替换头或尾)

**训练算法**:
```python
for epoch in range(num_epochs):
    for (h, r, t) in positive_triples:
        # 生成负样本
        h_neg = random_entity()  # 或 t_neg = random_entity()

        # 计算损失
        pos_score = ||h + r - t||
        neg_score = ||h_neg + r - t||
        loss = max(0, margin + pos_score - neg_score)

        # 更新
        loss.backward()
        optimizer.step()

        # 归一化实体嵌入
        normalize(entity_embeddings)
```

**局限性**:
- 无法建模对称关系: r ≠ 0 时，h + r ≠ t + r
- 无法建模1-N, N-1, N-N关系

**与AER任务的应用**:
```
三元组: (经济衰退, causes, 失业率上升)
嵌入:  e_recession + r_causes ≈ e_unemployment
```

---

#### 3.2 ComplEx

| 项目 | 内容 |
|------|------|
| **标题** | Complex Embeddings for Simple Link Prediction |
| **作者** | Théo Trouillon, et al. (Facebook AI) |
| **发表** | ICML 2016 |
| **链接** | https://arxiv.org/abs/1606.06357 |

**核心思想**:
使用**复数空间**来建模实体和关系，可以自然地处理非对称关系。

**数学表示**:
```
实体: e = e_re + i * e_im  (复数)
关系: r = r_re + i * r_im  (复数)
```

**得分函数** (Hermitian点积):
```
score(h, r, t) = Re(⟨h, r, t̄⟩)
              = Re(Σ h_i * r_i * conj(t_i))
              = h_re·r_re·t_re + h_re·r_im·t_im + h_im·r_re·t_im - h_im·r_im·t_re
```

**为什么复数有效**:
- 对称关系: 当 r_im = 0 时，score(h,r,t) = score(t,r,h)
- 非对称关系: 当 r_im ≠ 0 时，score(h,r,t) ≠ score(t,r,h)

**实验结果** (原论文):

| 数据集 | TransE | DistMult | ComplEx |
|--------|--------|----------|---------|
| FB15k (Hits@10) | 47.1 | 57.7 | **84.0** |
| WN18 (Hits@10) | 89.2 | 93.6 | **94.7** |

**与AER任务的应用**:
因果关系是非对称的（A导致B ≠ B导致A），ComplEx可以更好地建模这种关系。

---

#### 3.3 RotatE

| 项目 | 内容 |
|------|------|
| **标题** | RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space |
| **作者** | Zhiqing Sun, et al. (PKU) |
| **发表** | ICLR 2019 |
| **链接** | https://arxiv.org/abs/1902.10197 |
| **代码** | https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding |

**核心思想**:
将关系建模为复平面上的**旋转**操作：
```
t = h ◦ r
```
其中 ◦ 是复数乘法（Hadamard积），r 被约束为单位复数。

**数学表示**:
```
关系: r = e^(iθ) = cos(θ) + i*sin(θ)  (单位复数)
旋转: t = h ◦ r = |h| * e^(i(θ_h + θ_r))
```

**可建模的关系模式**:

| 模式 | 条件 | 示例 |
|------|------|------|
| 对称 | θ = 0 或 π | "相似于" |
| 反对称 | θ ≠ 0, π | "父亲是" |
| 逆 | θ_r1 = -θ_r2 | "父亲" vs "孩子" |
| 组合 | θ_r3 = θ_r1 + θ_r2 | "祖父 = 父亲 + 父亲" |

**得分函数**:
```
score(h, r, t) = -||h ◦ r - t||
```

**Self-Adversarial负采样**:
```python
# 根据当前模型给负样本赋予权重
weights = softmax(score(neg_samples))
loss = -log(σ(γ - d_pos)) - Σ w_i * log(σ(d_neg_i - γ))
```

**实验结果** (原论文):

| 数据集 | TransE | ComplEx | RotatE |
|--------|--------|---------|--------|
| FB15k-237 (MRR) | .294 | .247 | **.338** |
| WN18RR (MRR) | .226 | .440 | **.476** |

**与AER任务的应用**:
- 因果链可以建模为组合关系
- "A导致B，B导致C" → "A间接导致C"

---

#### 3.4 COMET-ATOMIC

| 项目 | 内容 |
|------|------|
| **标题** | COMET-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs |
| **作者** | Jena D. Hwang, et al. (AllenAI) |
| **发表** | AAAI 2021 |
| **链接** | https://arxiv.org/abs/2010.05953 |
| **代码** | https://github.com/allenai/comet-atomic-2020 |

**摘要解读**:
ATOMIC 2020是一个大规模常识知识图谱，COMET是在其上训练的神经知识生成模型，可以对任意事件生成结构化的常识推理。

**ATOMIC 2020知识图谱**:
- 1.33M 三元组
- 23种关系类型
- 覆盖社会、物理、事件常识

**关系类型**:

| 类型 | 关系 | 示例 |
|------|------|------|
| 因果 | Causes | "下雨" Causes "地面湿" |
| 因果 | xEffect | "吃饭" xEffect "不饿了" |
| 时序 | isBefore | "买票" isBefore "看电影" |
| 时序 | isAfter | "毕业" isAfter "找工作" |
| 意图 | xIntent | "学习" xIntent "获得知识" |
| 需求 | xNeed | "做饭" xNeed "有食材" |
| 反应 | xReact | "中奖" xReact "开心" |

**COMET模型**:
```
输入: "PersonX goes to school" + "xEffect"
      ↓
   [GPT-2 / BART]
      ↓
输出: "PersonX learns new things"
```

**生成示例**:
```python
event = "The stock market crashes"
comet.generate(event, "Causes")
# → ["Economic recession", "Trade war", "Interest rate hike"]

comet.generate(event, "xEffect")
# → ["Investors lose money", "Companies lay off workers"]
```

**与AER任务的应用**:
1. 为目标事件生成可能的原因
2. 为候选选项生成可能的影响
3. 通过比较因果链判断最可能的原因

---

#### 3.5 QA-GNN (参考方法)

| 项目 | 内容 |
|------|------|
| **标题** | QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering |
| **作者** | Michihiro Yasunaga, et al. (Stanford) |
| **发表** | NAACL 2021 |
| **链接** | https://arxiv.org/abs/2104.06378 |
| **代码** | https://github.com/michiyasunaga/qagnn |

**核心思想**:
将语言模型和知识图谱通过图神经网络进行联合推理。

**架构**:
```
问题+选项 → [LM Encoder] → 文本表示
              ↓
KG子图 → [GNN] → 图表示
              ↓
         [融合层]
              ↓
          答案预测
```

**方法细节**:
1. **子图提取**: 从ConceptNet提取与问题相关的子图
2. **节点初始化**: 用LM嵌入初始化KG节点
3. **消息传递**: GNN在子图上传播信息
4. **联合推理**: 文本和图表示融合后预测

**与AER任务的应用**:
- 构建因果关系子图
- 使用GNN进行因果推理
- 融合文档上下文和KG知识

---

## 综合对比

### 方法对比

| 方面 | Baseline 1 | Baseline 2 | Baseline 3 |
|------|------------|------------|------------|
| **方法** | LLM Prompting | Fine-tuned Models | KG + LLM |
| **是否需要训练** | 否 | UnifiedQA否, RoBERTa是 | KG嵌入是 |
| **知识来源** | LLM内部知识 | 预训练知识 | 外部KG |
| **推理方式** | 生成式 | 生成/判别 | 结构化+生成 |
| **可解释性** | 中(CoT) | 低 | 高 |
| **计算成本** | API费用 | GPU训练 | GPU训练 |

### 论文引用

```bibtex
% Baseline 1
@article{openai2023gpt4,
  title={GPT-4 Technical Report},
  author={OpenAI},
  journal={arXiv preprint arXiv:2303.08774},
  year={2023}
}

@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and others},
  journal={NeurIPS},
  year={2022}
}

% Baseline 2
@article{khashabi2020unifiedqa,
  title={UnifiedQA: Crossing Format Boundaries with a Single QA System},
  author={Khashabi, Daniel and Min, Sewon and others},
  journal={EMNLP Findings},
  year={2020}
}

@article{liu2019roberta,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Liu, Yinhan and Ott, Myle and others},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

% Baseline 3
@inproceedings{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and others},
  booktitle={NeurIPS},
  year={2013}
}

@inproceedings{trouillon2016complex,
  title={Complex embeddings for simple link prediction},
  author={Trouillon, Th{\'e}o and others},
  booktitle={ICML},
  year={2016}
}

@inproceedings{sun2019rotate,
  title={RotatE: Knowledge graph embedding by relational rotation in complex space},
  author={Sun, Zhiqing and others},
  booktitle={ICLR},
  year={2019}
}

@article{hwang2021comet,
  title={COMET-ATOMIC 2020: On symbolic and neural commonsense knowledge graphs},
  author={Hwang, Jena D and others},
  journal={AAAI},
  year={2021}
}
```

---

## 扩展阅读

### 因果推理相关
- [Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond](https://arxiv.org/abs/2109.00725) - ACL 2022 Tutorial
- [CausalBERT: Injecting Causal Knowledge Into Pre-trained Models](https://arxiv.org/abs/2107.00320)

### LLM推理能力
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - "Let's think step by step"
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)

### KG+LLM融合
- [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302) - 综述
- [KG-LLM Papers](https://github.com/zjukg/KG-LLM-Papers) - 论文列表
