# SemEval 2026 Task 12: Abductive Event Reasoning (AER)

## 任务简介

**溯因事件推理 (Abductive Event Reasoning)** 是 SemEval 2026 的共享任务，旨在评估大语言模型推理真实世界事件因果关系的能力。

### 核心任务
给定一个事件（如"加密货币价格飙升"）和相关文档，模型需要从候选选项中识别**最可能的直接原因**。

### 评估指标
- **1.0分**: 完全匹配（预测 = 标准答案）
- **0.5分**: 部分匹配（预测是标准答案的真子集）
- **0.0分**: 错误

---

## Baseline 1: LLM API 调用

本仓库提供了完整的baseline实验框架，支持多种模型。

```bash
cd baseline
python run_baseline.py --model-type openai --model-name gpt-4o-mini --data-split dev
```

支持: OpenAI, Anthropic, HuggingFace, Ollama, vLLM

---

## Baseline 2: 经典研究方法 (UnifiedQA / RoBERTa)

基于现有NLP研究的baseline实现。

参考论文:
- [UnifiedQA](https://arxiv.org/abs/2005.00700) (EMNLP 2020)
- [RoBERTa](https://arxiv.org/abs/1907.11692)

```bash
cd baseline2

# 数据预处理
python run_baseline2.py preprocess --dataset-dir ../data/semeval2026-task12-dataset

# UnifiedQA (零样本)
python run_baseline2.py unifiedqa --data-path ./processed_data/dev/unifiedqa.jsonl

# RoBERTa (需要微调)
python run_baseline2.py roberta --mode train --train-data ./processed_data/train/roberta_mcqa.jsonl
```

---

## Baseline 3: 知识图谱增强 (KG + LLM) ⭐ NEW

将**知识图谱嵌入**与**大语言模型**结合进行因果推理。

### 参考论文
- [TransE](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) - 知识图谱嵌入 (NeurIPS 2013)
- [ComplEx](https://arxiv.org/abs/1606.06357) - 复数空间嵌入 (ICML 2016)
- [RotatE](https://arxiv.org/abs/1902.10197) - 旋转嵌入 (ICLR 2019)
- [COMET-ATOMIC](https://arxiv.org/abs/2010.05953) - 常识知识生成 (EMNLP 2020)
- [QA-GNN](https://arxiv.org/abs/2104.06378) - KG+LLM QA (NAACL 2021)

### 核心组件

| 模块 | 说明 |
|------|------|
| `kg_embedding.py` | KG嵌入模型 (TransE, ComplEx, RotatE) |
| `comet_knowledge.py` | COMET知识生成 |
| `kg_llm_qa.py` | KG-LLM融合QA |

### 使用方法

```bash
cd baseline3

# 1. 构建知识图谱 + 训练嵌入
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output \
    --train-embedding \
    --kg-model TransE

# 2. 运行KG增强QA
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini
```

### KG Embedding 详解

详见 [`baseline3/KG_EMBEDDING_DOC.md`](baseline3/KG_EMBEDDING_DOC.md)，包含:

- TransE/ComplEx/RotatE 原理与实现
- 训练流程与超参数选择
- 嵌入在QA任务中的应用

### 融合方法

| 方法 | 说明 | 命令 |
|------|------|------|
| `prompt` | 将KG知识添加到prompt | `--fusion prompt` |
| `retrieval` | 从KG检索相关三元组 | `--fusion retrieval` |

---

## 研究方向建议

### 1. Prompt Engineering
- Chain-of-Thought (CoT) 提示
- Few-shot 示例选择策略
- 自我一致性 (Self-Consistency)

### 2. 检索增强 (RAG)
- 文档重排序 (Re-ranking)
- 关键信息抽取
- 多文档融合

### 3. 知识图谱增强
- 因果知识图谱构建
- KG嵌入学习 (TransE, ComplEx, RotatE)
- COMET常识推理

### 4. 微调方法
- LoRA/QLoRA 微调
- 指令微调 (Instruction Tuning)
- 强化学习 (RLHF)

---

## 文件结构

```
.
├── baseline/                      # Baseline 1: LLM API
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── models.py
│   ├── run_baseline.py
│   └── AER_Baseline_Colab.ipynb
│
├── baseline2/                     # Baseline 2: UnifiedQA/RoBERTa
│   ├── preprocessing.py
│   ├── unifiedqa_baseline.py
│   ├── roberta_mcqa_baseline.py
│   ├── run_baseline2.py
│   └── AER_Baseline2_Colab.ipynb
│
├── baseline3/                     # Baseline 3: KG + LLM ⭐
│   ├── kg_embedding.py            # TransE/ComplEx/RotatE
│   ├── comet_knowledge.py         # COMET知识生成
│   ├── kg_llm_qa.py               # KG-LLM融合
│   ├── run_baseline3.py           # 统一运行脚本
│   └── KG_EMBEDDING_DOC.md        # 技术文档
│
└── README.md
```

---

## Colab 快速体验

| Notebook | 说明 |
|----------|------|
| `baseline/AER_Baseline_Colab.ipynb` | LLM API调用 |
| `baseline2/AER_Baseline2_Colab.ipynb` | UnifiedQA + RoBERTa |

---

## 参考资源

### 数据集 & 竞赛
- [官方数据集](https://github.com/sooo66/semeval2026-task12-dataset)
- [竞赛页面 (Codabench)](https://www.codabench.org/competitions/12440/)
- [SemEval 2026](https://semeval.github.io/SemEval2026/)

### 相关研究
- [Awesome-LLM-Causal-Reasoning](https://github.com/chendl02/Awesome-LLM-causal-reasoning) - 因果推理论文集
- [KG-LLM-Papers](https://github.com/zjukg/KG-LLM-Papers) - KG+LLM论文集
- [OpenKE](https://github.com/thunlp/OpenKE) - 知识图谱嵌入工具包
- [TorchKGE](https://github.com/torchkge-team/torchkge) - PyTorch KG嵌入库
- [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020) - 常识知识生成
- [UnifiedQA](https://github.com/allenai/unifiedqa) - AllenAI统一问答模型
