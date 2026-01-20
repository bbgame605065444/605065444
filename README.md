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

### 快速开始

```bash
# 1. 克隆数据集
git clone https://github.com/sooo66/semeval2026-task12-dataset.git

# 2. 安装依赖
pip install -r baseline/requirements.txt

# 3. 运行baseline
cd baseline
python run_baseline.py --model-type openai --model-name gpt-4o-mini --data-split dev
```

### 支持的模型

| 模型类型 | 示例 | 说明 |
|---------|------|------|
| `openai` | gpt-4o, gpt-4o-mini | 需要 OPENAI_API_KEY |
| `anthropic` | claude-3-5-sonnet-20241022 | 需要 ANTHROPIC_API_KEY |
| `huggingface` | Qwen/Qwen2.5-7B-Instruct | 本地GPU运行 |
| `ollama` | llama3.1:8b | 本地Ollama服务 |
| `vllm` | meta-llama/Llama-3.1-8B-Instruct | 高性能批量推理 |

---

## Baseline 2: 经典研究方法 (UnifiedQA / RoBERTa)

基于现有NLP研究的baseline实现，包含完整的数据预处理流程。

### 参考论文
- [UnifiedQA: Crossing Format Boundaries with a Single QA System](https://arxiv.org/abs/2005.00700) (EMNLP 2020)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

### 数据预处理

```bash
cd baseline2

# 预处理数据（生成UnifiedQA和RoBERTa兼容格式）
python run_baseline2.py preprocess \
    --dataset-dir ../data/semeval2026-task12-dataset \
    --output-dir ./processed_data
```

预处理后生成:
```
processed_data/
├── train/
│   ├── unifiedqa.jsonl      # UnifiedQA格式
│   ├── roberta_mcqa.jsonl   # RoBERTa MCQA格式
│   └── data.json            # HuggingFace格式
├── dev/
└── test/
```

### 方法 1: UnifiedQA (零样本)

```bash
# 使用UnifiedQA-base
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-t5-base

# 使用UnifiedQA-v2 (更强)
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-v2-t5-large-1363200
```

可用模型:
| 模型 | 参数量 | 显存需求 |
|------|-------|---------|
| `allenai/unifiedqa-t5-small` | 60M | ~1GB |
| `allenai/unifiedqa-t5-base` | 220M | ~2GB |
| `allenai/unifiedqa-t5-large` | 770M | ~4GB |
| `allenai/unifiedqa-v2-t5-base-1363200` | 220M | ~2GB |

### 方法 2: RoBERTa/DeBERTa (需要微调)

```bash
# 训练
python run_baseline2.py roberta --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name roberta-base \
    --output-dir ./roberta_output

# 预测
python run_baseline2.py roberta --mode predict \
    --data-path ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name ./roberta_output
```

推荐模型:
| 模型 | 说明 |
|------|------|
| `roberta-base` | 通用baseline |
| `roberta-large` | 更大容量 |
| `microsoft/deberta-v3-base` | 推荐，效果更好 |

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

### 3. 推理增强
- 因果图构建
- 反事实推理
- 时序推理

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
├── baseline2/                     # Baseline 2: 经典研究方法
│   ├── preprocessing.py           # 数据预处理
│   ├── unifiedqa_baseline.py      # UnifiedQA模型
│   ├── roberta_mcqa_baseline.py   # RoBERTa MCQA模型
│   ├── run_baseline2.py           # 统一运行脚本
│   └── AER_Baseline2_Colab.ipynb
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
- [UnifiedQA](https://github.com/allenai/unifiedqa) - AllenAI统一问答模型
- [HuggingFace Multiple Choice](https://huggingface.co/docs/transformers/tasks/multiple_choice) - 多选题任务教程
