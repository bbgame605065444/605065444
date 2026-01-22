# Baseline 3: 知识图谱增强 LLM QA (KG + LLM)

将知识图谱嵌入与大语言模型结合进行因果推理的baseline实现。

## 目录
- [方法概述](#方法概述)
- [环境配置](#环境配置)
- [完整流程](#完整流程)
- [步骤1: 构建知识图谱](#步骤1-构建知识图谱)
- [步骤2: 训练KG嵌入](#步骤2-训练kg嵌入)
- [步骤3: KG增强QA](#步骤3-kg增强qa)
- [评估与结果](#评估与结果)
- [实验结果汇报](#实验结果汇报)
- [KG Embedding详解](#kg-embedding详解)
- [常见问题](#常见问题)

---

## 方法概述

### 核心思想

将**结构化因果知识**与**LLM的推理能力**结合：

```
┌─────────────────┐     ┌─────────────────┐
│   事件/选项      │────▶│  COMET知识生成   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  因果知识图谱    │◀────│   三元组抽取     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  KG Embedding    │────▶│  知识增强Prompt  │
│ (TransE/ComplEx) │     └────────┬────────┘
└─────────────────┘              │
                                 ▼
                        ┌─────────────────┐
                        │    LLM 推理      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    因果答案      │
                        └─────────────────┘
```

### 融合方法

| 方法 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| `prompt` | 将KG知识转为自然语言加入prompt | 简单直接 | 上下文长度限制 |
| `retrieval` | 从KG检索相关三元组 | 针对性强 | 需要好的检索策略 |
| `embedding` | 将KG嵌入与文本嵌入融合 | 端到端 | 需要额外训练 |

---

## 环境配置

### 1. 安装依赖

```bash
cd baseline3
pip install -r requirements.txt
```

### 2. 可选依赖

```bash
# COMET知识生成（需要~3GB显存）
pip install transformers sentencepiece

# 更多KG嵌入方法
pip install torchkge  # TorchKGE库
```

### 3. API配置

```bash
# 用于KG-LLM融合QA
export OPENAI_API_KEY="your-key"
# 或
export ANTHROPIC_API_KEY="your-key"
```

---

## 完整流程

### 快速开始（最简版本）

```bash
cd baseline3

# 1. 构建知识图谱（使用简单知识库，无需GPU）
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output \
    --max-samples 100

# 2. 运行KG增强QA
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --max-samples 50 \
    --output ./results/kg_llm_quick.json
```

### 完整流程（推荐）

```bash
# 1. 构建知识图谱 + 训练嵌入
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output \
    --train-embedding \
    --kg-model TransE \
    --embedding-dim 256 \
    --epochs 100

# 2. 运行评估
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --output ./results/kg_llm_dev.json
```

### 使用COMET（需要GPU）

```bash
# 1. 使用COMET生成知识
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output_comet \
    --use-comet \
    --train-embedding \
    --kg-model RotatE

# 2. 运行QA
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --use-comet \
    --llm-model gpt-4o \
    --output ./results/kg_comet_llm_dev.json
```

---

## 步骤1: 构建知识图谱

### 命令参数

```bash
python run_baseline3.py build-kg [OPTIONS]

必选:
  --data-path PATH       数据目录路径

可选:
  --output-dir DIR       输出目录 (默认: ./kg_output)
  --use-comet            使用COMET生成知识 (需要GPU)
  --train-embedding      训练KG嵌入
  --kg-model MODEL       嵌入模型: TransE, ComplEx, RotatE (默认: TransE)
  --embedding-dim N      嵌入维度 (默认: 256)
  --epochs N             训练轮数 (默认: 100)
  --batch-size N         批次大小 (默认: 256)
  --max-samples N        最大样本数
```

### 输出文件

```
kg_output/
├── knowledge_graph.json   # 知识图谱 (实体、关系、三元组)
├── kg_model.pt            # 训练好的KG嵌入模型
└── embeddings.npz         # 实体嵌入向量
```

### 知识图谱格式

```json
{
  "entities": {
    "Economic recession": 0,
    "Unemployment rises": 1,
    "Interest rate cut": 2,
    ...
  },
  "relations": {
    "causes": 0,
    "is_caused_by": 1,
    "enables": 2,
    ...
  },
  "triples": [
    [0, 0, 1],  // Economic recession causes Unemployment rises
    [2, 0, 3],  // Interest rate cut causes Stock market rises
    ...
  ]
}
```

### Python API

```python
from kg_embedding import CausalKnowledgeGraph, KGEmbeddingTrainer, KGEConfig
from comet_knowledge import build_event_knowledge_graph

# 方法1: 手动构建
kg = CausalKnowledgeGraph()
kg.add_causal_relation("利率下调", "股市上涨")
kg.add_causal_relation("经济衰退", "失业率上升")
kg.save("./kg_output/knowledge_graph.json")

# 方法2: 从数据自动构建
events = ["Event1", "Event2", "Option1", "Option2", ...]
enhanced_events, kg = build_event_knowledge_graph(
    events,
    use_comet=True  # 使用COMET生成因果知识
)
```

---

## 步骤2: 训练KG嵌入

### 支持的模型

| 模型 | 核心思想 | 得分函数 | 适用场景 |
|------|----------|----------|----------|
| **TransE** | h + r ≈ t | \|\|h+r-t\|\| | 快速原型、简单关系 |
| **ComplEx** | 复数空间 | Re(⟨h,r,t̄⟩) | 非对称关系 |
| **RotatE** | 旋转操作 | \|\|h∘r-t\|\| | 复杂关系模式 |

### 训练命令

```bash
# TransE (最快)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_transe \
    --train-embedding \
    --kg-model TransE \
    --embedding-dim 256 \
    --epochs 100

# RotatE (最强)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_rotate \
    --train-embedding \
    --kg-model RotatE \
    --embedding-dim 256 \
    --epochs 200
```

### 训练日志示例

```
Building Causal Knowledge Graph
============================================================
Loaded 2000 instances
Collected 5234 unique events

Knowledge Graph Statistics:
  Entities: 5234
  Relations: 8
  Triples: 12456

Training TransE embeddings...
Epoch 10/100, Loss: 0.8234
Epoch 20/100, Loss: 0.5421
Epoch 30/100, Loss: 0.3218
...
Epoch 100/100, Loss: 0.0842

Training complete!
  Final loss: 0.0842
  Model saved: ./kg_output/kg_model.pt
  Embeddings saved: ./kg_output/embeddings.npz
```

### Python API

```python
from kg_embedding import KGEmbeddingTrainer, KGEConfig

# 配置
config = KGEConfig(
    embedding_dim=256,
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001,
    margin=1.0  # TransE
)

# 训练
trainer = KGEmbeddingTrainer(kg, model_type="TransE", config=config)
results = trainer.train()

# 保存
trainer.save_model("./kg_model.pt")
trainer.save_embeddings("./embeddings.npz")

# 获取嵌入
embeddings = trainer.get_entity_embeddings()
print(embeddings["Economic recession"].shape)  # (256,)
```

### 使用嵌入

```python
import numpy as np

# 加载嵌入
embeddings = np.load("./kg_output/embeddings.npz")

# 计算相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

event1_emb = embeddings["Interest rate cut"]
event2_emb = embeddings["Stock market rises"]
similarity = cosine_sim(event1_emb, event2_emb)
print(f"相似度: {similarity:.4f}")
```

---

## 步骤3: KG增强QA

### 命令参数

```bash
python run_baseline3.py qa [OPTIONS]

必选:
  --data-path PATH       数据目录路径

可选:
  --fusion METHOD        融合方法: prompt, retrieval (默认: prompt)
  --llm-type TYPE        LLM类型: openai, anthropic (默认: openai)
  --llm-model MODEL      具体模型名称 (默认: gpt-4o-mini)
  --use-comet            使用COMET生成知识
  --kg-path PATH         KG输出目录 (用于retrieval)
  --output PATH          结果保存路径
  --max-samples N        最大样本数
```

### 融合方法详解

#### 方法1: Prompt增强 (`--fusion prompt`)

将KG知识转换为自然语言，添加到LLM prompt中：

```
Target Event: Stock market crashes

Causal Knowledge:
- Common causes of similar events: Economic recession; Trade war; Interest rate hike
- If Option A happens, likely effects: Unemployment rises; Consumer spending decreases
- If Option B happens, likely effects: Currency weakens; Import costs increase

Options:
A. Economic recession deepens
B. Trade deficit grows
C. New product launches
D. Weather changes

Answer:
```

#### 方法2: 检索增强 (`--fusion retrieval`)

从KG中检索与问题相关的三元组：

```
Target Event: Stock market crashes

Related causal facts:
- Economic recession causes Stock market crash
- Trade war leads_to Economic uncertainty
- Interest rate hike causes Stock market decline

Options:
...
```

### 运行示例

```bash
# Prompt增强 + GPT-4o-mini
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --output ./results/prompt_gpt4omini.json

# Prompt增强 + GPT-4o
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o \
    --output ./results/prompt_gpt4o.json

# 检索增强
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion retrieval \
    --kg-path ./kg_output \
    --llm-model gpt-4o-mini \
    --output ./results/retrieval_gpt4omini.json

# 使用Claude
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-type anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --output ./results/prompt_claude.json
```

### Python API

```python
from kg_llm_qa import KGLLMQA, KGLLMConfig

# 配置
config = KGLLMConfig(
    fusion_method="prompt",
    llm_type="openai",
    llm_model="gpt-4o-mini",
    use_comet=False,
    max_kg_context_length=500
)

# 初始化
qa_system = KGLLMQA(config)
qa_system.setup()

# 单条预测
from baseline.data_loader import AERInstance
instance = AERInstance(
    id="test_001",
    topic_id="topic_001",
    target_event="Stock market crashes",
    options={"A": "Economic recession", "B": "Good weather", ...},
    golden_answer=["A"]
)
prediction = qa_system.predict(instance)
print(prediction)  # {"A"}

# 批量评估
results = qa_system.evaluate(instances, output_path="./results.json")
```

---

## 评估与结果

### 评估指标

与Baseline 1/2相同：

| 情况 | 得分 |
|------|------|
| 完全匹配 | 1.0 |
| 部分匹配 | 0.5 |
| 错误 | 0.0 |

### 输出文件格式

```json
{
  "config": {
    "fusion_method": "prompt",
    "llm_model": "gpt-4o-mini",
    "use_comet": false
  },
  "results": {
    "score": 0.6720,
    "exact_match_rate": 0.6040,
    "partial_match_rate": 0.1360,
    "wrong_rate": 0.2600
  },
  "predictions": [
    {
      "id": "dev_001",
      "target_event": "Stock market crashes",
      "prediction": ["A"],
      "golden": ["A"]
    }
  ]
}
```

---

## 实验结果汇报

### 汇报模板

```markdown
## 实验结果报告

### 实验设置

| 项目 | 值 |
|------|-----|
| 任务 | SemEval 2026 Task 12: AER |
| 数据集 | Dev Set (500 samples) |
| KG嵌入模型 | TransE (dim=256) |
| 知识来源 | Simple KB / COMET |
| 硬件 | NVIDIA RTX 3090 |

### 主要结果

| 方法 | LLM | Score | Exact Match |
|------|-----|-------|-------------|
| Baseline (无KG) | GPT-4o-mini | 0.65 | 0.58 |
| KG-Prompt | GPT-4o-mini | **0.67** | **0.60** |
| KG-Prompt | GPT-4o | **0.72** | **0.66** |
| KG-Retrieval | GPT-4o-mini | 0.66 | 0.59 |

### 消融实验

| 设置 | Score | Δ |
|------|-------|---|
| 完整系统 | 0.67 | - |
| 不使用KG | 0.65 | -0.02 |
| 不使用COMET | 0.66 | -0.01 |
| 不使用文档上下文 | 0.58 | -0.09 |

### KG嵌入模型对比

| 模型 | 训练时间 | 最终Loss | 下游Score |
|------|---------|----------|----------|
| TransE | 5 min | 0.084 | 0.67 |
| ComplEx | 8 min | 0.052 | 0.67 |
| RotatE | 12 min | 0.031 | 0.68 |

### 结论

1. KG增强可提升约 **+0.02** 的Score
2. **RotatE** 嵌入略优于TransE
3. **COMET** 知识生成有小幅提升
4. 文档上下文仍是最重要的因素
```

---

## KG Embedding详解

详细技术文档见 [`KG_EMBEDDING_DOC.md`](./KG_EMBEDDING_DOC.md)，包括：

- TransE/ComplEx/RotatE 数学原理
- 损失函数与训练流程
- 超参数调优指南
- 在AER任务中的应用

### 快速参考

#### TransE
```
h + r ≈ t
Loss = max(0, margin + ||h+r-t|| - ||h'+r-t'||)
```

#### ComplEx
```
score = Re(⟨h, r, t̄⟩)
Loss = BCE(score) + L2_reg
```

#### RotatE
```
t = h ◦ r (复数乘法/旋转)
Loss = -log σ(γ - ||h◦r - t||)
```

---

## 常见问题

### Q: COMET模型下载失败？

```bash
# 手动下载
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "mismayil/comet-bart-ai2",
    cache_dir="./model_cache"  # 指定缓存目录
)
```

### Q: KG太小怎么办？

1. 使用COMET生成更多知识
2. 增加训练数据
3. 使用外部知识库（如ConceptNet、ATOMIC）

### Q: 如何添加外部知识库？

```python
# 加载ConceptNet
import requests

def query_conceptnet(concept):
    url = f"http://api.conceptnet.io/c/en/{concept}"
    response = requests.get(url).json()
    return response.get("edges", [])

# 添加到KG
for edge in query_conceptnet("economic_recession"):
    if edge["rel"]["label"] == "Causes":
        kg.add_triple(
            edge["start"]["label"],
            "causes",
            edge["end"]["label"]
        )
```

### Q: 嵌入维度如何选择？

| KG规模 | 推荐维度 |
|--------|---------|
| < 1K 实体 | 64-128 |
| 1K-10K | 128-256 |
| 10K-100K | 256-512 |
| > 100K | 512+ |

---

## 文件说明

```
baseline3/
├── kg_embedding.py            # KG嵌入模型 (TransE/ComplEx/RotatE)
├── comet_knowledge.py         # COMET知识生成
├── kg_llm_qa.py               # KG-LLM融合QA
├── run_baseline3.py           # 统一运行脚本
├── requirements.txt           # 依赖
├── KG_EMBEDDING_DOC.md        # KG嵌入技术文档
└── README.md                  # 本文档
```

---

## 参考文献

### KG Embedding
- [TransE (NeurIPS 2013)](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [ComplEx (ICML 2016)](https://arxiv.org/abs/1606.06357)
- [RotatE (ICLR 2019)](https://arxiv.org/abs/1902.10197)

### 知识增强QA
- [QA-GNN (NAACL 2021)](https://arxiv.org/abs/2104.06378)
- [DRAGON (NeurIPS 2022)](https://arxiv.org/abs/2210.09338)
- [KG-BERT (2019)](https://arxiv.org/abs/1909.03193)

### 常识知识
- [COMET-ATOMIC 2020 (EMNLP 2020)](https://arxiv.org/abs/2010.05953)
- [ATOMIC (AAAI 2019)](https://arxiv.org/abs/1811.00146)

### 代码库
- [OpenKE](https://github.com/thunlp/OpenKE)
- [TorchKGE](https://github.com/torchkge-team/torchkge)
- [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020)
