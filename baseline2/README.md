# Baseline 2: UnifiedQA + RoBERTa MCQA

基于经典NLP研究的baseline实现，包括生成式(UnifiedQA)和判别式(RoBERTa)两种方法。

## 目录
- [环境配置](#环境配置)
- [数据预处理](#数据预处理)
- [方法1: UnifiedQA](#方法1-unifiedqa-零样本)
- [方法2: RoBERTa MCQA](#方法2-robertadeberta-微调)
- [评估与结果](#评估与结果)
- [实验结果汇报](#实验结果汇报)
- [常见问题](#常见问题)

---

## 环境配置

### 1. 安装依赖

```bash
cd baseline2
pip install -r requirements.txt
```

### 2. 验证安装

```python
import torch
from transformers import T5Tokenizer, AutoTokenizer

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3. 显存需求

| 模型 | 最小显存 | 推荐显存 |
|------|---------|---------|
| UnifiedQA-t5-small | 2GB | 4GB |
| UnifiedQA-t5-base | 4GB | 6GB |
| UnifiedQA-t5-large | 8GB | 12GB |
| RoBERTa-base (训练) | 8GB | 12GB |
| DeBERTa-v3-base (训练) | 10GB | 16GB |

---

## 数据预处理

### 1. 下载原始数据集

```bash
# 在项目根目录
git clone https://github.com/sooo66/semeval2026-task12-dataset.git data/semeval2026-task12-dataset
```

### 2. 运行预处理

```bash
cd baseline2

# 完整预处理（包含上下文）
python run_baseline2.py preprocess \
    --dataset-dir ../data/semeval2026-task12-dataset \
    --output-dir ./processed_data

# 不包含上下文（更快，适合快速实验）
python run_baseline2.py preprocess \
    --dataset-dir ../data/semeval2026-task12-dataset \
    --output-dir ./processed_data_no_context \
    --no-context
```

### 3. 预处理输出

```
processed_data/
├── train/
│   ├── unifiedqa.jsonl      # UnifiedQA格式
│   ├── roberta_mcqa.jsonl   # RoBERTa MCQA格式 (SWAG-style)
│   └── data.json            # HuggingFace通用格式
├── dev/
│   ├── unifiedqa.jsonl
│   ├── roberta_mcqa.jsonl
│   └── data.json
└── test/
    ├── unifiedqa.jsonl
    ├── roberta_mcqa.jsonl
    └── data.json
```

### 4. 数据格式说明

#### UnifiedQA格式 (`unifiedqa.jsonl`)

```json
{
  "input": "Context... \\n What is the most plausible cause of: Event \\n (A) Option1 (B) Option2 (C) Option3 (D) Option4",
  "output": "A"
}
```

**注意**: UnifiedQA使用 `\\n` 作为分隔符

#### RoBERTa MCQA格式 (`roberta_mcqa.jsonl`)

```json
{
  "id": "train_001",
  "sent1": "Question: What is the most plausible cause of: Event",
  "sent2": "",
  "ending0": "Option A text",
  "ending1": "Option B text",
  "ending2": "Option C text",
  "ending3": "Option D text",
  "label": 0,
  "labels_all": [0, 1]
}
```

---

## 方法1: UnifiedQA (零样本)

UnifiedQA是AllenAI开发的统一问答模型，基于T5架构，**无需训练**即可直接使用。

### 可用模型

| 模型 | HuggingFace ID | 参数量 | 说明 |
|------|---------------|-------|------|
| UnifiedQA-small | `allenai/unifiedqa-t5-small` | 60M | 快速测试 |
| UnifiedQA-base | `allenai/unifiedqa-t5-base` | 220M | 推荐 |
| UnifiedQA-large | `allenai/unifiedqa-t5-large` | 770M | 更准确 |
| UnifiedQA-v2-base | `allenai/unifiedqa-v2-t5-base-1363200` | 220M | **v2版本，更强** |
| UnifiedQA-v2-large | `allenai/unifiedqa-v2-t5-large-1363200` | 770M | 最强 |

### 运行命令

```bash
# 基础版本
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-t5-base \
    --output ./results/unifiedqa_base_dev.json

# v2版本（推荐）
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-v2-t5-base-1363200 \
    --output ./results/unifiedqa_v2_base_dev.json

# 大模型版本
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-v2-t5-large-1363200 \
    --batch-size 4 \
    --output ./results/unifiedqa_v2_large_dev.json
```

### 参数说明

```bash
python run_baseline2.py unifiedqa [OPTIONS]

必选:
  --data-path PATH       预处理后的UnifiedQA格式数据路径

可选:
  --model-name NAME      模型名称 (默认: allenai/unifiedqa-t5-base)
  --output PATH          结果保存路径
  --batch-size N         批次大小 (默认: 8)
```

### Python API

```python
from unifiedqa_baseline import UnifiedQABaseline, UnifiedQAConfig

# 配置
config = UnifiedQAConfig(
    model_name="allenai/unifiedqa-v2-t5-base-1363200",
    batch_size=8,
    max_input_length=512
)

# 初始化
model = UnifiedQABaseline(config)

# 单条预测
input_text = "What causes rain? \\n (A) Sun (B) Clouds (C) Wind (D) Moon"
output = model.predict_single(input_text)
print(output)  # "B"

# 批量预测
inputs = [input_text1, input_text2, ...]
outputs = model.predict_batch(inputs)

# 完整评估
results = model.run_evaluation(
    data_path="./processed_data/dev/unifiedqa.jsonl",
    output_path="./results/eval.json"
)
```

---

## 方法2: RoBERTa/DeBERTa (微调)

判别式多选题分类方法，**需要在训练集上微调**。

### 可用模型

| 模型 | HuggingFace ID | 说明 |
|------|---------------|------|
| RoBERTa-base | `roberta-base` | 通用baseline |
| RoBERTa-large | `roberta-large` | 更大容量 |
| DeBERTa-v3-base | `microsoft/deberta-v3-base` | **推荐，效果最好** |
| DeBERTa-v3-large | `microsoft/deberta-v3-large` | 最强但需要更多显存 |
| BERT-base | `bert-base-uncased` | 经典baseline |

### 训练流程

#### 步骤1: 训练模型

```bash
python run_baseline2.py roberta --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name microsoft/deberta-v3-base \
    --output-dir ./models/deberta_v3_base \
    --batch-size 4 \
    --epochs 3
```

#### 步骤2: 预测评估

```bash
python run_baseline2.py roberta --mode predict \
    --data-path ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name ./models/deberta_v3_base \
    --output ./results/deberta_v3_base_dev.json
```

#### 步骤3: 测试集预测

```bash
python run_baseline2.py roberta --mode predict \
    --data-path ./processed_data/test/roberta_mcqa.jsonl \
    --model-name ./models/deberta_v3_base \
    --output ./results/deberta_v3_base_test.json
```

### 训练参数说明

```bash
python run_baseline2.py roberta --mode train [OPTIONS]

必选:
  --train-data PATH      训练数据路径

可选:
  --dev-data PATH        验证数据路径 (用于early stopping)
  --model-name NAME      预训练模型名称 (默认: roberta-base)
  --output-dir DIR       模型保存目录 (默认: ./roberta_output)
  --batch-size N         批次大小 (默认: 4)
  --epochs N             训练轮数 (默认: 3)
```

### 训练日志示例

```
Loading model: microsoft/deberta-v3-base
Model loaded on cuda

开始训练: 2000 样本, 3 epochs

Epoch 1/3: 100%|██████████| 500/500 [05:23<00:00]
Epoch 1 Loss: 0.8234
Dev Score: 0.5420

Epoch 2/3: 100%|██████████| 500/500 [05:21<00:00]
Epoch 2 Loss: 0.4521
Dev Score: 0.6180
保存最佳模型到 ./models/deberta_v3_base

Epoch 3/3: 100%|██████████| 500/500 [05:20<00:00]
Epoch 3 Loss: 0.2834
Dev Score: 0.6340
保存最佳模型到 ./models/deberta_v3_base

训练完成! 最佳Dev Score: 0.6340
```

### Python API

```python
from roberta_mcqa_baseline import RoBERTaMCQABaseline, RoBERTaMCQAConfig

# 配置
config = RoBERTaMCQAConfig(
    model_name="microsoft/deberta-v3-base",
    max_length=256,
    batch_size=4,
    learning_rate=2e-5,
    num_epochs=3
)

# 初始化
model = RoBERTaMCQABaseline(config)

# 训练
model.train(
    train_data_path="./processed_data/train/roberta_mcqa.jsonl",
    dev_data_path="./processed_data/dev/roberta_mcqa.jsonl",
    output_dir="./models/my_model"
)

# 预测
results = model.predict(
    data_path="./processed_data/dev/roberta_mcqa.jsonl",
    output_path="./results/predictions.json"
)
```

---

## 评估与结果

### 评估指标

与Baseline 1相同的官方评分规则：

| 情况 | 得分 |
|------|------|
| 完全匹配 | 1.0 |
| 部分匹配 | 0.5 |
| 错误 | 0.0 |

### 输出文件格式

```json
{
  "config": {
    "model_name": "allenai/unifiedqa-v2-t5-base-1363200",
    "data_path": "./processed_data/dev/unifiedqa.jsonl"
  },
  "results": {
    "score": 0.5840,
    "exact_match_rate": 0.5120,
    "partial_match_rate": 0.1440,
    "wrong_rate": 0.3440
  },
  "predictions": [
    {
      "id": "dev_001",
      "input": "What causes...",
      "raw_output": "A",
      "prediction": ["A"],
      "golden": ["A"]
    }
  ]
}
```

### 生成提交文件

```python
def generate_submission(predictions, ids, output_path):
    """生成Codabench提交文件"""
    with open(output_path, 'w') as f:
        for id_, pred in zip(ids, predictions):
            answer = ",".join(sorted(pred)) if pred else "A"
            f.write(f"{id_}\t{answer}\n")
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
| 硬件 | NVIDIA RTX 3090 24GB |
| 框架 | PyTorch 2.0, Transformers 4.35 |

### UnifiedQA 结果

| 模型 | Score | Exact Match | Partial | Wrong |
|------|-------|-------------|---------|-------|
| UnifiedQA-t5-small | 0.42 | 0.35 | 0.14 | 0.51 |
| UnifiedQA-t5-base | 0.52 | 0.45 | 0.14 | 0.41 |
| UnifiedQA-v2-t5-base | **0.58** | **0.51** | 0.14 | 0.35 |
| UnifiedQA-v2-t5-large | 0.62 | 0.55 | 0.14 | 0.31 |

### RoBERTa/DeBERTa 微调结果

| 模型 | Score | Exact Match | 训练时间 |
|------|-------|-------------|---------|
| BERT-base | 0.48 | 0.42 | 15 min |
| RoBERTa-base | 0.55 | 0.48 | 18 min |
| RoBERTa-large | 0.60 | 0.53 | 45 min |
| DeBERTa-v3-base | **0.63** | **0.56** | 25 min |

### 消融实验

| 设置 | Score | 说明 |
|------|-------|------|
| 有上下文 | 0.63 | 完整设置 |
| 无上下文 | 0.51 | -0.12 |
| 1 epoch | 0.52 | 欠拟合 |
| 5 epochs | 0.62 | 略有过拟合 |

### 结论

1. **UnifiedQA-v2** 在零样本设置下表现良好
2. **DeBERTa-v3-base** 微调后达到最佳性能
3. 上下文文档对性能有显著提升
4. 3 epochs 是最佳训练轮数
```

---

## 常见问题

### Q: UnifiedQA输出格式不正确？

UnifiedQA可能输出完整句子而非选项字母。使用 `parse_prediction` 函数：

```python
from evaluator import parse_prediction

raw_output = "The answer is A because..."
pred = parse_prediction(raw_output)  # {"A"}
```

### Q: RoBERTa训练显存不足？

```python
# 减小batch_size
config = RoBERTaMCQAConfig(batch_size=2)

# 或使用梯度累积
# 在训练循环中添加:
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Q: 如何使用混合精度训练？

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Q: 训练不收敛？

1. 降低学习率: `lr=1e-5`
2. 增加warmup: `warmup_ratio=0.1`
3. 检查数据格式是否正确

---

## 文件说明

```
baseline2/
├── preprocessing.py           # 数据预处理
├── unifiedqa_baseline.py      # UnifiedQA模型
├── roberta_mcqa_baseline.py   # RoBERTa MCQA模型
├── run_baseline2.py           # 统一运行脚本
├── requirements.txt           # 依赖
├── AER_Baseline2_Colab.ipynb  # Colab Notebook
└── README.md                  # 本文档
```

---

## 参考文献

- [UnifiedQA: Crossing Format Boundaries with a Single QA System](https://arxiv.org/abs/2005.00700) (EMNLP 2020)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (2019)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) (ICLR 2021)
- [HuggingFace Multiple Choice Tutorial](https://huggingface.co/docs/transformers/tasks/multiple_choice)
