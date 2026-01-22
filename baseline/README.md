# Baseline 1: LLM API 调用

基于大语言模型API的零样本/少样本因果推理baseline。

## 目录
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [评估指标](#评估指标)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

---

## 环境配置

### 1. 安装依赖

```bash
cd baseline
pip install -r requirements.txt
```

### 2. 配置API密钥

根据使用的模型，设置相应的环境变量：

```bash
# OpenAI
export OPENAI_API_KEY="sk-your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# HuggingFace (可选，用于私有模型)
export HF_TOKEN="your-token"
```

### 3. 本地模型配置 (可选)

如果使用Ollama本地模型：
```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull llama3.1:8b
```

---

## 数据准备

### 1. 下载数据集

```bash
# 在项目根目录执行
git clone https://github.com/sooo66/semeval2026-task12-dataset.git data/semeval2026-task12-dataset
```

### 2. 数据结构

```
data/semeval2026-task12-dataset/
├── train_data/
│   ├── questions.jsonl    # 训练集问题
│   └── docs.json          # 相关文档
├── dev_data/
│   ├── questions.jsonl    # 开发集问题
│   └── docs.json
└── test_data/
    ├── questions.jsonl    # 测试集问题（无答案）
    └── docs.json
```

### 3. 数据格式

**questions.jsonl** 每行格式：
```json
{
  "id": "train_001",
  "topic_id": "topic_001",
  "target_event": "Cryptocurrency market prices soar",
  "option_A": "Government announces national cryptocurrency reserve",
  "option_B": "New environmental regulations implemented",
  "option_C": "Central bank raises interest rates",
  "option_D": "Major tech company reports losses",
  "golden_answer": "A"
}
```

**docs.json** 格式：
```json
[
  {
    "topic_id": "topic_001",
    "topic": "Cryptocurrency market movement",
    "docs": [
      {
        "title": "Document title",
        "content": "Full document content...",
        "summary": "Brief summary..."
      }
    ]
  }
]
```

---

## 快速开始

### 最简单的运行方式

```bash
cd baseline

# 使用GPT-4o-mini在开发集上运行
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --data-split dev
```

### 快速测试（限制样本数）

```bash
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --data-split dev \
    --max-samples 10 \
    --output results/quick_test.json
```

---

## 详细使用

### 命令行参数

```bash
python run_baseline.py [OPTIONS]

必选参数:
  --model-type TYPE      模型类型: openai, anthropic, huggingface, ollama, vllm
  --data-split SPLIT     数据集划分: train, dev, test

可选参数:
  --model-name NAME      具体模型名称 (默认根据类型自动选择)
  --no-context           不使用上下文文档
  --max-samples N        最大样本数 (用于快速测试)
  --output PATH          结果保存路径
```

### 支持的模型

| 模型类型 | 模型名称示例 | 说明 |
|---------|-------------|------|
| `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | 需要API Key |
| `anthropic` | `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229` | 需要API Key |
| `huggingface` | `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | 本地GPU |
| `ollama` | `llama3.1:8b`, `qwen2:7b`, `mistral` | 本地Ollama服务 |
| `vllm` | `meta-llama/Llama-3.1-8B-Instruct` | 高性能批量推理 |

### 运行示例

#### OpenAI GPT-4o

```bash
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o \
    --data-split dev \
    --output results/gpt4o_dev.json
```

#### Anthropic Claude

```bash
python run_baseline.py \
    --model-type anthropic \
    --model-name claude-3-5-sonnet-20241022 \
    --data-split dev \
    --output results/claude_dev.json
```

#### 本地HuggingFace模型

```bash
python run_baseline.py \
    --model-type huggingface \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --data-split dev \
    --output results/qwen_dev.json
```

#### Ollama本地模型

```bash
# 确保Ollama服务运行中
python run_baseline.py \
    --model-type ollama \
    --model-name llama3.1:8b \
    --data-split dev \
    --output results/ollama_dev.json
```

#### 不使用上下文

```bash
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --data-split dev \
    --no-context \
    --output results/gpt4o_mini_no_context.json
```

---

## 评估指标

### 官方评分规则

| 情况 | 得分 | 说明 |
|------|------|------|
| 完全匹配 | 1.0 | 预测 = 标准答案 |
| 部分匹配 | 0.5 | 预测是标准答案的真子集 |
| 错误 | 0.0 | 其他情况（包括空预测、包含错误选项） |

**最终分数** = 所有实例得分的平均值

### 示例

```
标准答案: {A, B}
预测: {A, B}  → 1.0 (完全匹配)
预测: {A}     → 0.5 (部分匹配)
预测: {A, C}  → 0.0 (包含错误选项C)
预测: {}      → 0.0 (空预测)
```

### 单独运行评估

```python
from evaluator import evaluate, parse_prediction

# 假设有预测结果和标准答案
predictions = [{"A"}, {"B", "C"}, {"A"}]
goldens = [{"A"}, {"B"}, {"A", "B"}]

results = evaluate(predictions, goldens)
print(f"Score: {results['score']:.4f}")
print(f"Exact Match: {results['exact_match_rate']:.4f}")
```

---

## 实验结果

### 输出格式

运行完成后，结果文件 (`--output`) 格式如下：

```json
{
  "config": {
    "model_type": "openai",
    "model_name": "gpt-4o-mini",
    "data_split": "dev",
    "use_context": true,
    "num_samples": 500
  },
  "results": {
    "score": 0.6520,
    "exact_match_rate": 0.5840,
    "partial_match_rate": 0.1360,
    "wrong_rate": 0.2800,
    "total": 500,
    "exact_match": 292,
    "partial_match": 68,
    "wrong": 140
  },
  "predictions": [
    {
      "id": "dev_001",
      "target_event": "Stock market crashes",
      "golden": ["A"],
      "prediction": ["A"],
      "raw_output": "A"
    }
  ]
}
```

### 结果汇报模板

```markdown
## 实验结果

### 实验设置
- 模型: GPT-4o-mini
- 数据集: SemEval 2026 Task 12 Dev Set
- 样本数: 500
- 使用上下文: 是

### 主要指标

| 指标 | 值 |
|------|-----|
| **Score (官方)** | 0.6520 |
| Exact Match | 0.5840 |
| Partial Match | 0.1360 |
| Wrong | 0.2800 |

### 不同模型对比

| 模型 | Score | Exact Match | 备注 |
|------|-------|-------------|------|
| GPT-4o | 0.72 | 0.68 | 最佳 |
| GPT-4o-mini | 0.65 | 0.58 | 性价比高 |
| Claude-3.5-Sonnet | 0.70 | 0.65 | |
| Qwen2.5-7B | 0.55 | 0.48 | 本地运行 |
| Random | 0.15 | 0.10 | 基线 |
```

---

## Python API 使用

### 基本使用

```python
from data_loader import AERDataLoader, download_dataset
from models import get_model, AERPrompt
from evaluator import evaluate, parse_prediction

# 1. 下载并加载数据
dataset_path = download_dataset()
loader = AERDataLoader(dataset_path / "dev_data")
instances = loader.load()

# 2. 初始化模型
model = get_model("openai", model_name="gpt-4o-mini")

# 3. 预测
predictions = []
for inst in instances:
    prompt = AERPrompt(
        target_event=inst.target_event,
        options=inst.options,
        context=prepare_context(inst)  # 可选
    )
    raw_output = model.predict(prompt)
    predictions.append(parse_prediction(raw_output))

# 4. 评估
goldens = [set(inst.golden_answer) for inst in instances]
results = evaluate(predictions, goldens)
print(f"Score: {results['score']:.4f}")
```

### 自定义Prompt

```python
# 修改 models.py 中的 format_prompt 方法
def format_prompt(self, prompt: AERPrompt, include_context: bool = True) -> str:
    system_prompt = """你是因果推理专家。给定一个事件，从选项中选择最可能的直接原因。

规则:
1. 选择最直接的原因
2. 原因必须在结果之前发生
3. 可以选择多个同样直接的原因

输出格式: 只输出选项字母，多选用逗号分隔。"""

    # 自定义user_prompt...
```

---

## 常见问题

### Q: API调用失败怎么办？

```python
# 检查API Key
import os
print(os.getenv("OPENAI_API_KEY"))

# 测试连接
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### Q: 如何减少API成本？

1. 使用 `gpt-4o-mini` 而不是 `gpt-4o`
2. 使用 `--no-context` 减少token数
3. 使用 `--max-samples` 限制测试样本

### Q: 本地模型显存不足？

```python
# 使用4-bit量化
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4-bit量化
    device_map="auto"
)
```

### Q: 如何添加新模型？

在 `models.py` 中继承 `BaseModel` 类：

```python
class MyModel(BaseModel):
    def __init__(self, model_name: str):
        # 初始化
        pass

    def predict(self, prompt: AERPrompt) -> str:
        # 实现预测逻辑
        return "A"
```

---

## 文件说明

```
baseline/
├── data_loader.py       # 数据加载器
├── evaluator.py         # 评估器
├── models.py            # 模型封装 (OpenAI, Anthropic, HuggingFace, Ollama, vLLM)
├── run_baseline.py      # 主运行脚本
├── requirements.txt     # 依赖
├── AER_Baseline_Colab.ipynb  # Colab Notebook
└── README.md            # 本文档
```
