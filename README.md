# SemEval 2026 Task 12: Abductive Event Reasoning (AER)

## 任务简介

**溯因事件推理 (Abductive Event Reasoning)** 是 SemEval 2026 的共享任务，旨在评估大语言模型推理真实世界事件因果关系的能力。

### 核心任务
给定一个事件（如"加密货币价格飙升"）和相关文档，模型需要从候选选项中识别**最可能的直接原因**。

### 评估指标
- **1.0分**: 完全匹配（预测 = 标准答案）
- **0.5分**: 部分匹配（预测是标准答案的真子集）
- **0.0分**: 错误

## Baseline 框架

本仓库提供了完整的baseline实验框架，支持多种模型。

### 快速开始

```bash
# 1. 克隆仓库
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

### Colab 快速体验

打开 `baseline/AER_Baseline_Colab.ipynb` 在 Google Colab 中运行。

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

## 文件结构

```
baseline/
├── data_loader.py       # 数据加载器
├── evaluator.py         # 评估器
├── models.py            # 模型封装
├── run_baseline.py      # 主运行脚本
├── requirements.txt     # 依赖
└── AER_Baseline_Colab.ipynb  # Colab notebook
```

## 参考资源

- [官方数据集](https://github.com/sooo66/semeval2026-task12-dataset)
- [竞赛页面](https://www.codabench.org/competitions/12440/)
- [SemEval 2026](https://semeval.github.io/SemEval2026/)
