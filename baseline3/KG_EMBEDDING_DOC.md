# 知识图谱嵌入 (Knowledge Graph Embedding) 技术文档

## 1. 概述

知识图谱嵌入 (KGE) 是将知识图谱中的实体和关系映射到低维连续向量空间的技术。这些嵌入向量可以用于：

- **链接预测**: 预测缺失的三元组
- **实体分类**: 根据嵌入对实体进行分类
- **知识增强**: 为下游任务提供结构化知识

在本项目中，我们使用 KGE 来增强溯因事件推理任务。

---

## 2. 核心概念

### 2.1 知识图谱结构

知识图谱由**三元组** (Triple) 组成:

```
(头实体, 关系, 尾实体) = (h, r, t)
```

例如:
```
(经济衰退, 导致, 失业率上升)
(利率下调, 导致, 股市上涨)
(贸易战, 引发, 经济不确定性)
```

### 2.2 嵌入目标

将每个实体和关系映射为 d 维向量:
- 实体嵌入: `e ∈ ℝ^d`
- 关系嵌入: `r ∈ ℝ^d` 或 `r ∈ ℂ^d`

训练目标: 对于真实三元组 (h, r, t)，使得某种得分函数 `f(h, r, t)` 最大化。

---

## 3. 嵌入模型详解

### 3.1 TransE (翻译距离模型)

**论文**: Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)

**核心思想**: 关系被建模为向量空间中的**平移操作**

```
h + r ≈ t
```

**得分函数** (距离越小越好):
```python
score = ||h + r - t||_p  # p=1 或 p=2
```

**损失函数** (Margin Ranking Loss):
```python
L = max(0, margin + d(h+r, t) - d(h'+r, t'))
```
其中 (h', r, t') 是负样本（随机替换头或尾实体）

**代码实现**:
```python
class TransE(nn.Module):
    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # h + r - t 的范数
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score
```

**优点**:
- 简单高效
- 参数量少: O(n_e * d + n_r * d)

**局限**:
- 无法建模对称关系 (如 "相似于")
- 无法建模 1-N, N-1, N-N 关系

---

### 3.2 ComplEx (复数空间嵌入)

**论文**: Complex Embeddings for Simple Link Prediction (ICML 2016)

**核心思想**: 使用**复数**来建模实体和关系

```
实体: e = e_re + i * e_im
关系: r = r_re + i * r_im
```

**得分函数** (Hermitian 点积):
```python
score = Re(<h, r, conj(t)>)
      = Σ (h_re * r_re * t_re
         + h_re * r_im * t_im
         + h_im * r_re * t_im
         - h_im * r_im * t_re)
```

**代码实现**:
```python
class ComplEx(nn.Module):
    def forward(self, heads, relations, tails):
        h_re, h_im = self.entity_re(heads), self.entity_im(heads)
        r_re, r_im = self.relation_re(relations), self.relation_im(relations)
        t_re, t_im = self.entity_re(tails), self.entity_im(tails)

        score = (h_re * r_re * t_re).sum(dim=1) + \
                (h_re * r_im * t_im).sum(dim=1) + \
                (h_im * r_re * t_im).sum(dim=1) - \
                (h_im * r_im * t_re).sum(dim=1)
        return score
```

**优点**:
- 可以建模**非对称关系**: 因为 `f(h,r,t) ≠ f(t,r,h)`
- 表达能力比 TransE 更强

**损失函数**: Binary Cross Entropy + L2 正则化

---

### 3.3 RotatE (旋转模型)

**论文**: Knowledge Graph Embedding by Relational Rotation in Complex Space (ICLR 2019)

**核心思想**: 关系被建模为复平面上的**旋转**

```
t = h ◦ r  (复数乘法/旋转)
```

其中关系 r 被限制为单位复数: `|r| = 1`，即 `r = e^(iθ) = cos(θ) + i*sin(θ)`

**得分函数**:
```python
score = ||h ◦ r - t||
```

**代码实现**:
```python
class RotatE(nn.Module):
    def forward(self, heads, relations, tails):
        h_re, h_im = self.entity_re(heads), self.entity_im(heads)
        t_re, t_im = self.entity_re(tails), self.entity_im(tails)

        phase = self.relation_phase(relations)  # 角度
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # 复数乘法
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # 距离
        score = torch.sqrt((hr_re - t_re)**2 + (hr_im - t_im)**2).sum(dim=1)
        return score
```

**优点**:
- 可以建模: 对称、反对称、逆、组合关系
- 理论表达能力最强

**可建模的关系模式**:

| 关系模式 | 条件 | 示例 |
|---------|------|------|
| 对称 | r = r⁻¹ (θ = 0 或 π) | "相似于" |
| 反对称 | r ≠ r⁻¹ | "父亲是" |
| 逆 | r₁ = r₂⁻¹ | "父亲" vs "孩子" |
| 组合 | r₁ ◦ r₂ = r₃ | "祖父 = 父亲 ◦ 父亲" |

---

## 4. 训练流程

### 4.1 负采样

对于每个正样本三元组 (h, r, t)，生成负样本:

```python
def generate_negative(h, r, t, num_entities):
    if random() < 0.5:
        # 替换头实体
        h_neg = random_entity(exclude=h)
        return (h_neg, r, t)
    else:
        # 替换尾实体
        t_neg = random_entity(exclude=t)
        return (h, r, t_neg)
```

### 4.2 训练循环

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 正样本
        h, r, t = batch['pos_head'], batch['pos_relation'], batch['pos_tail']

        # 负样本
        h_neg, t_neg = batch['neg_head'], batch['neg_tail']

        # 计算损失
        pos_score = model(h, r, t)
        neg_score = model(h_neg, r, t_neg)

        loss = margin_ranking_loss(pos_score, neg_score)

        # 反向传播
        loss.backward()
        optimizer.step()

        # TransE 需要归一化
        if model_type == 'TransE':
            model.normalize_entities()
```

### 4.3 超参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| embedding_dim | 100-500 | 维度越大表达能力越强 |
| margin | 1.0-12.0 | TransE 用 1.0, RotatE 用 9.0 |
| learning_rate | 0.001-0.0001 | Adam 优化器 |
| batch_size | 128-1024 | 越大越稳定 |
| negative_samples | 5-50 | 每个正样本的负样本数 |

---

## 5. 在 AER 任务中的应用

### 5.1 知识图谱构建

从 AER 数据集构建因果知识图谱:

```python
# 从事件和选项构建三元组
for instance in dataset:
    event = instance.target_event
    for option in instance.options:
        # 使用 COMET 判断因果关系
        if is_cause(option, event):
            kg.add_triple(option, "causes", event)
```

### 5.2 嵌入生成

```python
# 训练 KG 嵌入
trainer = KGEmbeddingTrainer(kg, model_type="TransE")
trainer.train()

# 获取嵌入
embeddings = trainer.get_entity_embeddings()
# embeddings["经济衰退"] = array([0.1, -0.2, 0.3, ...])
```

### 5.3 嵌入融合

**方法 1: Prompt 增强**
```python
# 将 KG 知识转为自然语言
kg_context = "Based on causal knowledge: X causes Y, Z leads to W..."
prompt = f"{kg_context}\n\nQuestion: What caused {event}?"
```

**方法 2: 嵌入拼接**
```python
# 将 KG 嵌入与文本嵌入拼接
combined = torch.cat([text_embedding, kg_embedding], dim=-1)
logits = fusion_layer(combined)
```

**方法 3: 注意力融合**
```python
# 使用注意力机制融合
attention = softmax(query @ kg_embeddings.T)
kg_context = attention @ kg_embeddings
output = text_embedding + kg_context
```

---

## 6. 代码使用示例

### 6.1 构建知识图谱

```python
from kg_embedding import CausalKnowledgeGraph, KGEmbeddingTrainer, KGEConfig

# 创建知识图谱
kg = CausalKnowledgeGraph()

# 添加因果关系
kg.add_causal_relation("利率下调", "股市上涨")
kg.add_causal_relation("经济衰退", "失业率上升")
kg.add_causal_relation("贸易战", "经济不确定性")

print(f"实体数: {kg.num_entities}")
print(f"三元组数: {len(kg.triples)}")
```

### 6.2 训练嵌入

```python
# 配置
config = KGEConfig(
    embedding_dim=256,
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001
)

# 训练
trainer = KGEmbeddingTrainer(kg, model_type="TransE", config=config)
results = trainer.train()

# 保存
trainer.save_model("kg_model.pt")
trainer.save_embeddings("embeddings.npz")
```

### 6.3 使用嵌入

```python
import numpy as np

# 加载嵌入
embeddings = np.load("embeddings.npz")

# 获取特定实体的嵌入
event_emb = embeddings["利率下调"]
print(f"嵌入维度: {event_emb.shape}")
print(f"嵌入范数: {np.linalg.norm(event_emb):.4f}")

# 计算相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim = cosine_similarity(embeddings["利率下调"], embeddings["股市上涨"])
print(f"相似度: {sim:.4f}")
```

---

## 7. 参考资源

### 论文
- [TransE](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) - NeurIPS 2013
- [ComplEx](https://arxiv.org/abs/1606.06357) - ICML 2016
- [RotatE](https://arxiv.org/abs/1902.10197) - ICLR 2019
- [COMET-ATOMIC](https://arxiv.org/abs/2010.05953) - EMNLP 2020

### 代码库
- [OpenKE](https://github.com/thunlp/OpenKE) - 清华大学 KG 嵌入工具包
- [TorchKGE](https://github.com/torchkge-team/torchkge) - PyTorch KG 嵌入库
- [PyKG2Vec](https://github.com/Sujit-O/pykg2vec) - KG 嵌入 Python 库
- [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020) - 常识知识生成

### 综述
- [KG-LLM-Papers](https://github.com/zjukg/KG-LLM-Papers) - KG+LLM 论文列表
- [Awesome-LLM-Causal-Reasoning](https://github.com/chendl02/Awesome-LLM-causal-reasoning)

---

## 8. 常见问题

### Q: 应该选择哪个模型?

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 快速原型 | TransE | 简单高效 |
| 非对称关系 | ComplEx | 可建模方向性 |
| 复杂关系 | RotatE | 表达能力最强 |
| 大规模图谱 | TransE/PBG | 训练速度快 |

### Q: 嵌入维度如何选择?

- 小规模 KG (<10K 实体): 64-128 维
- 中规模 KG (10K-100K): 128-256 维
- 大规模 KG (>100K): 256-512 维

### Q: 如何评估嵌入质量?

1. **链接预测**: MRR, Hits@1/3/10
2. **三元组分类**: 准确率
3. **下游任务**: 在 QA 等任务上的表现
