"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 3: Knowledge Graph Embedding Module

实现多种KG Embedding方法:
1. TransE - 翻译距离模型
2. ComplEx - 复数空间嵌入
3. RotatE - 旋转模型

参考论文:
- TransE: Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)
- ComplEx: Complex Embeddings for Simple Link Prediction (ICML 2016)
- RotatE: Knowledge Graph Embedding by Relational Rotation (ICLR 2019)

参考实现:
- https://github.com/thunlp/OpenKE
- https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
- https://github.com/torchkge-team/torchkge
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class KGEConfig:
    """KG Embedding 配置"""
    embedding_dim: int = 256
    margin: float = 1.0  # TransE margin
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 100
    negative_samples: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data

    核心思想: h + r ≈ t
    对于三元组 (head, relation, tail)，头实体加上关系约等于尾实体

    损失函数: max(0, margin + d(h+r, t) - d(h'+r, t'))
    其中 d 是 L1 或 L2 距离
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 1.0,
        norm: int = 1  # L1 or L2
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        # 实体和关系嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # 归一化关系嵌入
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )

    def _normalize_entities(self):
        """归一化实体嵌入（每个batch后调用）"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组的得分（距离越小越好）

        Args:
            heads: (batch_size,) 头实体索引
            relations: (batch_size,) 关系索引
            tails: (batch_size,) 尾实体索引

        Returns:
            scores: (batch_size,) 距离得分
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # h + r - t 的范数
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Margin Ranking Loss

        L = max(0, margin + d_pos - d_neg)
        """
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        loss = torch.relu(self.margin + pos_score - neg_score)
        return loss.mean()

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取单个实体的嵌入"""
        idx = torch.tensor([entity_id], device=self.entity_embeddings.weight.device)
        return self.entity_embeddings(idx).squeeze(0)

    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """获取单个关系的嵌入"""
        idx = torch.tensor([relation_id], device=self.relation_embeddings.weight.device)
        return self.relation_embeddings(idx).squeeze(0)


class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction

    核心思想: 使用复数空间建模实体和关系
    得分函数: Re(<h, r, conj(t)>) = Re(Σ h_i * r_i * conj(t_i))

    优势: 可以建模非对称关系
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        reg_weight: float = 0.01
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.reg_weight = reg_weight

        # 实部和虚部分开存储
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化"""
        for emb in [self.entity_re, self.entity_im,
                    self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组得分

        score = Re(<h, r, conj(t)>)
              = h_re * r_re * t_re
              + h_re * r_im * t_im
              + h_im * r_re * t_im
              - h_im * r_im * t_re
        """
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        r_re = self.relation_re(relations)
        r_im = self.relation_im(relations)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)

        score = (
            (h_re * r_re * t_re).sum(dim=1) +
            (h_re * r_im * t_im).sum(dim=1) +
            (h_im * r_re * t_im).sum(dim=1) -
            (h_im * r_im * t_re).sum(dim=1)
        )

        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary Cross Entropy Loss + L2 正则化
        """
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        # Sigmoid + BCE
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()

        # L2 正则化
        reg = self._regularization(pos_heads, pos_relations, pos_tails)

        return pos_loss + neg_loss + self.reg_weight * reg

    def _regularization(self, heads, relations, tails):
        """L2 正则化"""
        reg = (
            self.entity_re(heads).norm(p=2, dim=1).mean() +
            self.entity_im(heads).norm(p=2, dim=1).mean() +
            self.relation_re(relations).norm(p=2, dim=1).mean() +
            self.relation_im(relations).norm(p=2, dim=1).mean() +
            self.entity_re(tails).norm(p=2, dim=1).mean() +
            self.entity_im(tails).norm(p=2, dim=1).mean()
        )
        return reg / 6

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取实体嵌入（拼接实部和虚部）"""
        idx = torch.tensor([entity_id], device=self.entity_re.weight.device)
        re = self.entity_re(idx).squeeze(0)
        im = self.entity_im(idx).squeeze(0)
        return torch.cat([re, im], dim=0)


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

    核心思想: t = h ◦ r (复数空间中的元素乘法/旋转)
    关系被建模为复平面上的旋转

    优势:
    - 可以建模对称、反对称、逆、组合关系
    - 比TransE和ComplEx表达能力更强
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        epsilon: float = 2.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = epsilon

        # 实体嵌入（复数）
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)

        # 关系嵌入（相位/角度）
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化"""
        embedding_range = (self.margin + self.epsilon) / self.embedding_dim

        nn.init.uniform_(
            self.entity_re.weight, -embedding_range, embedding_range
        )
        nn.init.uniform_(
            self.entity_im.weight, -embedding_range, embedding_range
        )
        nn.init.uniform_(
            self.relation_phase.weight, -math.pi, math.pi
        )

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算得分: ||h ◦ r - t||

        h ◦ r = (h_re + i*h_im) * (cos(θ) + i*sin(θ))
              = (h_re*cos(θ) - h_im*sin(θ)) + i*(h_re*sin(θ) + h_im*cos(θ))
        """
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)

        phase = self.relation_phase(relations)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # 复数乘法
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # 距离
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im

        score = torch.sqrt(diff_re ** 2 + diff_im ** 2).sum(dim=1)
        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.logsigmoid(self.margin - pos_score).mean()
        neg_loss = F.logsigmoid(neg_score - self.margin).mean()

        return -pos_loss - neg_loss


class CausalKnowledgeGraph:
    """
    因果知识图谱

    用于构建和管理事件因果关系的知识图谱
    """

    def __init__(self):
        self.entities: Dict[str, int] = {}  # entity_text -> entity_id
        self.relations: Dict[str, int] = {}  # relation_type -> relation_id
        self.triples: List[Tuple[int, int, int]] = []  # (head, relation, tail)

        # 反向映射
        self.id2entity: Dict[int, str] = {}
        self.id2relation: Dict[int, str] = {}

        # 预定义因果关系类型
        self.causal_relations = [
            "causes",           # 直接因果
            "enables",          # 使能
            "prevents",         # 阻止
            "leads_to",         # 导致
            "results_in",       # 结果
            "is_caused_by",     # 被...引起
            "happens_before",   # 时序在前
            "happens_after",    # 时序在后
        ]

        for rel in self.causal_relations:
            self._add_relation(rel)

    def _add_entity(self, entity_text: str) -> int:
        """添加实体"""
        if entity_text not in self.entities:
            entity_id = len(self.entities)
            self.entities[entity_text] = entity_id
            self.id2entity[entity_id] = entity_text
        return self.entities[entity_text]

    def _add_relation(self, relation_type: str) -> int:
        """添加关系"""
        if relation_type not in self.relations:
            relation_id = len(self.relations)
            self.relations[relation_type] = relation_id
            self.id2relation[relation_id] = relation_type
        return self.relations[relation_type]

    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str
    ):
        """添加三元组"""
        h_id = self._add_entity(head)
        r_id = self._add_relation(relation)
        t_id = self._add_entity(tail)
        self.triples.append((h_id, r_id, t_id))

    def add_causal_relation(self, cause: str, effect: str):
        """添加因果关系"""
        self.add_triple(cause, "causes", effect)
        self.add_triple(effect, "is_caused_by", cause)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def get_triples_tensor(self) -> torch.Tensor:
        """返回所有三元组的张量"""
        return torch.tensor(self.triples, dtype=torch.long)

    def save(self, path: str):
        """保存知识图谱"""
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "triples": self.triples
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载知识图谱"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.relations = data["relations"]
        self.triples = [tuple(t) for t in data["triples"]]

        self.id2entity = {v: k for k, v in self.entities.items()}
        self.id2relation = {v: k for k, v in self.relations.items()}


class KGDataset(Dataset):
    """KG训练数据集"""

    def __init__(
        self,
        triples: torch.Tensor,
        num_entities: int,
        negative_samples: int = 10
    ):
        self.triples = triples
        self.num_entities = num_entities
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.triples) * self.negative_samples

    def __getitem__(self, idx):
        triple_idx = idx // self.negative_samples
        h, r, t = self.triples[triple_idx]

        # 随机替换头或尾生成负样本
        if torch.rand(1).item() < 0.5:
            neg_h = torch.randint(0, self.num_entities, (1,)).item()
            neg_t = t.item()
        else:
            neg_h = h.item()
            neg_t = torch.randint(0, self.num_entities, (1,)).item()

        return {
            "pos_head": h,
            "pos_relation": r,
            "pos_tail": t,
            "neg_head": torch.tensor(neg_h),
            "neg_tail": torch.tensor(neg_t)
        }


class KGEmbeddingTrainer:
    """KG Embedding 训练器"""

    def __init__(
        self,
        kg: CausalKnowledgeGraph,
        model_type: str = "TransE",
        config: Optional[KGEConfig] = None
    ):
        self.kg = kg
        self.config = config or KGEConfig()
        self.model_type = model_type

        # 创建模型
        self.model = self._create_model(model_type)
        self.model.to(self.config.device)

    def _create_model(self, model_type: str) -> nn.Module:
        """创建模型"""
        models = {
            "TransE": TransE,
            "ComplEx": ComplEx,
            "RotatE": RotatE
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type](
            num_entities=self.kg.num_entities,
            num_relations=self.kg.num_relations,
            embedding_dim=self.config.embedding_dim
        )

    def train(self) -> Dict[str, float]:
        """训练模型"""
        triples = self.kg.get_triples_tensor()
        dataset = KGDataset(
            triples,
            self.kg.num_entities,
            self.config.negative_samples
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        print(f"Training {self.model_type} on {len(triples)} triples...")

        losses = []
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                loss = self.model.loss(
                    batch["pos_head"].to(self.config.device),
                    batch["pos_relation"].to(self.config.device),
                    batch["pos_tail"].to(self.config.device),
                    batch["neg_head"].to(self.config.device),
                    batch["neg_tail"].to(self.config.device)
                )

                loss.backward()
                optimizer.step()

                if hasattr(self.model, '_normalize_entities'):
                    self.model._normalize_entities()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")

        return {"final_loss": losses[-1], "losses": losses}

    def get_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """获取所有实体嵌入"""
        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for entity_text, entity_id in self.kg.entities.items():
                emb = self.model.get_entity_embedding(entity_id)
                embeddings[entity_text] = emb.cpu().numpy()

        return embeddings

    def save_embeddings(self, path: str):
        """保存嵌入"""
        embeddings = self.get_entity_embeddings()
        np.savez(path, **embeddings)
        print(f"Embeddings saved to {path}")

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "config": self.config,
            "kg_entities": self.kg.entities,
            "kg_relations": self.kg.relations
        }, path)
        print(f"Model saved to {path}")


def get_kg_embedding_model(model_type: str = "TransE") -> type:
    """获取KG embedding模型类"""
    models = {
        "TransE": TransE,
        "ComplEx": ComplEx,
        "RotatE": RotatE
    }
    return models.get(model_type, TransE)


if __name__ == "__main__":
    # 测试代码
    print("Testing KG Embedding Module...")

    # 创建测试知识图谱
    kg = CausalKnowledgeGraph()
    kg.add_causal_relation("Economic recession", "Unemployment rises")
    kg.add_causal_relation("Interest rate cut", "Stock market rises")
    kg.add_causal_relation("Trade war", "Economic uncertainty")
    kg.add_causal_relation("Government stimulus", "Economic recovery")

    print(f"KG: {kg.num_entities} entities, {kg.num_relations} relations")
    print(f"Triples: {len(kg.triples)}")

    # 训练TransE
    trainer = KGEmbeddingTrainer(
        kg,
        model_type="TransE",
        config=KGEConfig(embedding_dim=64, num_epochs=50)
    )
    results = trainer.train()
    print(f"Training complete. Final loss: {results['final_loss']:.4f}")

    # 获取嵌入
    embeddings = trainer.get_entity_embeddings()
    print(f"Generated embeddings for {len(embeddings)} entities")
    for entity, emb in list(embeddings.items())[:3]:
        print(f"  {entity}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
