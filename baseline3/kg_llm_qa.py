"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 3: KG-Enhanced LLM QA

将知识图谱嵌入与大语言模型结合进行因果推理

方法:
1. KG-Augmented Prompt: 用KG知识增强prompt
2. Embedding Fusion: 将KG embedding与LLM结合
3. Retrieval-Augmented: 从KG中检索相关知识

参考:
- QA-GNN: Reasoning with Language Models and Knowledge Graphs (NAACL 2021)
- KG-BERT: BERT for Knowledge Graph Completion (2019)
- DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining (NeurIPS 2022)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from baseline.evaluator import evaluate, parse_prediction
from baseline.data_loader import AERInstance, AERDataLoader


@dataclass
class KGLLMConfig:
    """KG-LLM 融合配置"""
    # LLM 配置
    llm_model: str = "gpt-4o-mini"  # 或本地模型
    llm_type: str = "openai"  # openai, anthropic, huggingface

    # KG 配置
    use_comet: bool = True
    use_kg_embedding: bool = True
    kg_embedding_dim: int = 256
    kg_model_type: str = "TransE"

    # 融合配置
    fusion_method: str = "prompt"  # prompt, embedding, retrieval
    max_kg_context_length: int = 500
    top_k_relations: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KGAugmentedPrompt:
    """
    知识图谱增强的Prompt构建器

    将KG中的知识转换为自然语言，添加到prompt中
    """

    def __init__(self, config: KGLLMConfig):
        self.config = config
        self.comet = None
        self.simple_kb = None

    def _load_knowledge_source(self):
        """加载知识来源"""
        if self.config.use_comet:
            try:
                from comet_knowledge import COMETKnowledgeGenerator
                self.comet = COMETKnowledgeGenerator()
                self.comet.load_model()
                print("COMET model loaded for knowledge augmentation")
            except Exception as e:
                print(f"COMET not available: {e}")
                self.config.use_comet = False

        if not self.config.use_comet:
            from comet_knowledge import SimpleCausalKnowledgeBase
            self.simple_kb = SimpleCausalKnowledgeBase()
            print("Using simple knowledge base")

    def get_event_knowledge(self, event: str) -> Dict[str, List[str]]:
        """获取事件的相关知识"""
        if self.comet is None and self.simple_kb is None:
            self._load_knowledge_source()

        if self.comet:
            return self.comet.generate_causal_knowledge(event)
        else:
            causes = self.simple_kb.find_potential_causes(event)
            return {"Causes": causes} if causes else {}

    def build_kg_context(
        self,
        target_event: str,
        options: Dict[str, str]
    ) -> str:
        """
        构建KG增强的上下文

        Args:
            target_event: 目标事件
            options: 候选选项

        Returns:
            知识增强的上下文字符串
        """
        context_parts = []

        # 1. 获取目标事件的因果知识
        event_knowledge = self.get_event_knowledge(target_event)

        if event_knowledge.get("Causes"):
            causes = event_knowledge["Causes"][:3]
            context_parts.append(
                f"Common causes of similar events: {'; '.join(causes)}"
            )

        if event_knowledge.get("isBefore"):
            preconditions = event_knowledge["isBefore"][:2]
            context_parts.append(
                f"Typical preconditions: {'; '.join(preconditions)}"
            )

        # 2. 为每个选项获取知识
        option_knowledge = []
        for opt_label, opt_text in options.items():
            opt_knowledge = self.get_event_knowledge(opt_text)

            if opt_knowledge.get("xEffect") or opt_knowledge.get("isAfter"):
                effects = (opt_knowledge.get("xEffect", []) +
                          opt_knowledge.get("isAfter", []))[:2]
                option_knowledge.append(
                    f"If {opt_label} happens, likely effects: {'; '.join(effects)}"
                )

        if option_knowledge:
            context_parts.append("Option analysis:")
            context_parts.extend(option_knowledge[:4])

        return "\n".join(context_parts)[:self.config.max_kg_context_length]


class EmbeddingFusion(nn.Module):
    """
    嵌入融合模块

    将KG嵌入与文本嵌入融合
    """

    def __init__(
        self,
        text_dim: int,
        kg_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 4  # 4个选项
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.kg_proj = nn.Linear(kg_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        kg_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_embedding: (batch, text_dim)
            kg_embedding: (batch, kg_dim)

        Returns:
            logits: (batch, 4)
        """
        text_hidden = self.text_proj(text_embedding)
        kg_hidden = self.kg_proj(kg_embedding)

        # 拼接融合
        fused = torch.cat([text_hidden, kg_hidden], dim=-1)
        logits = self.fusion(fused)

        return logits


class KGRetrievalAugmented:
    """
    知识图谱检索增强

    从构建的KG中检索与事件相关的三元组
    """

    def __init__(
        self,
        kg: 'CausalKnowledgeGraph',
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        self.kg = kg
        self.embeddings = embeddings

    def retrieve_related_triples(
        self,
        query_event: str,
        top_k: int = 5
    ) -> List[Tuple[str, str, str]]:
        """
        检索与查询事件相关的三元组

        Returns:
            [(head, relation, tail), ...]
        """
        # 简单的文本匹配检索
        query_words = set(query_event.lower().split())
        scored_triples = []

        for h_id, r_id, t_id in self.kg.triples:
            head = self.kg.id2entity[h_id]
            relation = self.kg.id2relation[r_id]
            tail = self.kg.id2entity[t_id]

            # 计算相似度（简单的词重叠）
            head_words = set(head.lower().split())
            tail_words = set(tail.lower().split())

            score = len(query_words & head_words) + len(query_words & tail_words)

            if score > 0:
                scored_triples.append((score, (head, relation, tail)))

        # 排序并返回top-k
        scored_triples.sort(reverse=True)
        return [triple for _, triple in scored_triples[:top_k]]

    def retrieve_with_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, str, str]]:
        """
        使用嵌入相似度检索

        Args:
            query_embedding: 查询事件的嵌入
            top_k: 返回数量
        """
        if self.embeddings is None:
            return []

        # 计算与所有实体的相似度
        similarities = []
        for entity, emb in self.embeddings.items():
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            )
            similarities.append((sim, entity))

        # 获取最相似的实体
        similarities.sort(reverse=True)
        top_entities = [entity for _, entity in similarities[:top_k]]

        # 返回包含这些实体的三元组
        results = []
        for h_id, r_id, t_id in self.kg.triples:
            head = self.kg.id2entity[h_id]
            tail = self.kg.id2entity[t_id]
            relation = self.kg.id2relation[r_id]

            if head in top_entities or tail in top_entities:
                results.append((head, relation, tail))
                if len(results) >= top_k:
                    break

        return results


class KGLLMQA:
    """
    KG增强的LLM问答系统

    支持三种融合方式:
    1. prompt: 将KG知识添加到prompt中
    2. embedding: 将KG嵌入与文本嵌入融合
    3. retrieval: 从KG中检索相关知识
    """

    def __init__(self, config: Optional[KGLLMConfig] = None):
        self.config = config or KGLLMConfig()
        self.prompt_builder = None
        self.kg = None
        self.kg_embeddings = None
        self.retriever = None
        self.llm_client = None

    def setup(
        self,
        kg: Optional['CausalKnowledgeGraph'] = None,
        kg_embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        """初始化组件"""
        # 设置KG
        if kg:
            self.kg = kg
            if kg_embeddings:
                self.kg_embeddings = kg_embeddings
                self.retriever = KGRetrievalAugmented(kg, kg_embeddings)

        # 设置prompt构建器
        if self.config.fusion_method in ["prompt", "retrieval"]:
            self.prompt_builder = KGAugmentedPrompt(self.config)

        # 设置LLM客户端
        self._setup_llm()

    def _setup_llm(self):
        """设置LLM客户端"""
        if self.config.llm_type == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
            except:
                print("OpenAI client not available")

        elif self.config.llm_type == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic()
            except:
                print("Anthropic client not available")

    def _build_prompt(
        self,
        instance: AERInstance,
        kg_context: str = ""
    ) -> Tuple[str, str]:
        """构建prompt"""
        system_prompt = """You are an expert in causal reasoning. Given an observed event and background knowledge, identify the most plausible and direct cause(s) from the options.

Use the provided knowledge graph information to help identify causal relationships.

Rules:
1. Select the option(s) that represent the DIRECT cause of the target event
2. Consider temporal order: causes happen BEFORE effects
3. Use the causal knowledge to verify your reasoning
4. You may select multiple options if multiple causes are equally direct

Output: Answer with ONLY the letter(s), separated by commas if multiple."""

        user_prompt = f"Target Event: {instance.target_event}\n\n"

        # 添加KG知识
        if kg_context:
            user_prompt += f"Causal Knowledge:\n{kg_context}\n\n"

        # 添加文档上下文
        if instance.docs:
            doc_context = self._prepare_doc_context(instance.docs)
            if doc_context:
                user_prompt += f"Context Documents:\n{doc_context}\n\n"

        # 添加选项
        user_prompt += "Options:\n"
        for opt, text in instance.options.items():
            user_prompt += f"{opt}. {text}\n"

        user_prompt += "\nAnswer:"

        return system_prompt, user_prompt

    def _prepare_doc_context(self, docs: List[Dict], max_len: int = 1500) -> str:
        """准备文档上下文"""
        parts = []
        total = 0

        for doc in docs[:5]:
            title = doc.get("title", "")
            content = doc.get("content", doc.get("summary", ""))[:400]
            text = f"[{title}] {content}" if title else content

            if total + len(text) > max_len:
                break
            parts.append(text)
            total += len(text)

        return "\n".join(parts)

    def predict(self, instance: AERInstance) -> Set[str]:
        """
        预测单个实例

        Returns:
            预测的答案集合 {"A"} 或 {"A", "B"}
        """
        # 1. 获取KG知识
        kg_context = ""
        if self.config.fusion_method == "prompt" and self.prompt_builder:
            kg_context = self.prompt_builder.build_kg_context(
                instance.target_event,
                instance.options
            )
        elif self.config.fusion_method == "retrieval" and self.retriever:
            triples = self.retriever.retrieve_related_triples(
                instance.target_event,
                top_k=self.config.top_k_relations
            )
            if triples:
                kg_context = "Related causal facts:\n"
                for h, r, t in triples:
                    kg_context += f"- {h} {r} {t}\n"

        # 2. 构建prompt
        system_prompt, user_prompt = self._build_prompt(instance, kg_context)

        # 3. 调用LLM
        if self.config.llm_type == "openai" and self.llm_client:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=50
            )
            raw_output = response.choices[0].message.content.strip()

        elif self.config.llm_type == "anthropic" and self.llm_client:
            response = self.llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=50,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw_output = response.content[0].text.strip()

        else:
            # 回退到简单启发式
            raw_output = self._heuristic_predict(instance, kg_context)

        return parse_prediction(raw_output)

    def _heuristic_predict(
        self,
        instance: AERInstance,
        kg_context: str
    ) -> str:
        """简单启发式预测（当LLM不可用时）"""
        # 基于关键词匹配
        target_words = set(instance.target_event.lower().split())
        kg_words = set(kg_context.lower().split()) if kg_context else set()

        scores = {}
        for opt, text in instance.options.items():
            opt_words = set(text.lower().split())

            # 与目标事件的重叠
            target_overlap = len(opt_words & target_words)
            # 与KG知识的重叠
            kg_overlap = len(opt_words & kg_words)

            scores[opt] = target_overlap * 2 + kg_overlap

        # 返回得分最高的选项
        best_opt = max(scores.keys(), key=lambda x: scores[x])
        return best_opt

    def evaluate(
        self,
        instances: List[AERInstance],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        评估模型

        Args:
            instances: 测试实例列表
            output_path: 结果保存路径
        """
        predictions = []
        goldens = []
        details = []

        print(f"Evaluating on {len(instances)} instances...")

        for instance in tqdm(instances):
            pred = self.predict(instance)
            predictions.append(pred)

            if instance.golden_answer:
                goldens.append(set(instance.golden_answer))

            details.append({
                "id": instance.id,
                "target_event": instance.target_event,
                "prediction": list(pred),
                "golden": instance.golden_answer
            })

        # 计算指标
        if goldens:
            results = evaluate(predictions, goldens)
            print("\n" + "=" * 50)
            print("KG-LLM QA Evaluation Results")
            print("=" * 50)
            print(f"Fusion method: {self.config.fusion_method}")
            print(f"LLM: {self.config.llm_model}")
            print(f"Score: {results['score']:.4f}")
            print(f"Exact Match: {results['exact_match_rate']:.4f}")
            print(f"Partial Match: {results['partial_match_rate']:.4f}")
        else:
            results = {"note": "No labels available"}

        # 保存结果
        if output_path:
            output_data = {
                "config": {
                    "fusion_method": self.config.fusion_method,
                    "llm_model": self.config.llm_model,
                    "use_comet": self.config.use_comet
                },
                "results": results,
                "predictions": details
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {output_path}")

        return results


def run_kg_llm_baseline(
    data_path: str,
    fusion_method: str = "prompt",
    llm_type: str = "openai",
    llm_model: str = "gpt-4o-mini",
    use_comet: bool = False,  # 默认关闭COMET以避免大模型下载
    output_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict:
    """
    运行KG-LLM baseline

    Args:
        data_path: 数据目录路径
        fusion_method: 融合方法 (prompt, retrieval)
        llm_type: LLM类型 (openai, anthropic)
        llm_model: 具体模型名称
        use_comet: 是否使用COMET
        output_path: 结果保存路径
        max_samples: 最大样本数
    """
    # 加载数据
    loader = AERDataLoader(data_path)
    instances = loader.load()

    if max_samples:
        instances = instances[:max_samples]

    print(f"Loaded {len(instances)} instances")

    # 配置
    config = KGLLMConfig(
        fusion_method=fusion_method,
        llm_type=llm_type,
        llm_model=llm_model,
        use_comet=use_comet
    )

    # 创建QA系统
    qa_system = KGLLMQA(config)
    qa_system.setup()

    # 评估
    return qa_system.evaluate(instances, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KG-LLM QA Baseline")
    parser.add_argument("--data-path", type=str, required=True,
                       help="数据目录路径")
    parser.add_argument("--fusion", type=str, default="prompt",
                       choices=["prompt", "retrieval"],
                       help="融合方法")
    parser.add_argument("--llm-type", type=str, default="openai",
                       choices=["openai", "anthropic"])
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use-comet", action="store_true",
                       help="使用COMET生成知识")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    run_kg_llm_baseline(
        args.data_path,
        args.fusion,
        args.llm_type,
        args.llm_model,
        args.use_comet,
        args.output,
        args.max_samples
    )
