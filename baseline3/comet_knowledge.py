"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 3: COMET-ATOMIC 知识增强模块

使用 COMET (Commonsense Transformers) 生成事件的因果知识

COMET 可以生成多种类型的推理:
- xNeed: 在事件发生前，PersonX需要什么
- xIntent: PersonX的意图是什么
- xWant: 事件后PersonX想要什么
- oWant: 事件后其他人想要什么
- xEffect: 事件对PersonX的影响
- oEffect: 事件对其他人的影响
- xReact: PersonX的情感反应
- oReact: 其他人的情感反应
- Causes: 导致这个事件的原因 (最相关!)
- isAfter: 这个事件之后会发生什么
- isBefore: 这个事件之前发生了什么

参考:
- COMET-ATOMIC 2020: https://github.com/allenai/comet-atomic-2020
- ATOMIC 2020: https://arxiv.org/abs/2010.05953
"""

import os
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class COMETConfig:
    """COMET 配置"""
    model_name: str = "mismayil/comet-bart-ai2"  # HuggingFace上的COMET模型
    max_length: int = 64
    num_beams: int = 5
    num_return_sequences: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ATOMIC 关系类型及其描述
ATOMIC_RELATIONS = {
    # 因果相关 (最重要)
    "Causes": "What could cause this event?",
    "xEffect": "As a result, PersonX will",
    "oEffect": "As a result, others will",
    "isAfter": "This happens after",
    "isBefore": "This happens before",

    # 意图和需求
    "xNeed": "Before, PersonX needed",
    "xIntent": "PersonX's intent was",
    "xWant": "After, PersonX wants",
    "oWant": "After, others want",

    # 情感反应
    "xReact": "PersonX feels",
    "oReact": "Others feel",

    # 属性
    "xAttr": "PersonX is seen as",
}


class COMETKnowledgeGenerator:
    """
    使用 COMET 生成常识知识

    COMET 是在 ATOMIC 知识图谱上训练的生成模型，
    可以对任意事件生成结构化的常识推理。
    """

    def __init__(self, config: Optional[COMETConfig] = None):
        self.config = config or COMETConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载 COMET 模型"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"Loading COMET model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        self.model.eval()
        print(f"Model loaded on {self.config.device}")

    def generate(
        self,
        event: str,
        relation: str,
        num_sequences: int = None
    ) -> List[str]:
        """
        为事件生成指定关系的推理结果

        Args:
            event: 输入事件描述
            relation: ATOMIC关系类型 (如 "Causes", "xEffect")
            num_sequences: 生成的结果数量

        Returns:
            生成的推理结果列表
        """
        if self.model is None:
            self.load_model()

        num_sequences = num_sequences or self.config.num_return_sequences

        # 构造输入: "{event} {relation} [GEN]"
        input_text = f"{event} {relation} [GEN]"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                num_return_sequences=num_sequences,
                early_stopping=True,
                do_sample=False
            )

        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            # 清理输出
            decoded = decoded.strip()
            if decoded and decoded not in ["none", "None", ""]:
                results.append(decoded)

        return results

    def generate_causal_knowledge(
        self,
        event: str,
        include_effects: bool = True
    ) -> Dict[str, List[str]]:
        """
        生成事件的因果相关知识

        Args:
            event: 事件描述
            include_effects: 是否包含影响/结果

        Returns:
            {relation: [inferences]} 字典
        """
        causal_relations = ["Causes", "isBefore"]
        if include_effects:
            causal_relations.extend(["xEffect", "oEffect", "isAfter"])

        knowledge = {}
        for relation in causal_relations:
            inferences = self.generate(event, relation)
            if inferences:
                knowledge[relation] = inferences

        return knowledge

    def generate_full_knowledge(self, event: str) -> Dict[str, List[str]]:
        """生成所有类型的常识知识"""
        knowledge = {}
        for relation in ATOMIC_RELATIONS.keys():
            inferences = self.generate(event, relation)
            if inferences:
                knowledge[relation] = inferences
        return knowledge


class SimpleCausalKnowledgeBase:
    """
    简化版因果知识库

    当 COMET 模型不可用时，使用基于规则的简单知识库
    """

    def __init__(self):
        # 预定义的因果模板
        self.causal_templates = {
            # 经济领域
            "economic": [
                ("interest rate cut", "stock market rises"),
                ("interest rate hike", "borrowing costs increase"),
                ("unemployment rises", "consumer spending decreases"),
                ("inflation increases", "central bank raises interest rates"),
                ("trade deficit grows", "currency weakens"),
                ("GDP growth", "employment increases"),
                ("recession", "government stimulus"),
                ("tax cut", "consumer spending increases"),
            ],
            # 政治领域
            "political": [
                ("election", "policy change"),
                ("diplomatic tensions", "trade restrictions"),
                ("sanctions imposed", "economic pressure"),
                ("peace agreement", "trade normalization"),
                ("government shutdown", "public services disrupted"),
            ],
            # 自然灾害
            "disaster": [
                ("earthquake", "infrastructure damage"),
                ("flood", "crop destruction"),
                ("hurricane", "power outage"),
                ("drought", "water shortage"),
                ("wildfire", "air quality decline"),
            ],
            # 技术领域
            "technology": [
                ("new product launch", "stock price changes"),
                ("data breach", "user trust decline"),
                ("regulation change", "business model adaptation"),
                ("AI breakthrough", "industry disruption"),
            ],
        }

        self.keyword_to_domain = {
            "stock": "economic", "market": "economic", "price": "economic",
            "interest": "economic", "inflation": "economic", "gdp": "economic",
            "unemployment": "economic", "recession": "economic", "trade": "economic",
            "election": "political", "government": "political", "policy": "political",
            "sanction": "political", "diplomatic": "political",
            "earthquake": "disaster", "flood": "disaster", "hurricane": "disaster",
            "drought": "disaster", "wildfire": "disaster",
            "tech": "technology", "ai": "technology", "data": "technology",
            "crypto": "technology", "bitcoin": "technology",
        }

    def find_potential_causes(self, event: str) -> List[str]:
        """查找可能的原因"""
        event_lower = event.lower()
        causes = []

        # 确定领域
        domain = None
        for keyword, dom in self.keyword_to_domain.items():
            if keyword in event_lower:
                domain = dom
                break

        if domain:
            for cause, effect in self.causal_templates.get(domain, []):
                # 如果effect与event相似，cause可能是原因
                if any(word in event_lower for word in effect.split()):
                    causes.append(cause)
                # 如果cause与event相似，effect可能是结果
                if any(word in event_lower for word in cause.split()):
                    causes.append(f"Before this: {cause}")

        return causes[:5]  # 最多返回5个


class KnowledgeEnhancedEvent:
    """知识增强的事件表示"""

    def __init__(
        self,
        event_text: str,
        comet_knowledge: Optional[Dict[str, List[str]]] = None,
        kg_embedding: Optional[torch.Tensor] = None
    ):
        self.event_text = event_text
        self.comet_knowledge = comet_knowledge or {}
        self.kg_embedding = kg_embedding

    def get_causes(self) -> List[str]:
        """获取可能的原因"""
        return self.comet_knowledge.get("Causes", [])

    def get_effects(self) -> List[str]:
        """获取可能的影响"""
        effects = []
        effects.extend(self.comet_knowledge.get("xEffect", []))
        effects.extend(self.comet_knowledge.get("oEffect", []))
        effects.extend(self.comet_knowledge.get("isAfter", []))
        return effects

    def get_preconditions(self) -> List[str]:
        """获取前置条件"""
        preconditions = []
        preconditions.extend(self.comet_knowledge.get("xNeed", []))
        preconditions.extend(self.comet_knowledge.get("isBefore", []))
        return preconditions

    def to_context_string(self) -> str:
        """转换为上下文字符串，用于LLM输入"""
        parts = [f"Event: {self.event_text}"]

        if self.get_causes():
            parts.append(f"Possible causes: {'; '.join(self.get_causes()[:3])}")

        if self.get_preconditions():
            parts.append(f"Preconditions: {'; '.join(self.get_preconditions()[:3])}")

        if self.get_effects():
            parts.append(f"Possible effects: {'; '.join(self.get_effects()[:3])}")

        return "\n".join(parts)


def build_event_knowledge_graph(
    events: List[str],
    use_comet: bool = True,
    comet_config: Optional[COMETConfig] = None
) -> Tuple[Dict[str, KnowledgeEnhancedEvent], 'CausalKnowledgeGraph']:
    """
    为事件列表构建知识图谱

    Args:
        events: 事件文本列表
        use_comet: 是否使用COMET生成知识
        comet_config: COMET配置

    Returns:
        (enhanced_events, knowledge_graph)
    """
    from kg_embedding import CausalKnowledgeGraph

    # 初始化
    kg = CausalKnowledgeGraph()
    enhanced_events = {}

    if use_comet:
        try:
            comet = COMETKnowledgeGenerator(comet_config)
            comet.load_model()
        except Exception as e:
            print(f"Warning: Could not load COMET model: {e}")
            print("Falling back to simple knowledge base")
            use_comet = False

    if not use_comet:
        simple_kb = SimpleCausalKnowledgeBase()

    print(f"Building knowledge graph for {len(events)} events...")

    for event in tqdm(events):
        if use_comet:
            knowledge = comet.generate_causal_knowledge(event)
        else:
            causes = simple_kb.find_potential_causes(event)
            knowledge = {"Causes": causes} if causes else {}

        # 创建增强事件
        enhanced = KnowledgeEnhancedEvent(event, knowledge)
        enhanced_events[event] = enhanced

        # 添加到知识图谱
        kg._add_entity(event)
        for cause in enhanced.get_causes():
            kg.add_causal_relation(cause, event)

        for effect in enhanced.get_effects():
            kg.add_causal_relation(event, effect)

    print(f"Knowledge graph: {kg.num_entities} entities, {len(kg.triples)} triples")
    return enhanced_events, kg


if __name__ == "__main__":
    # 测试代码
    print("Testing COMET Knowledge Module...")

    # 测试简单知识库
    kb = SimpleCausalKnowledgeBase()
    test_events = [
        "Stock market crashes",
        "Government announces stimulus package",
        "Cryptocurrency prices surge",
        "Major earthquake hits region",
    ]

    print("\n=== Simple Knowledge Base ===")
    for event in test_events:
        causes = kb.find_potential_causes(event)
        print(f"\nEvent: {event}")
        print(f"  Possible causes: {causes}")

    # 测试COMET (如果可用)
    print("\n=== COMET Knowledge Generator ===")
    try:
        comet = COMETKnowledgeGenerator()
        comet.load_model()

        event = "The stock market crashed"
        print(f"\nEvent: {event}")

        for relation in ["Causes", "xEffect", "isBefore"]:
            inferences = comet.generate(event, relation, num_sequences=3)
            print(f"  {relation}: {inferences}")

    except Exception as e:
        print(f"COMET not available: {e}")
        print("To use COMET, install: pip install transformers")
