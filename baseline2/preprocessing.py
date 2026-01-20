"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 2: UnifiedQA (T5-based) + RoBERTa Multiple Choice

这个模块实现了两个经典的baseline方法：
1. UnifiedQA - AllenAI的统一问答模型 (text-to-text)
2. RoBERTa for Multiple Choice - HuggingFace官方多选题分类

参考论文：
- UnifiedQA: Crossing Format Boundaries with a Single QA System (EMNLP 2020)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)
"""

import os
import json
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 添加父目录到路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from baseline.data_loader import AERInstance, AERDataLoader


@dataclass
class ProcessedInstance:
    """预处理后的实例，兼容多种模型"""
    id: str
    # UnifiedQA 格式
    unifiedqa_input: str  # "question \n (A) opt1 (B) opt2..."
    # RoBERTa/BERT Multiple Choice 格式
    question: str
    choices: List[str]  # ["opt1", "opt2", "opt3", "opt4"]
    context: Optional[str] = None
    # 标签
    label: Optional[List[int]] = None  # [0] or [0, 1] for multi-label


class AERPreprocessor:
    """
    AER数据预处理器
    将原始数据转换为兼容不同baseline模型的格式
    """

    def __init__(
        self,
        max_context_length: int = 2048,
        max_doc_count: int = 5,
        include_context: bool = True
    ):
        self.max_context_length = max_context_length
        self.max_doc_count = max_doc_count
        self.include_context = include_context

    def _prepare_context(self, instance: AERInstance) -> str:
        """准备上下文文档"""
        if not instance.docs or not self.include_context:
            return ""

        context_parts = []
        total_len = 0

        for doc in instance.docs[:self.max_doc_count]:
            title = doc.get("title", "")
            content = doc.get("content", doc.get("summary", ""))

            # 截断单个文档
            max_per_doc = self.max_context_length // self.max_doc_count
            if len(content) > max_per_doc:
                content = content[:max_per_doc] + "..."

            doc_text = f"[{title}] {content}" if title else content

            if total_len + len(doc_text) > self.max_context_length:
                break

            context_parts.append(doc_text)
            total_len += len(doc_text)

        return " ".join(context_parts)

    def _format_unifiedqa(self, instance: AERInstance, context: str) -> str:
        """
        格式化为UnifiedQA输入格式

        UnifiedQA格式: "question \\n (A) choice1 (B) choice2 (C) choice3 (D) choice4"
        如果有context: "context \\n question \\n (A) choice1..."

        注意: UnifiedQA使用 \\n 作为分隔符
        """
        question = f"What is the most plausible cause of the following event: {instance.target_event}"

        # 格式化选项
        options_str = " ".join([
            f"({opt}) {text}"
            for opt, text in instance.options.items()
        ])

        if context:
            return f"{context} \\n {question} \\n {options_str}"
        else:
            return f"{question} \\n {options_str}"

    def _get_labels(self, instance: AERInstance) -> Optional[List[int]]:
        """转换答案为索引"""
        if not instance.golden_answer:
            return None

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        return [label_map[ans] for ans in instance.golden_answer if ans in label_map]

    def process_instance(self, instance: AERInstance) -> ProcessedInstance:
        """处理单个实例"""
        context = self._prepare_context(instance)

        # 构建问题
        question = f"What is the most plausible cause of: {instance.target_event}"
        if context:
            question = f"Context: {context}\n\nQuestion: {question}"

        return ProcessedInstance(
            id=instance.id,
            unifiedqa_input=self._format_unifiedqa(instance, context),
            question=question,
            choices=list(instance.options.values()),
            context=context if context else None,
            label=self._get_labels(instance)
        )

    def process_dataset(self, instances: List[AERInstance]) -> List[ProcessedInstance]:
        """处理整个数据集"""
        return [self.process_instance(inst) for inst in instances]

    def save_for_unifiedqa(self, instances: List[ProcessedInstance], output_path: str):
        """
        保存为UnifiedQA兼容格式 (JSONL)

        每行格式: {"input": "...", "output": "A" or "A, B"}
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for inst in instances:
                data = {"input": inst.unifiedqa_input}
                if inst.label is not None:
                    # 转换回字母
                    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                    data["output"] = ", ".join([label_map[l] for l in inst.label])
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"UnifiedQA格式数据已保存到: {output_path}")

    def save_for_roberta_mcqa(self, instances: List[ProcessedInstance], output_path: str):
        """
        保存为RoBERTa Multiple Choice兼容格式 (JSONL)

        格式: HuggingFace SWAG-style
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for inst in instances:
                data = {
                    "id": inst.id,
                    "sent1": inst.question,  # 问题/前提
                    "sent2": "",  # SWAG格式中的第二句（这里留空）
                    "ending0": inst.choices[0],
                    "ending1": inst.choices[1],
                    "ending2": inst.choices[2],
                    "ending3": inst.choices[3],
                }
                if inst.label is not None:
                    # 多标签情况取第一个作为主标签
                    data["label"] = inst.label[0]
                    data["labels_all"] = inst.label  # 保存所有正确答案
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"RoBERTa MCQA格式数据已保存到: {output_path}")

    def save_for_huggingface(self, instances: List[ProcessedInstance], output_dir: str):
        """
        保存为HuggingFace datasets兼容格式

        创建 train.json, dev.json 等文件
        """
        os.makedirs(output_dir, exist_ok=True)

        data_list = []
        for inst in instances:
            item = {
                "id": inst.id,
                "question": inst.question,
                "choices": inst.choices,
            }
            if inst.label is not None:
                item["label"] = inst.label[0]  # 主标签
                item["labels_multi"] = inst.label  # 多标签
            data_list.append(item)

        output_path = os.path.join(output_dir, "data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"HuggingFace格式数据已保存到: {output_path}")


def preprocess_all(
    dataset_dir: str,
    output_dir: str,
    include_context: bool = True
):
    """
    预处理整个数据集，生成多种格式

    Args:
        dataset_dir: 原始数据集目录
        output_dir: 输出目录
    """
    preprocessor = AERPreprocessor(include_context=include_context)

    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(dataset_dir, f"{split}_data")
        if not os.path.exists(split_dir):
            print(f"跳过不存在的目录: {split_dir}")
            continue

        print(f"\n处理 {split} 数据集...")

        # 加载数据
        loader = AERDataLoader(split_dir)
        instances = loader.load()
        print(f"  加载了 {len(instances)} 个样本")

        # 预处理
        processed = preprocessor.process_dataset(instances)

        # 创建输出目录
        split_output = os.path.join(output_dir, split)
        os.makedirs(split_output, exist_ok=True)

        # 保存多种格式
        preprocessor.save_for_unifiedqa(
            processed,
            os.path.join(split_output, "unifiedqa.jsonl")
        )
        preprocessor.save_for_roberta_mcqa(
            processed,
            os.path.join(split_output, "roberta_mcqa.jsonl")
        )
        preprocessor.save_for_huggingface(
            processed,
            split_output
        )

    print(f"\n所有数据已预处理完成，保存到: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AER数据预处理")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="原始数据集目录")
    parser.add_argument("--output-dir", type=str, default="./processed_data",
                       help="输出目录")
    parser.add_argument("--no-context", action="store_true",
                       help="不包含上下文文档")

    args = parser.parse_args()

    preprocess_all(
        args.dataset_dir,
        args.output_dir,
        include_context=not args.no_context
    )
