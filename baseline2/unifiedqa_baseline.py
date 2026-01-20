"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 2a: UnifiedQA (T5-based)

UnifiedQA是AllenAI开发的统一问答模型，使用T5架构，
支持多种QA格式：抽取式、多选题、是否题等。

参考:
- 论文: https://arxiv.org/abs/2005.00700
- GitHub: https://github.com/allenai/unifiedqa
- HuggingFace: https://huggingface.co/allenai/unifiedqa-t5-base
"""

import os
import json
import torch
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from dataclasses import dataclass

# 添加父目录
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from baseline.evaluator import evaluate, parse_prediction


@dataclass
class UnifiedQAConfig:
    """UnifiedQA配置"""
    model_name: str = "allenai/unifiedqa-t5-base"  # 可选: small, base, large, 3b
    max_input_length: int = 512
    max_output_length: int = 32
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class UnifiedQABaseline:
    """
    UnifiedQA Baseline

    可用模型 (按大小排序):
    - allenai/unifiedqa-t5-small  (~60M参数, ~250MB)
    - allenai/unifiedqa-t5-base   (~220M参数, ~900MB)
    - allenai/unifiedqa-t5-large  (~770M参数, ~3GB)
    - allenai/unifiedqa-t5-3b     (~3B参数, ~11GB)

    UnifiedQA-v2 (更强):
    - allenai/unifiedqa-v2-t5-small-1363200
    - allenai/unifiedqa-v2-t5-base-1363200
    - allenai/unifiedqa-v2-t5-large-1363200
    """

    def __init__(self, config: Optional[UnifiedQAConfig] = None):
        self.config = config or UnifiedQAConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        print(f"加载模型: {self.config.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        self.model.eval()
        print(f"模型已加载到 {self.config.device}")

    def predict_single(self, input_text: str) -> str:
        """单条预测"""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.max_input_length,
            truncation=True,
            padding=True
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_output_length,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict_batch(self, input_texts: List[str]) -> List[str]:
        """批量预测"""
        if self.model is None:
            self.load_model()

        results = []

        for i in range(0, len(input_texts), self.config.batch_size):
            batch = input_texts[i:i + self.config.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.config.max_input_length,
                truncation=True,
                padding=True
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_output_length,
                    num_beams=4,
                    early_stopping=True
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        return results

    def run_evaluation(
        self,
        data_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        运行评估

        Args:
            data_path: UnifiedQA格式的JSONL文件路径
            output_path: 结果保存路径
        """
        # 加载数据
        inputs = []
        goldens = []
        ids = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                inputs.append(data["input"])
                if "output" in data:
                    golden = set(ans.strip() for ans in data["output"].split(","))
                    goldens.append(golden)
                if "id" in data:
                    ids.append(data["id"])

        print(f"加载了 {len(inputs)} 个样本")

        # 预测
        print("开始预测...")
        raw_outputs = []
        predictions = []

        for i in tqdm(range(0, len(inputs), self.config.batch_size)):
            batch = inputs[i:i + self.config.batch_size]
            batch_outputs = self.predict_batch(batch)
            raw_outputs.extend(batch_outputs)
            predictions.extend([parse_prediction(out) for out in batch_outputs])

        # 评估
        if goldens:
            results = evaluate(predictions, goldens)
            print("\n" + "=" * 50)
            print("UnifiedQA 评估结果")
            print("=" * 50)
            print(f"模型: {self.config.model_name}")
            print(f"官方分数: {results['score']:.4f}")
            print(f"完全匹配率: {results['exact_match_rate']:.4f}")
            print(f"部分匹配率: {results['partial_match_rate']:.4f}")
        else:
            results = {"note": "测试集无标签"}

        # 保存结果
        if output_path:
            output_data = {
                "config": {
                    "model_name": self.config.model_name,
                    "data_path": data_path
                },
                "results": results,
                "predictions": [
                    {
                        "id": ids[i] if ids else str(i),
                        "input": inputs[i][:200] + "..." if len(inputs[i]) > 200 else inputs[i],
                        "raw_output": raw_outputs[i],
                        "prediction": list(predictions[i]),
                        "golden": list(goldens[i]) if goldens else None
                    }
                    for i in range(len(inputs))
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_path}")

        return results


def run_unifiedqa_baseline(
    data_path: str,
    model_name: str = "allenai/unifiedqa-t5-base",
    output_path: Optional[str] = None,
    batch_size: int = 8
) -> Dict:
    """
    运行UnifiedQA baseline的便捷函数

    Args:
        data_path: 预处理后的UnifiedQA格式数据路径
        model_name: 模型名称
        output_path: 结果保存路径
        batch_size: 批次大小

    Returns:
        评估结果字典
    """
    config = UnifiedQAConfig(
        model_name=model_name,
        batch_size=batch_size
    )

    baseline = UnifiedQABaseline(config)
    return baseline.run_evaluation(data_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UnifiedQA Baseline")
    parser.add_argument("--data-path", type=str, required=True,
                       help="UnifiedQA格式的JSONL数据路径")
    parser.add_argument("--model-name", type=str,
                       default="allenai/unifiedqa-t5-base",
                       help="模型名称")
    parser.add_argument("--output", type=str, default=None,
                       help="结果保存路径")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="批次大小")

    args = parser.parse_args()

    run_unifiedqa_baseline(
        args.data_path,
        args.model_name,
        args.output,
        args.batch_size
    )
