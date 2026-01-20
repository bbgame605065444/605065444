"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline运行脚本
"""
import argparse
import json
from pathlib import Path
from typing import List, Set
from tqdm import tqdm

from data_loader import AERDataLoader, AERInstance, download_dataset
from evaluator import evaluate, parse_prediction
from models import get_model, AERPrompt


def prepare_context(instance: AERInstance, max_docs: int = 5) -> str:
    """准备上下文文档"""
    if not instance.docs:
        return ""

    context_parts = []
    for i, doc in enumerate(instance.docs[:max_docs]):
        title = doc.get("title", f"Document {i+1}")
        content = doc.get("content", doc.get("summary", ""))

        # 限制单个文档长度
        max_doc_len = 800
        if len(content) > max_doc_len:
            content = content[:max_doc_len] + "..."

        context_parts.append(f"[{title}]\n{content}")

    return "\n\n".join(context_parts)


def run_baseline(
    model_type: str,
    model_name: str = None,
    data_split: str = "dev",
    use_context: bool = True,
    max_samples: int = None,
    output_file: str = None,
    **model_kwargs
):
    """
    运行baseline实验

    Args:
        model_type: 模型类型 (openai, anthropic, huggingface, ollama, vllm)
        model_name: 具体模型名称
        data_split: 数据集划分 (train, dev, test)
        use_context: 是否使用上下文文档
        max_samples: 最大样本数（用于快速测试）
        output_file: 输出文件路径
    """
    # 1. 下载并加载数据
    print("=" * 60)
    print("SemEval 2026 Task 12: Abductive Event Reasoning Baseline")
    print("=" * 60)

    dataset_path = download_dataset()
    data_dir = dataset_path / f"{data_split}_data"

    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")

    loader = AERDataLoader(data_dir)
    instances = loader.load()

    if max_samples:
        instances = instances[:max_samples]

    print(f"\n数据集: {data_split}")
    print(f"样本数: {len(instances)}")
    print(f"使用上下文: {use_context}")

    # 2. 初始化模型
    print(f"\n初始化模型: {model_type}" + (f" ({model_name})" if model_name else ""))
    if model_name:
        model_kwargs["model_name"] = model_name
    model = get_model(model_type, **model_kwargs)

    # 3. 运行推理
    print("\n开始推理...")
    predictions = []
    raw_outputs = []

    for instance in tqdm(instances, desc="Processing"):
        # 准备prompt
        context = prepare_context(instance) if use_context else None
        prompt = AERPrompt(
            target_event=instance.target_event,
            options=instance.options,
            context=context
        )

        # 获取预测
        try:
            raw_output = model.predict(prompt)
            pred_set = parse_prediction(raw_output)
        except Exception as e:
            print(f"\n警告: 实例 {instance.id} 预测失败: {e}")
            raw_output = ""
            pred_set = set()

        predictions.append(pred_set)
        raw_outputs.append(raw_output)

    # 4. 评估
    goldens = [set(inst.golden_answer) if inst.golden_answer else set() for inst in instances]
    results = evaluate(predictions, goldens)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"官方分数 (Score): {results['score']:.4f}")
    print(f"完全匹配率: {results['exact_match_rate']:.4f} ({results['exact_match']}/{results['total']})")
    print(f"部分匹配率: {results['partial_match_rate']:.4f} ({results['partial_match']}/{results['total']})")
    print(f"错误率: {results['wrong_rate']:.4f} ({results['wrong']}/{results['total']})")

    # 5. 保存结果
    if output_file:
        output_data = {
            "config": {
                "model_type": model_type,
                "model_name": model_name,
                "data_split": data_split,
                "use_context": use_context,
                "num_samples": len(instances)
            },
            "results": results,
            "predictions": [
                {
                    "id": inst.id,
                    "target_event": inst.target_event,
                    "golden": list(inst.golden_answer) if inst.golden_answer else [],
                    "prediction": list(pred),
                    "raw_output": raw_out
                }
                for inst, pred, raw_out in zip(instances, predictions, raw_outputs)
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AER Baseline Runner")
    parser.add_argument("--model-type", type=str, default="openai",
                       choices=["openai", "anthropic", "huggingface", "ollama", "vllm"],
                       help="模型类型")
    parser.add_argument("--model-name", type=str, default=None,
                       help="具体模型名称 (如 gpt-4o, claude-3-5-sonnet-20241022, llama3.1:8b)")
    parser.add_argument("--data-split", type=str, default="dev",
                       choices=["train", "dev", "test"],
                       help="数据集划分")
    parser.add_argument("--no-context", action="store_true",
                       help="不使用上下文文档")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大样本数（用于快速测试）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")

    args = parser.parse_args()

    run_baseline(
        model_type=args.model_type,
        model_name=args.model_name,
        data_split=args.data_split,
        use_context=not args.no_context,
        max_samples=args.max_samples,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
