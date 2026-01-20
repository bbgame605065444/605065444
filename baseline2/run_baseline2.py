"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 2: 统一运行脚本

支持的模型:
1. UnifiedQA (T5-based) - 生成式
2. RoBERTa/BERT/DeBERTa for Multiple Choice - 判别式

使用流程:
1. 先运行数据预处理
2. 再选择模型运行baseline
"""

import os
import argparse
from pathlib import Path

# 添加父目录
import sys
sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="AER Baseline 2: UnifiedQA / RoBERTa MCQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 数据预处理:
   python run_baseline2.py preprocess --dataset-dir ./data/semeval2026-task12-dataset

2. 运行UnifiedQA (零样本):
   python run_baseline2.py unifiedqa --data-path ./processed_data/dev/unifiedqa.jsonl

3. 运行UnifiedQA-v2 (更强):
   python run_baseline2.py unifiedqa --data-path ./processed_data/dev/unifiedqa.jsonl \\
       --model-name allenai/unifiedqa-v2-t5-base-1363200

4. 训练RoBERTa:
   python run_baseline2.py roberta --mode train \\
       --train-data ./processed_data/train/roberta_mcqa.jsonl \\
       --dev-data ./processed_data/dev/roberta_mcqa.jsonl

5. 预测RoBERTa:
   python run_baseline2.py roberta --mode predict \\
       --data-path ./processed_data/dev/roberta_mcqa.jsonl \\
       --model-name ./roberta_output

推荐模型选择:
- UnifiedQA (零样本, 快速验证): allenai/unifiedqa-t5-base
- UnifiedQA-v2 (更强): allenai/unifiedqa-v2-t5-large-1363200
- RoBERTa (需要微调): roberta-base, roberta-large
- DeBERTa (推荐微调): microsoft/deberta-v3-base
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="数据预处理")
    preprocess_parser.add_argument("--dataset-dir", type=str, required=True,
                                   help="原始数据集目录")
    preprocess_parser.add_argument("--output-dir", type=str, default="./processed_data",
                                   help="输出目录")
    preprocess_parser.add_argument("--no-context", action="store_true",
                                   help="不包含上下文文档")

    # UnifiedQA命令
    unifiedqa_parser = subparsers.add_parser("unifiedqa", help="运行UnifiedQA baseline")
    unifiedqa_parser.add_argument("--data-path", type=str, required=True,
                                  help="UnifiedQA格式数据路径")
    unifiedqa_parser.add_argument("--model-name", type=str,
                                  default="allenai/unifiedqa-t5-base",
                                  help="模型名称")
    unifiedqa_parser.add_argument("--output", type=str, default=None,
                                  help="结果保存路径")
    unifiedqa_parser.add_argument("--batch-size", type=int, default=8)

    # RoBERTa命令
    roberta_parser = subparsers.add_parser("roberta", help="运行RoBERTa MCQA baseline")
    roberta_parser.add_argument("--mode", type=str, choices=["train", "predict"],
                                default="predict")
    roberta_parser.add_argument("--data-path", type=str,
                                help="预测数据路径")
    roberta_parser.add_argument("--train-data", type=str,
                                help="训练数据路径")
    roberta_parser.add_argument("--dev-data", type=str,
                                help="验证数据路径")
    roberta_parser.add_argument("--model-name", type=str, default="roberta-base",
                                help="模型名称或已训练模型路径")
    roberta_parser.add_argument("--output", type=str, default=None,
                                help="结果保存路径")
    roberta_parser.add_argument("--output-dir", type=str, default="./roberta_output",
                                help="模型保存目录")
    roberta_parser.add_argument("--batch-size", type=int, default=4)
    roberta_parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()

    if args.command == "preprocess":
        from preprocessing import preprocess_all
        preprocess_all(
            args.dataset_dir,
            args.output_dir,
            include_context=not args.no_context
        )

    elif args.command == "unifiedqa":
        from unifiedqa_baseline import run_unifiedqa_baseline
        run_unifiedqa_baseline(
            args.data_path,
            args.model_name,
            args.output,
            args.batch_size
        )

    elif args.command == "roberta":
        from roberta_mcqa_baseline import run_roberta_baseline
        run_roberta_baseline(
            args.mode,
            args.data_path,
            args.model_name,
            args.output,
            args.train_data,
            args.dev_data,
            args.output_dir,
            args.batch_size,
            args.epochs
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
