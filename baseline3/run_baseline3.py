"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 3: KG-Enhanced LLM QA 统一运行脚本

使用方法:

1. 构建知识图谱并训练embedding:
   python run_baseline3.py build-kg --data-path ./data/semeval2026-task12-dataset/train_data

2. 运行KG-Augmented Prompt baseline:
   python run_baseline3.py qa --data-path ./data/semeval2026-task12-dataset/dev_data \\
       --fusion prompt --llm-model gpt-4o-mini

3. 运行KG-Retrieval baseline:
   python run_baseline3.py qa --data-path ./data/semeval2026-task12-dataset/dev_data \\
       --fusion retrieval --kg-path ./kg_output
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def build_knowledge_graph(args):
    """构建知识图谱并训练embedding"""
    from baseline.data_loader import AERDataLoader
    from kg_embedding import CausalKnowledgeGraph, KGEmbeddingTrainer, KGEConfig
    from comet_knowledge import build_event_knowledge_graph

    print("=" * 60)
    print("Building Causal Knowledge Graph")
    print("=" * 60)

    # 加载数据
    loader = AERDataLoader(args.data_path)
    instances = loader.load()

    if args.max_samples:
        instances = instances[:args.max_samples]

    print(f"Loaded {len(instances)} instances")

    # 收集所有事件
    events = set()
    for inst in instances:
        events.add(inst.target_event)
        for opt_text in inst.options.values():
            events.add(opt_text)

    events = list(events)
    print(f"Collected {len(events)} unique events")

    # 构建知识图谱
    enhanced_events, kg = build_event_knowledge_graph(
        events,
        use_comet=args.use_comet
    )

    print(f"\nKnowledge Graph Statistics:")
    print(f"  Entities: {kg.num_entities}")
    print(f"  Relations: {kg.num_relations}")
    print(f"  Triples: {len(kg.triples)}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存知识图谱
    kg_path = os.path.join(args.output_dir, "knowledge_graph.json")
    kg.save(kg_path)
    print(f"\nKnowledge graph saved to: {kg_path}")

    # 训练embedding
    if args.train_embedding and len(kg.triples) > 0:
        print(f"\nTraining {args.kg_model} embeddings...")

        config = KGEConfig(
            embedding_dim=args.embedding_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )

        trainer = KGEmbeddingTrainer(kg, args.kg_model, config)
        results = trainer.train()

        # 保存模型和embedding
        model_path = os.path.join(args.output_dir, "kg_model.pt")
        trainer.save_model(model_path)

        emb_path = os.path.join(args.output_dir, "embeddings.npz")
        trainer.save_embeddings(emb_path)

        print(f"\nTraining complete!")
        print(f"  Final loss: {results['final_loss']:.4f}")
        print(f"  Model saved: {model_path}")
        print(f"  Embeddings saved: {emb_path}")
    else:
        if len(kg.triples) == 0:
            print("\nNo triples to train on. Skipping embedding training.")
        else:
            print("\nSkipping embedding training (use --train-embedding to enable)")

    return kg


def run_qa(args):
    """运行QA评估"""
    from kg_llm_qa import run_kg_llm_baseline

    print("=" * 60)
    print("KG-Enhanced LLM QA Evaluation")
    print("=" * 60)

    results = run_kg_llm_baseline(
        data_path=args.data_path,
        fusion_method=args.fusion,
        llm_type=args.llm_type,
        llm_model=args.llm_model,
        use_comet=args.use_comet,
        output_path=args.output,
        max_samples=args.max_samples
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="KG-Enhanced LLM QA Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

1. 构建知识图谱 (使用简单知识库):
   python run_baseline3.py build-kg \\
       --data-path ./data/semeval2026-task12-dataset/train_data \\
       --output-dir ./kg_output

2. 构建知识图谱 (使用COMET, 需要GPU):
   python run_baseline3.py build-kg \\
       --data-path ./data/semeval2026-task12-dataset/train_data \\
       --output-dir ./kg_output \\
       --use-comet \\
       --train-embedding \\
       --kg-model TransE

3. 运行QA (Prompt增强):
   python run_baseline3.py qa \\
       --data-path ./data/semeval2026-task12-dataset/dev_data \\
       --fusion prompt \\
       --llm-model gpt-4o-mini

4. 运行QA (检索增强):
   python run_baseline3.py qa \\
       --data-path ./data/semeval2026-task12-dataset/dev_data \\
       --fusion retrieval \\
       --kg-path ./kg_output
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # build-kg 命令
    build_parser = subparsers.add_parser("build-kg", help="构建知识图谱")
    build_parser.add_argument("--data-path", type=str, required=True,
                              help="数据目录路径")
    build_parser.add_argument("--output-dir", type=str, default="./kg_output",
                              help="输出目录")
    build_parser.add_argument("--use-comet", action="store_true",
                              help="使用COMET生成知识 (需要~3GB显存)")
    build_parser.add_argument("--train-embedding", action="store_true",
                              help="训练KG embedding")
    build_parser.add_argument("--kg-model", type=str, default="TransE",
                              choices=["TransE", "ComplEx", "RotatE"],
                              help="KG embedding模型")
    build_parser.add_argument("--embedding-dim", type=int, default=256)
    build_parser.add_argument("--epochs", type=int, default=100)
    build_parser.add_argument("--batch-size", type=int, default=256)
    build_parser.add_argument("--max-samples", type=int, default=None)

    # qa 命令
    qa_parser = subparsers.add_parser("qa", help="运行QA评估")
    qa_parser.add_argument("--data-path", type=str, required=True,
                           help="数据目录路径")
    qa_parser.add_argument("--fusion", type=str, default="prompt",
                           choices=["prompt", "retrieval"],
                           help="融合方法")
    qa_parser.add_argument("--llm-type", type=str, default="openai",
                           choices=["openai", "anthropic"])
    qa_parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    qa_parser.add_argument("--use-comet", action="store_true")
    qa_parser.add_argument("--kg-path", type=str, default=None,
                           help="KG输出目录 (用于retrieval)")
    qa_parser.add_argument("--output", type=str, default=None)
    qa_parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    if args.command == "build-kg":
        build_knowledge_graph(args)
    elif args.command == "qa":
        run_qa(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
