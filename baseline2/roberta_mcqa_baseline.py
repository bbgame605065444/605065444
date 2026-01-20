"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 2b: RoBERTa/BERT for Multiple Choice

使用HuggingFace的RobertaForMultipleChoice进行多选题分类。
这是一个判别式模型，与UnifiedQA的生成式方法形成对比。

参考:
- HuggingFace: https://huggingface.co/docs/transformers/tasks/multiple_choice
- RoBERTa: https://arxiv.org/abs/1907.11692
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from baseline.evaluator import evaluate


@dataclass
class RoBERTaMCQAConfig:
    """RoBERTa MCQA配置"""
    model_name: str = "roberta-base"  # 可选: roberta-base, roberta-large, bert-base-uncased
    max_length: int = 256
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_choices: int = 4


class AERMultipleChoiceDataset(Dataset):
    """AER多选题数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        num_choices: int = 4
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_choices = num_choices
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        question = item["sent1"]
        choices = [
            item[f"ending{i}"] for i in range(self.num_choices)
        ]

        # 为每个选项创建 [CLS] question [SEP] choice [SEP] 格式
        encodings = []
        for choice in choices:
            encoding = self.tokenizer(
                question,
                choice,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encodings.append({k: v.squeeze(0) for k, v in encoding.items()})

        # 堆叠所有选项
        input_ids = torch.stack([e["input_ids"] for e in encodings])
        attention_mask = torch.stack([e["attention_mask"] for e in encodings])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item.get("id", str(idx))
        }

        if "label" in item:
            result["labels"] = torch.tensor(item["label"])
            # 保存多标签用于评估
            result["labels_all"] = item.get("labels_all", [item["label"]])

        return result


class RoBERTaMCQABaseline:
    """
    RoBERTa Multiple Choice Baseline

    可用模型:
    - roberta-base (~125M参数)
    - roberta-large (~355M参数)
    - bert-base-uncased (~110M参数)
    - bert-large-uncased (~340M参数)
    - microsoft/deberta-v3-base (~86M参数, 推荐)
    - microsoft/deberta-v3-large (~304M参数)
    """

    def __init__(self, config: Optional[RoBERTaMCQAConfig] = None):
        self.config = config or RoBERTaMCQAConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self, for_training: bool = False):
        """加载模型"""
        from transformers import AutoTokenizer, AutoModelForMultipleChoice

        print(f"加载模型: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForMultipleChoice.from_pretrained(
            self.config.model_name
        ).to(self.config.device)

        if not for_training:
            self.model.eval()

        print(f"模型已加载到 {self.config.device}")

    def train(
        self,
        train_data_path: str,
        dev_data_path: Optional[str] = None,
        output_dir: str = "./roberta_mcqa_output"
    ):
        """
        训练模型

        Args:
            train_data_path: 训练数据路径
            dev_data_path: 验证数据路径
            output_dir: 模型保存目录
        """
        from transformers import get_linear_schedule_with_warmup

        self.load_model(for_training=True)

        # 创建数据集
        train_dataset = AERMultipleChoiceDataset(
            train_data_path,
            self.tokenizer,
            self.config.max_length,
            self.config.num_choices
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        dev_loader = None
        if dev_data_path:
            dev_dataset = AERMultipleChoiceDataset(
                dev_data_path,
                self.tokenizer,
                self.config.max_length,
                self.config.num_choices
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.config.batch_size
            )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练
        print(f"\n开始训练...")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"Epochs: {self.config.num_epochs}")

        best_score = 0
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # 验证
            if dev_loader:
                dev_results = self._evaluate_loader(dev_loader)
                print(f"Dev Score: {dev_results['score']:.4f}")

                if dev_results['score'] > best_score:
                    best_score = dev_results['score']
                    self.model.save_pretrained(output_dir)
                    self.tokenizer.save_pretrained(output_dir)
                    print(f"保存最佳模型到 {output_dir}")

        print(f"\n训练完成! 最佳Dev Score: {best_score:.4f}")

    def _evaluate_loader(self, data_loader: DataLoader) -> Dict:
        """评估数据加载器"""
        self.model.eval()

        predictions = []
        goldens = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = outputs.logits.argmax(dim=-1).cpu().tolist()

                for i, pred in enumerate(preds):
                    # 预测转为集合
                    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                    predictions.append({label_map[pred]})

                    # 获取真实标签
                    labels_all = batch["labels_all"][i]
                    if isinstance(labels_all, list):
                        goldens.append({label_map[l] for l in labels_all})
                    else:
                        goldens.append({label_map[labels_all.item()]})

        return evaluate(predictions, goldens)

    def predict(self, data_path: str, output_path: Optional[str] = None) -> Dict:
        """
        预测并评估

        Args:
            data_path: RoBERTa MCQA格式的JSONL数据路径
            output_path: 结果保存路径
        """
        if self.model is None:
            self.load_model()

        dataset = AERMultipleChoiceDataset(
            data_path,
            self.tokenizer,
            self.config.max_length,
            self.config.num_choices
        )
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size)

        self.model.eval()

        all_ids = []
        all_predictions = []
        all_probs = []
        all_goldens = []
        has_labels = False

        print("开始预测...")
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).cpu().tolist()

                all_ids.extend(batch["id"])
                all_probs.extend(probs.cpu().tolist())

                label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                for i, pred in enumerate(preds):
                    all_predictions.append({label_map[pred]})

                    if "labels_all" in batch:
                        has_labels = True
                        labels_all = batch["labels_all"][i]
                        if isinstance(labels_all, list):
                            all_goldens.append({label_map[l] for l in labels_all})
                        else:
                            all_goldens.append({label_map[labels_all]})

        # 评估
        if has_labels:
            results = evaluate(all_predictions, all_goldens)
            print("\n" + "=" * 50)
            print("RoBERTa MCQA 评估结果")
            print("=" * 50)
            print(f"模型: {self.config.model_name}")
            print(f"官方分数: {results['score']:.4f}")
            print(f"完全匹配率: {results['exact_match_rate']:.4f}")
            print(f"部分匹配率: {results['partial_match_rate']:.4f}")
        else:
            results = {"note": "测试集无标签"}

        # 保存结果
        if output_path:
            label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            output_data = {
                "config": {
                    "model_name": self.config.model_name,
                    "data_path": data_path
                },
                "results": results,
                "predictions": [
                    {
                        "id": all_ids[i],
                        "prediction": list(all_predictions[i]),
                        "probabilities": {
                            label_map[j]: round(all_probs[i][j], 4)
                            for j in range(4)
                        },
                        "golden": list(all_goldens[i]) if has_labels else None
                    }
                    for i in range(len(all_ids))
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_path}")

        return results


def run_roberta_baseline(
    mode: str,
    data_path: str,
    model_name: str = "roberta-base",
    output_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    dev_data_path: Optional[str] = None,
    output_dir: str = "./roberta_mcqa_output",
    batch_size: int = 4,
    num_epochs: int = 3
) -> Dict:
    """
    运行RoBERTa MCQA baseline

    Args:
        mode: "train" 或 "predict"
        data_path: 预测时的数据路径
        model_name: 模型名称
        ...
    """
    config = RoBERTaMCQAConfig(
        model_name=model_name,
        batch_size=batch_size,
        num_epochs=num_epochs
    )

    baseline = RoBERTaMCQABaseline(config)

    if mode == "train":
        baseline.train(
            train_data_path or data_path,
            dev_data_path,
            output_dir
        )
        return {"status": "training_complete"}
    else:
        return baseline.predict(data_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoBERTa MCQA Baseline")
    parser.add_argument("--mode", type=str, choices=["train", "predict"],
                       default="predict", help="运行模式")
    parser.add_argument("--data-path", type=str, required=True,
                       help="数据路径")
    parser.add_argument("--train-data", type=str, default=None,
                       help="训练数据路径 (train模式)")
    parser.add_argument("--dev-data", type=str, default=None,
                       help="验证数据路径 (train模式)")
    parser.add_argument("--model-name", type=str, default="roberta-base",
                       help="模型名称")
    parser.add_argument("--output", type=str, default=None,
                       help="结果保存路径")
    parser.add_argument("--output-dir", type=str, default="./roberta_output",
                       help="模型保存目录 (train模式)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()

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
