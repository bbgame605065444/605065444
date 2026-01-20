"""
SemEval 2026 Task 12: Abductive Event Reasoning
数据加载器
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AERInstance:
    """单个AER任务实例"""
    id: str
    topic_id: str
    target_event: str
    options: Dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    golden_answer: Optional[List[str]] = None  # ["A"] 或 ["A", "B"] 等
    docs: Optional[List[Dict]] = None


class AERDataLoader:
    """AER数据集加载器"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 数据目录路径，包含questions.jsonl和docs.json
        """
        self.data_dir = Path(data_dir)
        self.questions_file = self.data_dir / "questions.jsonl"
        self.docs_file = self.data_dir / "docs.json"

    def load_docs(self) -> Dict[str, Dict]:
        """加载文档，返回 {topic_id: doc_info} 映射"""
        docs_map = {}
        with open(self.docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
            for item in docs_data:
                topic_id = item['topic_id']
                docs_map[topic_id] = item
        return docs_map

    def load_questions(self, docs_map: Optional[Dict] = None) -> List[AERInstance]:
        """加载问题数据"""
        instances = []

        with open(self.questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                # 解析选项
                options = {
                    "A": data.get("option_A", ""),
                    "B": data.get("option_B", ""),
                    "C": data.get("option_C", ""),
                    "D": data.get("option_D", "")
                }

                # 解析答案
                golden_answer = None
                if "golden_answer" in data and data["golden_answer"]:
                    golden_answer = [ans.strip() for ans in data["golden_answer"].split(",")]

                # 获取相关文档
                docs = None
                if docs_map and data["topic_id"] in docs_map:
                    docs = docs_map[data["topic_id"]].get("docs", [])

                instance = AERInstance(
                    id=data["id"],
                    topic_id=data["topic_id"],
                    target_event=data["target_event"],
                    options=options,
                    golden_answer=golden_answer,
                    docs=docs
                )
                instances.append(instance)

        return instances

    def load(self) -> List[AERInstance]:
        """加载完整数据集"""
        docs_map = self.load_docs()
        return self.load_questions(docs_map)


def download_dataset(output_dir: str = "./data"):
    """从GitHub下载数据集"""
    import subprocess
    import os

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/sooo66/semeval2026-task12-dataset.git"

    if not (output_path / "semeval2026-task12-dataset").exists():
        subprocess.run(
            ["git", "clone", repo_url],
            cwd=output_path,
            check=True
        )
        print(f"数据集已下载到 {output_path / 'semeval2026-task12-dataset'}")
    else:
        print("数据集已存在")

    return output_path / "semeval2026-task12-dataset"


if __name__ == "__main__":
    # 示例用法
    dataset_path = download_dataset()

    # 加载训练数据
    train_loader = AERDataLoader(dataset_path / "train_data")
    train_data = train_loader.load()

    print(f"训练集样本数: {len(train_data)}")
    if train_data:
        sample = train_data[0]
        print(f"\n样例:")
        print(f"  ID: {sample.id}")
        print(f"  目标事件: {sample.target_event}")
        print(f"  选项A: {sample.options['A'][:50]}...")
        print(f"  正确答案: {sample.golden_answer}")
