"""
SemEval 2026 Task 12: Abductive Event Reasoning
评估器 - 官方评估指标实现
"""
from typing import List, Set, Optional


def calculate_instance_score(prediction: Set[str], golden: Set[str]) -> float:
    """
    计算单个实例的分数

    评分规则:
    - 1.0: 完全匹配 (预测 == 标准答案)
    - 0.5: 部分匹配 (预测是标准答案的真子集)
    - 0.0: 错误 (预测为空、包含错误选项、或是标准答案的超集)

    Args:
        prediction: 预测的答案集合，如 {"A"} 或 {"A", "B"}
        golden: 标准答案集合

    Returns:
        分数 (0.0, 0.5, 或 1.0)
    """
    if not prediction:
        return 0.0

    if prediction == golden:
        return 1.0

    # 检查是否是真子集（预测的每个答案都在标准答案中，但不完全相同）
    if prediction < golden:  # 真子集
        return 0.5

    # 其他情况（包含错误选项或是超集）
    return 0.0


def evaluate(predictions: List[Set[str]], goldens: List[Set[str]]) -> dict:
    """
    评估整个数据集

    Args:
        predictions: 预测答案列表
        goldens: 标准答案列表

    Returns:
        包含各项指标的字典
    """
    assert len(predictions) == len(goldens), "预测和标准答案数量不匹配"

    scores = []
    exact_match = 0
    partial_match = 0
    wrong = 0

    for pred, gold in zip(predictions, goldens):
        score = calculate_instance_score(pred, gold)
        scores.append(score)

        if score == 1.0:
            exact_match += 1
        elif score == 0.5:
            partial_match += 1
        else:
            wrong += 1

    n = len(scores)
    return {
        "score": sum(scores) / n if n > 0 else 0.0,  # 官方指标
        "exact_match_rate": exact_match / n if n > 0 else 0.0,
        "partial_match_rate": partial_match / n if n > 0 else 0.0,
        "wrong_rate": wrong / n if n > 0 else 0.0,
        "total": n,
        "exact_match": exact_match,
        "partial_match": partial_match,
        "wrong": wrong
    }


def parse_prediction(pred_str: str) -> Set[str]:
    """
    解析模型输出为答案集合

    支持多种格式:
    - "A"
    - "A, B"
    - "A,B"
    - "The answer is A and B"
    - etc.
    """
    valid_options = {"A", "B", "C", "D"}
    result = set()

    # 简单解析：提取所有出现的有效选项
    pred_upper = pred_str.upper()
    for opt in valid_options:
        # 检查选项是否独立出现（避免误匹配其他词中的字母）
        import re
        if re.search(rf'\b{opt}\b', pred_upper):
            result.add(opt)

    return result


if __name__ == "__main__":
    # 测试评估器
    test_cases = [
        ({"A"}, {"A"}, 1.0),  # 完全匹配
        ({"A", "B"}, {"A", "B"}, 1.0),  # 完全匹配（多选）
        ({"A"}, {"A", "B"}, 0.5),  # 部分匹配
        ({"A", "B"}, {"A"}, 0.0),  # 超集，错误
        ({"C"}, {"A"}, 0.0),  # 错误答案
        (set(), {"A"}, 0.0),  # 空预测
    ]

    print("评估器测试:")
    for pred, gold, expected in test_cases:
        score = calculate_instance_score(pred, gold)
        status = "✓" if score == expected else "✗"
        print(f"  {status} pred={pred}, gold={gold} -> {score} (expected: {expected})")
