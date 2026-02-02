"""Step 4: LLM as Judge 评估脚本

判断预测答案是否正确
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_core.llm import call_llm_json
from memory_core.config import get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate_results"

JUDGE_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {ground_truth}
Generated answer: {predicted}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def load_predictions(user_pair: str) -> list[dict]:
    """加载预测结果"""
    pred_file = INTERMEDIATE_DIR / "predictions" / f"{user_pair}.jsonl"
    if not pred_file.exists():
        return []

    predictions = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def evaluate_single_prediction(pred: dict, model: str, total: int) -> dict:
    """评估单个预测（用于多进程调用）"""
    qa_id = pred["qa_id"]

    try:
        # 跳过错误的预测
        if pred["predicted_answer"].startswith("ERROR:"):
            judgement = {
                "qa_id": qa_id,
                "question": pred["question"],
                "ground_truth": pred["ground_truth"],
                "predicted_answer": pred["predicted_answer"],
                "category": pred.get("category", ""),
                "evidence": pred.get("evidence", []),
                "is_correct": False,
                "label": "WRONG",
                "judge_time": datetime.now().isoformat(),
            }
        else:
            # 构建 prompt
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=pred["question"],
                ground_truth=pred["ground_truth"],
                predicted=pred["predicted_answer"]
            )

            # 调用 LLM Judge
            result = call_llm_json(prompt, model=model)

            # 将 label "CORRECT"/"WRONG" 转换为布尔值
            label = result.get("label", "WRONG").upper()
            is_correct = label == "CORRECT"

            judgement = {
                "qa_id": qa_id,
                "question": pred["question"],
                "ground_truth": pred["ground_truth"],
                "predicted_answer": pred["predicted_answer"],
                "category": pred.get("category", ""),
                "evidence": pred.get("evidence", []),
                "is_correct": is_correct,
                "label": label,
                "judge_time": datetime.now().isoformat(),
            }

        logger.info(f"[{qa_id + 1}/{total}] 评估完成，正确: {judgement['is_correct']}")
        return judgement

    except Exception as e:
        logger.error(f"评估问题 {qa_id} 失败: {e}")
        return {
            "qa_id": qa_id,
            "question": pred["question"],
            "ground_truth": pred["ground_truth"],
            "predicted_answer": pred["predicted_answer"],
            "category": pred.get("category", ""),
            "evidence": pred.get("evidence", []),
            "is_correct": False,
            "label": "WRONG",
            "judge_time": datetime.now().isoformat(),
        }


def evaluate_for_user(user_pair: str, force: bool = False, num_workers: int = None) -> dict:
    """评估一对用户的预测"""
    output_dir = INTERMEDIATE_DIR / "judgements"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{user_pair}.jsonl"

    # 加载预测结果
    predictions = load_predictions(user_pair)
    if not predictions:
        logger.error(f"预测结果不存在，请先运行 step3")
        return None

    # 加载已有评估结果
    existing_judgements = {}
    if output_file.exists() and not force:
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                existing_judgements[j["qa_id"]] = j
        logger.info(f"加载 {len(existing_judgements)} 条已有评估")

    # model = get_model(None, "DEFAULT_MODEL")
    model = "deepseek/deepseek-chat"

    # 过滤出需要评估的 predictions
    predictions_to_evaluate = []
    cached_judgements = []

    for pred in predictions:
        qa_id = pred["qa_id"]
        if qa_id in existing_judgements and not force:
            cached_judgements.append(existing_judgements[qa_id])
        else:
            predictions_to_evaluate.append(pred)

    logger.info(f"开始评估 {user_pair}，共 {len(predictions)} 个预测，"
                f"跳过 {len(cached_judgements)} 个已评估，需评估 {len(predictions_to_evaluate)} 个")

    # 使用多进程评估
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 默认最多 8 个进程

    new_judgements = []
    if predictions_to_evaluate:
        total = len(predictions)
        evaluate_func = partial(evaluate_single_prediction, model=model, total=total)

        with Pool(processes=num_workers) as pool:
            logger.info(f"使用 {num_workers} 个进程并行评估...")
            new_judgements = pool.map(evaluate_func, predictions_to_evaluate)

    # 合并结果并按 qa_id 排序
    judgements = cached_judgements + new_judgements
    judgements.sort(key=lambda x: x["qa_id"])

    # 重写完整文件
    with open(output_file, "w", encoding="utf-8") as f:
        for j in judgements:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # 统计
    correct = sum(1 for j in judgements if j["is_correct"])
    total = len(judgements)
    accuracy = correct / total if total > 0 else 0

    summary = {
        "user_pair": user_pair,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
    }

    logger.info(f"完成 {user_pair}，正确率: {accuracy:.2%} ({correct}/{total})")

    return summary




if __name__ == "__main__":
    user_pairs = [
        "0__Caroline__Melanie",
        "1__Jon__Gina",
        "2__John__Maria",
        "3__Joanna__Nate",
        "4__Tim__John",
        "5__Audrey__Andrew",
        "6__James__John",
        "7__Deborah__Jolene",
        "8__Evan__Sam",
        "9__Calvin__Dave"
    ]
    for user_pair in user_pairs:
        evaluate_for_user(user_pair, force=True, num_workers=32)

    
