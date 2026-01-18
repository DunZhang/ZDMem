"""
Step 3: 使用 Locomo QA 数据评估记忆系统

对每个 QA:
1. 使用 hybrid_search 检索相关记忆
2. LLM 基于记忆预测答案
3. LLM as Judge 评估预测是否正确

保存评估结果到 evaluation_results/{user_pair}/qa_results.json
"""

import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import litellm
litellm.suppress_debug_info = True

from jinja2 import Environment, FileSystemLoader

from memory_core.llm import call_llm_json
from memory_core.local_manager import LocalFileMemoryManager
from memory_core.config import get_model, get_default_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 加载 Jinja2 模板
SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))


def predict_answer(
    question: str,
    memories: list[dict],
    speakers: list[str],
    model: str | None = None,
) -> dict:
    """
    基于记忆预测答案

    Args:
        question: 问题
        memories: 检索到的记忆列表
        speakers: 对话人名字
        model: LLM 模型

    Returns:
        预测结果 dict
    """
    model = model or get_default_model()
    template = jinja_env.get_template("qa_answer.j2")

    prompt = template.render(
        question=question,
        memories=memories,
        speakers=" and ".join(speakers),
    )

    try:
        result = call_llm_json(prompt, model=model)
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", "none"),
            "used_memory_indices": result.get("used_memory_indices", []),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        logger.error(f"预测答案时出错: {e}")
        return {
            "answer": f"Error: {e}",
            "confidence": "none",
            "used_memory_indices": [],
            "reasoning": str(e),
        }


def judge_answer(
    question: str,
    ground_truth: str,
    prediction: str,
    model: str | None = None,
) -> dict:
    """
    评估预测答案是否正确

    Args:
        question: 问题
        ground_truth: 标准答案
        prediction: 预测答案
        model: LLM 模型

    Returns:
        评估结果 dict
    """
    model = model or get_default_model()
    template = jinja_env.get_template("llm_judge.j2")

    prompt = template.render(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )

    try:
        result = call_llm_json(prompt, model=model)
        return {
            "is_correct": result.get("is_correct", False),
            "score": result.get("score", 0.0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        logger.error(f"评估答案时出错: {e}")
        return {
            "is_correct": False,
            "score": 0.0,
            "explanation": f"Error: {e}",
        }


def evaluate_qa_for_user_pair(
    user_pair_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    model: str | None = None,
) -> dict:
    """
    评估一对对话人的所有 QA

    Args:
        user_pair_dir: 原始数据目录
        output_dir: 评估结果目录
        top_k: 检索返回的记忆数量
        model: LLM 模型

    Returns:
        评估统计结果
    """
    user_id = user_pair_dir.name
    user_output_dir = output_dir / user_id
    logger.info(f"开始评估: {user_id}")

    # 检查是否已抽取记忆
    memories_file = user_output_dir / "extracted_memories.json"
    if not memories_file.exists():
        raise FileNotFoundError(
            f"未找到抽取的记忆文件: {memories_file}\n"
            f"请先运行 step2_extract_memories.py"
        )

    # 加载 manager
    manager = LocalFileMemoryManager(data_dir=str(user_output_dir / "memory_data"))

    # 加载 QA 测试集
    qa_file = user_pair_dir / "test_qa.json"
    qa_list = json.loads(qa_file.read_text(encoding="utf-8"))

    # 解析对话人名字
    parts = user_id.split("__")
    speakers = parts[1:] if len(parts) > 1 else [user_id]

    results = []
    correct_count = 0

    for idx, qa in enumerate(qa_list):
        question = qa["question"]
        ground_truth = qa["answer"]
        evidence = qa.get("evidence", [])
        category = qa.get("category", "").strip()

        logger.info(f"  [{idx + 1}/{len(qa_list)}] {question[:50]}...")

        # Step 1: 检索相关记忆
        search_results = manager.hybrid_search(user_id, question, top_k=top_k)
        retrieved_memories = [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "keywords": r.memory.keywords,
                "occurred_string": r.memory.occurred_string,
                "score": r.score,
            }
            for r in search_results
        ]

        # Step 2: LLM 预测答案
        prediction_result = predict_answer(
            question=question,
            memories=retrieved_memories,
            speakers=speakers,
            model=model,
        )

        # Step 3: LLM Judge 评估
        judge_result = judge_answer(
            question=question,
            ground_truth=str(ground_truth),
            prediction=prediction_result["answer"],
            model=model,
        )

        if judge_result["is_correct"]:
            correct_count += 1

        # 整合结果
        result = {
            "id": idx + 1,
            "question": question,
            "ground_truth": ground_truth,
            "evidence": evidence,
            "category": category,
            "prediction": prediction_result["answer"],
            "prediction_confidence": prediction_result["confidence"],
            "used_memory_indices": prediction_result["used_memory_indices"],
            "prediction_reasoning": prediction_result["reasoning"],
            "is_correct": judge_result["is_correct"],
            "judge_score": judge_result["score"],
            "judge_explanation": judge_result["explanation"],
            "retrieved_memories": retrieved_memories,
        }
        results.append(result)

        # 实时显示结果
        status = "CORRECT" if judge_result["is_correct"] else "WRONG"
        logger.info(f"    {status} (score: {judge_result['score']:.2f})")

    # 计算统计
    accuracy = correct_count / len(qa_list) if qa_list else 0

    # 按类别统计
    category_stats = {}
    for r in results:
        cat = r["category"] or "Unknown"
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if r["is_correct"]:
            category_stats[cat]["correct"] += 1

    for cat in category_stats:
        stats = category_stats[cat]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    # 保存评估结果
    output = {
        "user_pair": user_id,
        "total_questions": len(qa_list),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "category_stats": category_stats,
        "results": results,
    }

    qa_results_file = user_output_dir / "qa_results.json"
    with open(qa_results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 {user_id}: 准确率 {accuracy:.2%} ({correct_count}/{len(qa_list)})")

    return output


def main(
    user_pairs: list[str] | None = None,
    top_k: int = 10,
    model: str | None = None,
):
    """
    主函数

    Args:
        user_pairs: 要处理的对话人列表
        top_k: 检索返回的记忆数量
        model: LLM 模型
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "processed_locomo_test_data"
    output_dir = script_dir / "evaluation_results"

    # 获取所有对话人目录
    all_user_pairs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    # 过滤
    if user_pairs:
        all_user_pairs = [d for d in all_user_pairs if d.name in user_pairs]

    logger.info(f"待评估的对话人数量: {len(all_user_pairs)}")
    logger.info(f"使用模型: {model or get_default_model()}")

    all_stats = []
    for user_pair_dir in all_user_pairs:
        try:
            stats = evaluate_qa_for_user_pair(
                user_pair_dir,
                output_dir,
                top_k=top_k,
                model=model,
            )
            all_stats.append({
                "user_pair": stats["user_pair"],
                "total": stats["total_questions"],
                "correct": stats["correct_count"],
                "accuracy": stats["accuracy"],
                "category_stats": stats["category_stats"],
            })
        except Exception as e:
            logger.error(f"评估 {user_pair_dir.name} 时出错: {e}")
            raise

    # 保存汇总统计
    summary_file = output_dir / "evaluation_summary.json"
    total_questions = sum(s["total"] for s in all_stats)
    total_correct = sum(s["correct"] for s in all_stats)
    summary = {
        "total_user_pairs": len(all_stats),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": total_correct / total_questions if total_questions > 0 else 0,
        "user_pairs": all_stats,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n=== 评估完成 ===")
    logger.info(f"总问题数: {total_questions}")
    logger.info(f"正确数: {total_correct}")
    logger.info(f"总体准确率: {summary['overall_accuracy']:.2%}")
    logger.info(f"结果保存至: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估 Locomo QA")
    parser.add_argument(
        "--user-pairs",
        nargs="+",
        help="要处理的对话人列表",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索返回的记忆数量 (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 模型 (default: 使用环境变量 DEFAULT_MODEL)",
    )
    args = parser.parse_args()

    main(
        user_pairs=args.user_pairs,
        top_k=args.top_k,
        model=args.model,
    )
