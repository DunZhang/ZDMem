"""Step 3: 预测答案脚本

基于检索到的记忆，使用 LLM 回答问题
"""
import os
import sys
import json
import logging
import random
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_core.local_manager import LocalFileMemoryManager
from memory_core.llm import call_llm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用 litellm 日志
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 配置文件日志：记录 memory_core 的日志（包含采样的 prompt/response）
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "step3_predict_answers.log"
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 给 memory_core logger 添加文件 handler
memory_core_logger = logging.getLogger('memory_core')
memory_core_logger.setLevel(logging.INFO)
memory_core_logger.addHandler(file_handler)
DATA_DIR = SCRIPT_DIR / "processed_locomo_test_data"
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate_results"

ANSWER_PROMPT_TEMPLATE = """Based on the following memories about conversations between two people, please answer the question.

Memories (with occurred time string):
{memories}



Question: {question}

Instructions:
- Pay attention to the time information embedded in the content to understand when events occurred
- Keep your answer concise and direct

Answer:"""


def load_test_qa(user_pair: str) -> list[dict]:
    """加载测试 QA 数据"""
    qa_file = DATA_DIR / user_pair / "test_qa.json"
    with open(qa_file, encoding="utf-8") as f:
        return json.load(f)


def process_single_question(
        qa_with_id: tuple[int, dict],
        user_pair: str,
        memories_dir: Path
) -> dict:
    """处理单个问题（供多进程调用）"""
    qa_id, qa = qa_with_id
    question = qa["question"]

    try:
        # 每个进程需要独立初始化 manager
        manager = LocalFileMemoryManager(data_dir=str(memories_dir))

        # 检索相关记忆
        # TODO 暂时只用向量
        retrieved = manager.search_by_vector(
            user_id=user_pair,
            query=question,
            top_k=int(os.getenv("SEARCH_TOP_K")),
            score_threshold=0.1
        )

        # 构建记忆文本
        if retrieved:
            memories_text = "\n\n\n\n\n".join([
                r.memory.content for r in retrieved
            ])
        else:
            memories_text = "(No relevant memories found)"

        # 构建 prompt
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            memories=memories_text,
            question=question
        )
        # print(prompt)
        # 调用 LLM
        predicted_answer, usage = call_llm(prompt, model=os.getenv("QA_MODEL"), return_usage=True)
        prediction = {
            "qa_id": qa_id,
            "question": question,
            "ground_truth": qa["answer"],
            "evidence": qa.get("evidence", []),
            "category": qa.get("category", "").strip(),
            "retrieved_memories": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "occurred_string": r.memory.occurred_string,
                    "score": r.score,
                    "references": r.memory.ref_dial_ids,
                }
                for r in retrieved
            ],
            "predicted_answer": predicted_answer.strip(),
            "prediction_time": datetime.now().isoformat(),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }

        return prediction

    except Exception as e:
        logger.error(f"处理问题 {qa_id} 失败: {e}")
        return {
            "qa_id": qa_id,
            "question": question,
            "ground_truth": qa["answer"],
            "evidence": qa.get("evidence", []),
            "category": qa.get("category", "").strip(),
            "retrieved_memories": [],
            "predicted_answer": f"ERROR: {e}",
            "prediction_time": datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def predict_answers_for_user(
        user_pair: str,
        max_questions: int | None = None,
        force: bool = False,
        num_workers: int | None = None
) -> dict:
    """为一对用户预测答案（多进程版本）

    Args:
        user_pair: 用户对标识
        max_questions: 最大问题数量限制
        force: 是否强制重新预测
        num_workers: 进程数，默认为 CPU 核心数
    """
    memories_dir = INTERMEDIATE_DIR / "memories"
    output_dir = INTERMEDIATE_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{user_pair}.jsonl"

    # 检查记忆是否已抽取
    user_memories_dir = memories_dir / user_pair
    if not user_memories_dir.exists():
        logger.error(f"记忆目录不存在: {user_memories_dir}，请先运行 step2")
        return None

    # 加载已有预测结果（支持断点续传）
    existing_predictions = {}
    if output_file.exists() and not force:
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                pred = json.loads(line)
                existing_predictions[pred["qa_id"]] = pred
        logger.info(f"加载 {len(existing_predictions)} 条已有预测")

    # 加载测试数据
    test_qa = load_test_qa(user_pair)
    if max_questions:
        test_qa = test_qa[:max_questions]

    logger.info(f"开始预测 {user_pair}，共 {len(test_qa)} 个问题")

    # 筛选需要处理的问题（支持断点续传）
    questions_to_process = []
    existing_results = []
    for i, qa in enumerate(test_qa):
        qa_id = i
        if qa_id in existing_predictions and not force:
            existing_results.append(existing_predictions[qa_id])
        else:
            questions_to_process.append((qa_id, qa))

    logger.info(f"需要处理 {len(questions_to_process)} 个新问题，跳过 {len(existing_results)} 个已有预测")

    # 多进程处理
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 限制最大进程数为 8

    new_predictions = []
    if questions_to_process:
        # 使用 partial 固定部分参数
        process_func = partial(
            process_single_question,
            user_pair=user_pair,
            memories_dir=memories_dir
        )

        logger.info(f"使用 {num_workers} 个进程并行处理")

        with Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 获取结果并实时保存
            for i, prediction in enumerate(pool.imap_unordered(process_func, questions_to_process)):
                new_predictions.append(prediction)
                logger.info(f"[{i + 1}/{len(questions_to_process)}] 完成问题 {prediction['qa_id']}")

                # 实时追加保存（断点续传）
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    # 合并所有结果并按 qa_id 排序
    all_predictions = existing_results + new_predictions
    all_predictions.sort(key=lambda x: x["qa_id"])

    # 重写完整文件（保证顺序）
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    logger.info(f"完成 {user_pair}，共预测 {len(all_predictions)} 个问题")

    return {
        "user_pair": user_pair,
        "total_questions": len(all_predictions),
        "output_file": str(output_file),
    }


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
        predict_answers_for_user(
            user_pair=user_pair,
            max_questions=200000,
            force=True,
            num_workers=32  # 可调整进程数
        )
