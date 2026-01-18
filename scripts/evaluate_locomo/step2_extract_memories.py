"""
Step 2: 从 Locomo 对话数据中抽取记忆

对每对对话人的所有 session 文件调用 process_memories() 抽取记忆，
保存到 evaluation_results/{user_pair}/ 目录下。
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

from memory_core import process_memories
from memory_core.local_manager import LocalFileMemoryManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def extract_memories_for_user_pair(
    user_pair_dir: Path,
    output_dir: Path,
) -> dict:
    """
    为一对对话人抽取所有对话的记忆

    Args:
        user_pair_dir: 对话人目录，如 processed_locomo_test_data/0__Caroline__Melanie
        output_dir: 输出目录，如 evaluation_results

    Returns:
        抽取结果统计
    """
    user_id = user_pair_dir.name  # e.g., "0__Caroline__Melanie"
    logger.info(f"开始处理: {user_id}")

    # 创建独立的 manager
    user_output_dir = output_dir / user_id
    user_output_dir.mkdir(parents=True, exist_ok=True)
    manager = LocalFileMemoryManager(data_dir=str(user_output_dir / "memory_data"))

    # 按 session 顺序处理（保持时序）
    session_files = sorted(
        user_pair_dir.glob("session_*.txt"),
        key=lambda x: int(x.stem.split("_")[1])
    )

    stats = {
        "user_pair": user_id,
        "total_sessions": len(session_files),
        "total_created": 0,
        "total_updated": 0,
        "total_deleted": 0,
        "sessions": [],
    }

    all_memories = []

    for session_file in session_files:
        text = session_file.read_text(encoding="utf-8")
        text_id = session_file.stem  # e.g., "session_1"

        logger.info(f"  处理 {text_id}...")

        # 抽取记忆
        result = process_memories(
            text=text,
            text_id=text_id,
            user_id=user_id,
            manager=manager,
        )

        session_stats = {
            "session_id": text_id,
            "created": len(result.created),
            "updated": len(result.updated),
            "deleted": len(result.deleted),
        }
        stats["sessions"].append(session_stats)
        stats["total_created"] += session_stats["created"]
        stats["total_updated"] += session_stats["updated"]
        stats["total_deleted"] += session_stats["deleted"]

        # 保存新创建的记忆
        for op in result.created:
            if op.created_memory:
                saved = manager.add_memory(user_id, op.created_memory)
                all_memories.append({
                    "id": saved.id,
                    "content": saved.content,
                    "keywords": saved.keywords,
                    "occurred_string": saved.occurred_string,
                    "occurred_at": saved.occurred_at,
                    "source_session": text_id,
                    "references": [
                        {"text_id": ref.text_id, "lines": ref.lines}
                        for ref in saved.references
                    ] if saved.references else [],
                })

        logger.info(f"    创建: {session_stats['created']}, "
                   f"更新: {session_stats['updated']}, "
                   f"删除: {session_stats['deleted']}")

    # 保存抽取的记忆列表
    memories_file = user_output_dir / "extracted_memories.json"
    with open(memories_file, "w", encoding="utf-8") as f:
        json.dump(all_memories, f, ensure_ascii=False, indent=2)

    # 保存统计信息
    stats_file = user_output_dir / "extraction_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 {user_id}: 共抽取 {len(all_memories)} 条记忆")

    return stats


def main(user_pairs: list[str] | None = None):
    """
    主函数

    Args:
        user_pairs: 要处理的对话人列表，如 ["0__Caroline__Melanie"]
                   如果为 None 则处理所有
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "processed_locomo_test_data"
    output_dir = script_dir / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    # 获取所有对话人目录
    all_user_pairs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    # 过滤
    if user_pairs:
        all_user_pairs = [d for d in all_user_pairs if d.name in user_pairs]

    logger.info(f"待处理的对话人数量: {len(all_user_pairs)}")

    all_stats = []
    for user_pair_dir in all_user_pairs:
        try:
            stats = extract_memories_for_user_pair(user_pair_dir, output_dir)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"处理 {user_pair_dir.name} 时出错: {e}")
            raise

    # 保存汇总统计
    summary_file = output_dir / "extraction_summary.json"
    summary = {
        "total_user_pairs": len(all_stats),
        "total_memories": sum(s["total_created"] for s in all_stats),
        "user_pairs": all_stats,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n=== 抽取完成 ===")
    logger.info(f"处理对话人数: {len(all_stats)}")
    logger.info(f"总记忆数: {summary['total_memories']}")
    logger.info(f"结果保存至: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 Locomo 对话中抽取记忆")
    parser.add_argument(
        "--user-pairs",
        nargs="+",
        help="要处理的对话人列表，如: 0__Caroline__Melanie 1__Jon__Gina",
    )
    args = parser.parse_args()

    main(user_pairs=args.user_pairs)
