"""Step 2: 记忆抽取脚本

从 processed_locomo_test_data 中的对话抽取记忆
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_core.main import process_memories
from memory_core.local_manager import LocalFileMemoryManager

# 配置控制台日志（INFO级别）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用 litellm 日志
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 配置文件日志：只记录 memory_core 的 DEBUG 及以上级别日志
LOG_FILE = Path(__file__).parent / "memory_core_debug.log"
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 只给 memory_core logger 添加文件 handler
memory_core_logger = logging.getLogger('memory_core')
memory_core_logger.setLevel(logging.DEBUG)
memory_core_logger.addHandler(file_handler)

# 路径配置
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "processed_locomo_test_data"
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate_results"


def extract_memories_for_user(user_pair: str) -> dict:
    """为一对用户抽取记忆"""
    user_data_dir = DATA_DIR / user_pair
    output_dir = INTERMEDIATE_DIR / "memories" / user_pair
    result_file = output_dir / "extraction_result.json"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 dial_id2content 映射
    dial_id2content_file = user_data_dir / "dial_id2content.json"
    dial_id2content = {}
    if dial_id2content_file.exists():
        with open(dial_id2content_file, encoding="utf-8") as f:
            dial_id2content = json.load(f)
        logger.info(f"加载 dial_id2content: {len(dial_id2content)} 条")

    # 获取所有 session 文件并按序号排序
    session_files = sorted(
        user_data_dir.glob("session_*.txt"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    logger.info(f"开始处理 {user_pair}，共 {len(session_files)} 个 session")

    all_memories = []
    session_stats = []

    for session_file in session_files:
        text = session_file.read_text(encoding='utf-8')
        text_id = session_file.stem  # e.g., "session_1"

        logger.info(f"处理 {text_id}...")

        try:
            memories = process_memories(
                text=text,
                text_id=text_id,
                user_id=user_pair,
                dial_id2content=dial_id2content,
            )

            session_stats.append({
                "session": text_id,
                "memory_count": len(memories)
            })

            all_memories.extend(memories)
            logger.info(f"  {text_id}: 抽取 {len(memories)} 条记忆")

        except Exception as e:
            logger.error(f"处理 {text_id} 失败: {e}")
            session_stats.append({
                "session": text_id,
                "memory_count": 0,
                "error": str(e)
            })

    # 使用 LocalFileMemoryManager 保存记忆并生成向量
    manager = LocalFileMemoryManager(data_dir=str(INTERMEDIATE_DIR / "memories"))
    for memory in all_memories:
        manager.add_memory(user_pair, memory)

    logger.info(f"已保存 {len(all_memories)} 条记忆及向量到 {output_dir}")

    # 保存结果摘要
    result = {
        "user_pair": user_pair,
        "extraction_time": datetime.now().isoformat(),
        "total_sessions": len(session_files),
        "total_memories": len(all_memories),
        "session_stats": session_stats,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 {user_pair}，共抽取 {len(all_memories)} 条记忆")

    return result


def main(user_pairs, max_workers=None):
    from multiprocessing import Pool, cpu_count

    user_pairs = sorted(user_pairs)
    if max_workers is None:
        max_workers = min(len(user_pairs), cpu_count())

    logger.info(f"使用 {max_workers} 个进程并行处理 {len(user_pairs)} 个用户对")

    with Pool(processes=max_workers) as pool:
        pool.map(extract_memories_for_user, user_pairs)


if __name__ == "__main__":
    main(
        user_pairs=[
            # "0__Caroline__Melanie",
            # "1__Jon__Gina",
            "2__John__Maria",
            "3__Joanna__Nate",
            "4__Tim__John",
            "5__Audrey__Andrew",
            "6__James__John",
            "7__Deborah__Jolene",
            "8__Evan__Sam",
            "9__Calvin__Dave"
        ]
    )
