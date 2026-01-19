"""主入口 - 记忆处理流程"""

import logging
from .models import Memory
from .extract import extract_memories

logger = logging.getLogger(__name__)


def process_memories(
    text: str,
    text_id: str,
    user_id: str,
    *,
    memory_extract_model: str | None = None,
    dial_id2content: dict[str, str] | None = None,
) -> list[Memory]:
    """
    从文本抽取新记忆

    Args:
        text: 输入文本（对话记录、用户笔记等）
        text_id: 文本唯一标识，用于记录记忆来源
        user_id: 用户标识
        memory_extract_model: 记忆抽取模型
        dial_id2content: 对话ID到内容的映射字典，用于填充 ref_contents

    Returns:
        新抽取的 Memory 列表（id 为空，由调用方赋值后存储）
    """
    logger.info("Processing memories for user %s, text_id %s", user_id, text_id)

    # 抽取记忆
    new_memories = extract_memories(
        text=text,
        user_id=user_id,
        model=memory_extract_model,
        dial_id2content=dial_id2content,
    )

    logger.info("Extracted %d new memories", len(new_memories))

    return new_memories
