"""记忆抽取 - 基于语义分段"""

import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from jinja2 import Template

from .llm import call_llm_json
from .config import get_memory_extract_model
from .models import Memory

logger = logging.getLogger(__name__)

# 加载模板
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "extract_memories.j2"
with open(_TEMPLATE_PATH, encoding="utf-8") as f:
    _TEMPLATE = Template(f.read())


def _complete_occurred_at(occurred_string: str | None) -> str | None:
    """
    将残缺时间字符串补全为完整 ISO 8601 格式

    补全规则：缺失部分用最小值补全（月=1，日=1，时=0，分=0，秒=0）

    Args:
        occurred_string: 可残缺的时间字符串

    Returns:
        完整 ISO 8601 格式字符串，或 None
    """
    if not occurred_string:
        return None

    try:
        # 尝试解析各种格式
        formats = [
            ("%Y-%m-%dT%H:%M:%S", None),
            ("%Y-%m-%dT%H:%M", None),
            ("%Y-%m-%dT%H", None),
            ("%Y-%m-%d", None),
            ("%Y-%m", None),
            ("%Y", None),
        ]

        for fmt, _ in formats:
            try:
                dt = datetime.strptime(occurred_string, fmt)
                # 补全为完整格式
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue

        return None
    except Exception:
        return None


def _parse_dial_id(dial_id: str) -> tuple[int, int]:
    """
    解析对话ID为session编号和序列号

    Args:
        dial_id: 对话ID，格式为 "D{session}:{sequence}" (例如 "D1:5")

    Returns:
        (session_number, sequence_number) 元组

    Raises:
        ValueError: 如果dial_id格式无效
    """
    match = re.match(r"D(\d+):(\d+)", dial_id)
    if not match:
        raise ValueError(f"Invalid dial_id format: {dial_id}")
    return int(match.group(1)), int(match.group(2))


def _expand_dial_id_range(
    start_dial_id: str,
    end_dial_id: str,
    dial_id2content: dict[str, str]
) -> list[str]:
    """
    展开对话ID范围为完整的对话ID列表

    Args:
        start_dial_id: 起始对话ID (例如 "D1:1")
        end_dial_id: 结束对话ID (例如 "D1:9")
        dial_id2content: 所有有效对话ID到内容的映射

    Returns:
        从start到end的所有对话ID列表（包含边界）

    Raises:
        ValueError: 如果start和end的session编号不一致
    """
    start_session, start_seq = _parse_dial_id(start_dial_id)
    end_session, end_seq = _parse_dial_id(end_dial_id)

    if start_session != end_session:
        raise ValueError(
            f"Cross-session span not allowed: {start_dial_id} to {end_dial_id}"
        )

    # 确保start <= end
    if start_seq > end_seq:
        start_seq, end_seq = end_seq, start_seq

    # 生成范围内的所有ID，只保留存在于dial_id2content中的
    result = []
    for seq in range(start_seq, end_seq + 1):
        dial_id = f"D{start_session}:{seq}"
        if dial_id in dial_id2content:
            result.append(dial_id)

    return result


def _extract_timestamp_from_content(content: str) -> str | None:
    """
    从dial_id2content的值中提取时间戳

    Args:
        content: dial_id2content的值，格式为 "2023-01-20T16:04:00,\tGina: ..."

    Returns:
        ISO时间戳字符串 (例如 "2023-01-20T16:04:00") 或 None
    """
    if not content:
        return None

    # 尝试按 ",\t" 分割
    parts = content.split(",\t", 1)
    if len(parts) >= 1:
        timestamp = parts[0].strip()
        # 验证是否像时间戳格式
        if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", timestamp):
            return timestamp

    return None


def _extract_header_info_from_text(text: str) -> tuple[str, str, str] | None:
    """
    从session文本头部提取说话人姓名和对话时间

    Args:
        text: session文本，以 "A conversation between X and Y. This conversation takes place on TIMESTAMP." 开头

    Returns:
        (speaker_a, speaker_b, timestamp) 元组，或 None（如果解析失败）
    """
    # 匹配 "A conversation between X and Y. This conversation takes place on TIMESTAMP."
    pattern = r"A conversation between (\w+) and (\w+)\. This conversation takes place on ([^.]+)\."
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2), match.group(3).strip()
    return None


def _extract_speaker_content_from_dial(dial_content: str) -> str:
    """
    从dial_id2content值中提取 "{Speaker}: {Content}" 部分

    Args:
        dial_content: dial_id2content的值，格式为 "TIMESTAMP,\tSpeaker: Content"

    Returns:
        "{Speaker}: {Content}" 格式的字符串
    """
    if not dial_content:
        return ""

    # 尝试按 ",\t" 分割，取后面的部分
    parts = dial_content.split(",\t", 1)
    if len(parts) >= 2:
        return parts[1].strip()

    # 如果没有时间戳前缀，返回原内容
    return dial_content.strip()


def _build_memory_content(
    ref_dial_ids: list[str],
    dial_id2content: dict[str, str],
    speaker_a: str,
    speaker_b: str,
    timestamp: str,
) -> str:
    """
    构建Memory的content字段

    Args:
        ref_dial_ids: 对话ID列表
        dial_id2content: 对话ID到内容的映射
        speaker_a: 第一个说话人
        speaker_b: 第二个说话人
        timestamp: 对话时间戳

    Returns:
        格式化的content字符串
    """
    # 构建头部
    header = (
        f"A conversation between {speaker_a} and {speaker_b}. "
        f"This conversation takes place on {timestamp}.\n\n"
        f"The specific content of the conversation is:"
    )

    # 构建对话内容
    dialogue_lines = []
    for dial_id in ref_dial_ids:
        dial_content = dial_id2content.get(dial_id, "")
        speaker_content = _extract_speaker_content_from_dial(dial_content)
        if speaker_content:
            dialogue_lines.append(speaker_content)

    # 组合
    content = header + "\n" + "\n".join(dialogue_lines)
    return content


def extract_memories(
    text: str,
    user_id: str,
    *,
    model: str | None = None,
    dial_id2content: dict[str, str] | None = None,
) -> list[Memory]:
    """
    从对话文本抽取新记忆（基于语义分段）

    LLM识别语义连贯的对话分段，后处理将这些分段转换为Memory对象。

    Args:
        text: 对话文本（包含 dialogue_id 标识）
        user_id: 用户标识
        model: 使用的 LLM 模型
        dial_id2content: 对话ID到内容的映射字典（必需）

    Returns:
        新抽取的 Memory 列表（id 为空，由调用方赋值后存储）
    """
    if dial_id2content is None:
        logger.warning("dial_id2content is None, returning empty list")
        return []

    selected_model = get_memory_extract_model(model)

    # 从session文本头部提取说话人和时间信息
    header_info = _extract_header_info_from_text(text)
    if header_info is None:
        logger.warning("Could not extract header info from text")
        return []
    speaker_a, speaker_b, conversation_timestamp = header_info

    # 构建 prompt
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    prompt = _TEMPLATE.render(
        text=text,
        current_time=current_time,
    )

    result = call_llm_json(prompt=prompt, model=selected_model)

    logger.debug("Extract memories prompt:\n%s", prompt)
    logger.debug("Extract memories response:\n%s", result)

    if not isinstance(result, list):
        return []

    memories = []
    for item in result:
        if not isinstance(item, dict):
            continue

        # 提取span边界
        start_dial_id = item.get("start_dial_id")
        end_dial_id = item.get("end_dial_id")

        if not start_dial_id or not end_dial_id:
            logger.warning("Missing start_dial_id or end_dial_id: %s", item)
            continue

        # 展开span为对话ID列表
        try:
            ref_dial_ids = _expand_dial_id_range(
                str(start_dial_id), str(end_dial_id), dial_id2content
            )
        except ValueError as e:
            logger.warning(
                "Invalid dial_id range %s to %s: %s",
                start_dial_id, end_dial_id, e
            )
            continue

        if not ref_dial_ids:
            logger.warning(
                "No valid dialogue IDs in range %s to %s",
                start_dial_id, end_dial_id
            )
            continue

        # 获取ref_contents
        ref_contents = [dial_id2content.get(did, "") for did in ref_dial_ids]

        # 从第一条对话中提取时间戳作为occurred_string
        first_content = dial_id2content.get(ref_dial_ids[0], "")
        occurred_string = _extract_timestamp_from_content(first_content)
        occurred_at = _complete_occurred_at(occurred_string)

        # 构建memory content
        content = _build_memory_content(
            ref_dial_ids=ref_dial_ids,
            dial_id2content=dial_id2content,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            timestamp=occurred_string or conversation_timestamp,
        )

        memory = Memory(
            user_id=user_id,
            content=content,
            keywords=[],  # 不再抽取关键词
            occurred_string=occurred_string,
            occurred_at=occurred_at,
            ref_dial_ids=ref_dial_ids,
            ref_contents=ref_contents,
        )

        memories.append(memory)

    return memories
