"""关键词抽取"""

import logging
from pathlib import Path
from jinja2 import Template

from .llm import call_llm_json
from .config import get_keyword_extract_model

logger = logging.getLogger(__name__)

# 加载模板
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "extract_keywords.j2"
with open(_TEMPLATE_PATH, encoding="utf-8") as f:
    _TEMPLATE = Template(f.read())


def extract_keywords(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    从文本抽取关键词，仅用于检索场景

    注意：此函数不用于记忆抽取流程。记忆的 keywords 字段由 extract_memories 时 LLM 一并生成。

    典型使用场景：
    - 检索时从用户问题中抽取关键词
    - 从检索字符串中抽取关键词用于关键词检索

    Args:
        text: 输入文本（用户问题、检索字符串等）
        model: 使用的 LLM 模型，None 则从 KEYWORD_EXTRACT_MODEL 或 DEFAULT_MODEL 获取

    Returns:
        关键词列表（去重）
    """
    selected_model = get_keyword_extract_model(model)
    prompt = _TEMPLATE.render(text=text)

    result = call_llm_json(prompt=prompt, model=selected_model)

    logger.debug("Extract keywords prompt:\n%s", prompt)
    logger.debug("Extract keywords response:\n%s", result)

    # 确保返回的是字符串列表并去重
    if isinstance(result, list):
        return list(set(str(k) for k in result))
    return []
