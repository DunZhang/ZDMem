"""检索字符串生成"""

import logging
from pathlib import Path
from jinja2 import Template

from .llm import call_llm_json
from .config import get_search_string_gen_model

logger = logging.getLogger(__name__)

# 加载模板
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "generate_search_strings.j2"
with open(_TEMPLATE_PATH, encoding="utf-8") as f:
    _TEMPLATE = Template(f.read())


def generate_search_strings(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    从用户文本生成多个检索字符串

    要点：
    - 检索字符串必须是完整的句子或问题，不能过短
    - 必须覆盖文本中所有可能相关的话题和信息点
    - 宁可冗余生成，也不能遗漏
    - 返回的列表不包含原始文本（由调用方自行添加）

    Args:
        text: 用户输入文本
        model: 使用的 LLM 模型，None 则从 SEARCH_STRING_GEN_MODEL 或 DEFAULT_MODEL 获取

    Returns:
        检索字符串列表
    """
    selected_model = get_search_string_gen_model(model)
    prompt = _TEMPLATE.render(text=text)

    result = call_llm_json(prompt=prompt, model=selected_model)

    logger.debug("Generate search strings prompt:\n%s", prompt)
    logger.debug("Generate search strings response:\n%s", result)

    if isinstance(result, list):
        return [str(s) for s in result]
    return []
