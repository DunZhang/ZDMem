"""同义词生成"""

import logging
from pathlib import Path
from jinja2 import Template

from .llm import call_llm_json
from .config import (
    get_synonym_gen_model,
    get_synonym_batch_size,
    get_synonym_max_retries,
)

logger = logging.getLogger(__name__)

# 加载模板
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "generate_synonyms.j2"
with open(_TEMPLATE_PATH, encoding="utf-8") as f:
    _TEMPLATE = Template(f.read())


def _generate_synonyms_batch(
    keywords: list[str],
    model: str,
) -> dict[str, list[str]]:
    """单批次同义词生成"""
    prompt = _TEMPLATE.render(keywords=keywords)
    result = call_llm_json(prompt=prompt, model=model)

    logger.debug("Generate synonyms prompt:\n%s", prompt)
    logger.debug("Generate synonyms response:\n%s", result)

    if isinstance(result, dict):
        # 确保所有值都是字符串列表
        return {
            str(k): [str(s) for s in v] if isinstance(v, list) else []
            for k, v in result.items()
        }
    return {}


def generate_synonyms(
    keywords: list[str],
    *,
    batch_size: int | None = None,
    max_retries: int | None = None,
    model: str | None = None,
) -> dict[str, list[str]]:
    """
    批量生成同义词

    注意：本函数只负责生成同义词，不负责存储。调用方需要：
    1. 自行决定何时调用（如定期触发、新关键词达到一定数量时）
    2. 调用 manager.save_synonyms 存储结果
    3. 在 search_by_keywords 中使用同义词表进行扩展

    处理流程：
    1. 将 keywords 分批，每批 batch_size 个
    2. 调用 LLM 为各批次生成同义词
    3. 纠错：检查结果中是否有关键词被遗漏，遗漏的重新生成
    4. 合并所有批次结果

    Args:
        keywords: 关键词列表
        batch_size: 每批处理数量，None 则从 SYNONYM_BATCH_SIZE 环境变量获取，默认 20
        max_retries: 纠错重试次数，None 则从 SYNONYM_MAX_RETRIES 环境变量获取，默认 2
        model: 使用的 LLM 模型，None 则从 SYNONYM_GEN_MODEL 或 DEFAULT_MODEL 获取

    Returns:
        同义词映射表，格式：{keyword: [synonym1, synonym2, ...]}
        - 没有同义词的关键词返回空数组
    """
    if not keywords:
        return {}

    selected_model = get_synonym_gen_model(model)
    actual_batch_size = get_synonym_batch_size(batch_size)
    actual_max_retries = get_synonym_max_retries(max_retries)

    # 去重
    unique_keywords = list(set(keywords))

    # 分批
    batches = [
        unique_keywords[i:i + actual_batch_size]
        for i in range(0, len(unique_keywords), actual_batch_size)
    ]

    all_synonyms = {}

    # 处理各批次
    for batch in batches:
        batch_result = _generate_synonyms_batch(batch, selected_model)
        all_synonyms.update(batch_result)

    # 纠错：检查遗漏的关键词
    missing_keywords = [k for k in unique_keywords if k not in all_synonyms]
    retry_count = 0

    while missing_keywords and retry_count < actual_max_retries:
        retry_count += 1
        logger.warning(f"Missing keywords: {missing_keywords}, retry {retry_count}")

        retry_result = _generate_synonyms_batch(missing_keywords, selected_model)
        all_synonyms.update(retry_result)

        missing_keywords = [k for k in unique_keywords if k not in all_synonyms]

    # 确保所有关键词都有结果（即使为空列表）
    for k in unique_keywords:
        if k not in all_synonyms:
            all_synonyms[k] = []

    return all_synonyms
