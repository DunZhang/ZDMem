"""LLM 调用封装"""

import logging
import json
import random
from typing import Any
from litellm import completion, embedding
from json_repair import repair_json

logger = logging.getLogger(__name__)

# 采样输出的概率
_SAMPLE_LOG_RATE = 0.01


def parse_json_response(response_text: str) -> Any:
    """
    解析 LLM 的 JSON 响应

    处理逻辑：
    1. 去除可能的 ```json 或 ``` 包裹
    2. 使用 json_repair 修复常见格式问题
    3. 解析为 Python 对象

    Args:
        response_text: LLM 返回的原始文本

    Returns:
        解析后的 Python 对象
    """
    text = response_text.strip()

    # 处理 ```json ... ``` 包裹
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # 使用 json_repair 修复并解析
    repaired = repair_json(text)
    return json.loads(repaired)


def call_llm(
    prompt: str,
    model: str,
    max_retries: int = 3,
    response_format: dict | None = None,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """
    调用 LLM 并返回响应文本

    Args:
        prompt: 用户 prompt
        model: 模型名称
        max_retries: 最大重试次数
        response_format: 响应格式配置
        return_usage: 是否返回 token 使用信息

    Returns:
        如果 return_usage=False: LLM 响应文本
        如果 return_usage=True: (响应文本, {"prompt_tokens": int, "completion_tokens": int})

    Raises:
        Exception: 重试耗尽后抛出最后一个异常
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = completion(**kwargs)
            content = response.choices[0].message.content

            # 以 _SAMPLE_LOG_RATE 的概率采样输出 prompt 和原始响应
            # 必须一起输出，避免多线程环境下对不上
            if random.random() < _SAMPLE_LOG_RATE:
                logger.info(
                    "[LLM采样日志] model=%s\n"
                    "========== PROMPT ==========\n%s\n"
                    "========== RESPONSE ==========\n%s\n"
                    "========== END ==========",
                    model, prompt, content
                )

            if return_usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                return content, usage
            return content

        except Exception as e:
            last_exception = e
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue

    raise last_exception


def call_llm_json(
    prompt: str,
    model: str,
    max_retries: int = 3,
) -> Any:
    """
    调用 LLM 并解析 JSON 响应

    Args:
        prompt: 用户 prompt
        model: 模型名称
        max_retries: 最大重试次数

    Returns:
        解析后的 Python 对象
    """
    response_text = call_llm(
        prompt=prompt,
        model=model,
        max_retries=max_retries,
        response_format={"type": "json_object"},
    )
    return parse_json_response(response_text)


def get_embedding(text: str, model: str, input_type: str = "document") -> list[float]:
    """
    获取文本的 embedding 向量

    Args:
        text: 输入文本
        model: embedding 模型名称
        input_type: 输入类型，"document" 用于存储的文档，"query" 用于搜索查询

    Returns:
        embedding 向量
    """
    response = embedding(model=model, input=[text], input_type=input_type)
    return response.data[0]["embedding"]


def get_embeddings(texts: list[str], model: str, input_type: str = "document") -> list[list[float]]:
    """
    批量获取文本的 embedding 向量

    Args:
        texts: 输入文本列表
        model: embedding 模型名称
        input_type: 输入类型，"document" 用于存储的文档，"query" 用于搜索查询

    Returns:
        embedding 向量列表
    """
    if not texts:
        return []

    response = embedding(model=model, input=texts, input_type=input_type)
    return [item["embedding"] for item in response.data]
