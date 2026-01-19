"""配置管理"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


def get_model(
    param_value: str | None,
    env_var_name: str,
    default_env_var: str = "DEFAULT_MODEL"
) -> str:
    """
    模型选择逻辑：参数值 > 对应环境变量 > DEFAULT_MODEL

    Args:
        param_value: 函数参数传入的模型值
        env_var_name: 对应的环境变量名
        default_env_var: 默认模型环境变量名

    Returns:
        选定的模型名称
    """
    if param_value:
        return param_value

    env_value = os.getenv(env_var_name)
    if env_value:
        return env_value

    default_model = os.getenv(default_env_var)
    if default_model:
        return default_model

    return "deepseek/deepseek-chat"  # 最终兜底


def get_env_int(name: str, default: int) -> int:
    """获取整数环境变量"""
    value = os.getenv(name)
    return int(value) if value else default


def get_env_float(name: str, default: float) -> float:
    """获取浮点数环境变量"""
    value = os.getenv(name)
    return float(value) if value else default


# 预定义的配置获取函数
def get_search_string_gen_model(param: str | None = None) -> str:
    """获取检索字符串生成模型"""
    return get_model(param, "SEARCH_STRING_GEN_MODEL")


def get_keyword_extract_model(param: str | None = None) -> str:
    """获取关键词抽取模型"""
    return get_model(param, "KEYWORD_EXTRACT_MODEL")


def get_memory_extract_model(param: str | None = None) -> str:
    """获取记忆抽取模型"""
    return get_model(param, "MEMORY_EXTRACT_MODEL")


def get_synonym_gen_model(param: str | None = None) -> str:
    """获取同义词生成模型"""
    return get_model(param, "SYNONYM_GEN_MODEL")


def get_embedding_model() -> str:
    """获取 embedding 模型"""
    return os.getenv("EMBEDDING_MODEL", "voyage/voyage-3-large")


def get_memory_data_dir() -> str:
    """获取数据存储目录"""
    return os.getenv("MEMORY_DATA_DIR", "./data")


def get_search_top_k(param: int | None = None) -> int:
    """获取检索返回数量"""
    return param if param is not None else get_env_int("SEARCH_TOP_K", 10)


def get_vector_score_threshold(param: float | None = None) -> float:
    """获取向量检索相似度阈值"""
    return param if param is not None else get_env_float("VECTOR_SCORE_THRESHOLD", 0.5)


def get_synonym_batch_size(param: int | None = None) -> int:
    """获取同义词生成批次大小"""
    return param if param is not None else get_env_int("SYNONYM_BATCH_SIZE", 20)


def get_synonym_max_retries(param: int | None = None) -> int:
    """获取同义词生成最大重试次数"""
    return param if param is not None else get_env_int("SYNONYM_MAX_RETRIES", 2)


def get_rrf_k() -> int:
    """获取 RRF 融合参数 k"""
    return get_env_int("RRF_K", 60)
