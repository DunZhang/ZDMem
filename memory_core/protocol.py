"""协议接口定义"""

from typing import Protocol
from .models import ScoredMemory


class MemoryDatabaseManager(Protocol):
    """记忆数据库管理器协议"""

    def search_by_vector(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None
    ) -> list[ScoredMemory]:
        """
        向量检索

        Args:
            user_id: 用户标识
            query: 查询文本
            top_k: 返回数量
            score_threshold: 相似度阈值，低于此值的结果不返回

        Returns:
            按相似度降序排列的 ScoredMemory 列表
        """
        ...

    def search_by_keywords(
        self,
        user_id: str,
        keywords: list[str],
        top_k: int = 10
    ) -> list[ScoredMemory]:
        """
        关键词精确匹配

        调用方负责：
        1. 使用 get_synonyms 获取同义词表
        2. 在本方法内部对输入的 keywords 进行同义词扩展后再匹配

        Args:
            user_id: 用户标识
            keywords: 关键词列表
            top_k: 返回数量

        Returns:
            按相关性降序排列的 ScoredMemory 列表
        """
        ...

    def get_all_keywords(self, user_id: str) -> list[str]:
        """
        获取记忆库中所有关键词（去重）

        Args:
            user_id: 用户标识

        Returns:
            关键词列表
        """
        ...

    def get_synonyms(self, user_id: str) -> dict[str, list[str]]:
        """
        获取同义词表

        Args:
            user_id: 用户标识

        Returns:
            格式：{canonical_keyword: [synonym1, synonym2, ...]}
        """
        ...

    def hybrid_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None
    ) -> list[ScoredMemory]:
        """
        混合检索：向量检索 + 关键词检索 + RRF 融合

        Args:
            user_id: 用户标识
            query: 查询文本
            top_k: 返回数量
            score_threshold: 向量检索的相似度阈值

        Returns:
            RRF 融合后的 ScoredMemory 列表，按分数降序排列
        """
        ...
