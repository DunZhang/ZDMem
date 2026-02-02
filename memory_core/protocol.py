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
