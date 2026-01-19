"""数据结构定义"""

from dataclasses import dataclass, field
import json


@dataclass
class Memory:
    """记忆实体"""
    user_id: str
    content: str
    id: str = ""                          # 新记忆为空，由调用方赋值
    keywords: list[str] = field(default_factory=list)  # 服务于检索的关键词
    occurred_string: str | None = None    # 可残缺的时间字符串，如 "2024-12-25"、"2024-12"
    occurred_at: str | None = None        # ISO 8601 格式，如 "2024-12-25T06:30:45.123456+00:00"
    ref_dial_ids: list[str] = field(default_factory=list)    # 参考的对话 ID 列表
    ref_contents: list[str] = field(default_factory=list)    # 与 ref_dial_ids 一一对应的对话内容

    def to_str_for_dense_retrieval(self) -> str:
        """返回用于向量编码检索的字符串"""
        return self.content

    def to_str_for_dedup(self) -> str:
        """返回用于去重判断时展示在 prompt 中的 JSON 字符串"""
        return json.dumps({
            "content": self.content,
            "occurred_string": self.occurred_string,
        }, ensure_ascii=False, indent=2)

    def get_keywords(self) -> list[str]:
        """返回与此记忆相关的所有关键词"""
        return self.keywords.copy()


@dataclass
class ScoredMemory:
    """带分数的记忆（用于检索结果）"""
    memory: Memory
    score: float
