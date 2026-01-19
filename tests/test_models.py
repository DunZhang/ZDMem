"""数据结构测试"""

import json
from memory_core.models import Memory, ScoredMemory


class TestMemory:
    """Memory 测试"""

    def test_create_minimal(self):
        """最小创建测试"""
        memory = Memory(user_id="user_001", content="测试内容")

        assert memory.user_id == "user_001"
        assert memory.content == "测试内容"
        assert memory.id == ""
        assert memory.keywords == []
        assert memory.occurred_string is None
        assert memory.occurred_at is None
        assert memory.ref_dial_ids == []
        assert memory.ref_contents == []

    def test_create_full(self):
        """完整创建测试"""
        memory = Memory(
            user_id="user_001",
            content="测试内容",
            id="mem_001",
            keywords=["关键词1", "关键词2"],
            occurred_string="2024-12-25",
            occurred_at="2024-12-25T00:00:00+00:00",
            ref_dial_ids=["text_001", "text_001"],
            ref_contents=["原文1", "原文2"],
        )

        assert memory.id == "mem_001"
        assert memory.keywords == ["关键词1", "关键词2"]
        assert memory.occurred_string == "2024-12-25"
        assert memory.ref_dial_ids == ["text_001", "text_001"]
        assert memory.ref_contents == ["原文1", "原文2"]

    def test_to_str_for_dense_retrieval(self):
        """向量检索字符串测试"""
        memory = Memory(user_id="user_001", content="这是一条记忆内容")
        result = memory.to_str_for_dense_retrieval()
        assert result == "这是一条记忆内容"

    def test_to_str_for_dedup(self):
        """去重字符串测试"""
        memory = Memory(
            user_id="user_001",
            content="测试内容",
            occurred_string="2024-12-25",
        )
        result = memory.to_str_for_dedup()

        # 应该是有效的 JSON
        parsed = json.loads(result)
        assert parsed["content"] == "测试内容"
        assert parsed["occurred_string"] == "2024-12-25"

    def test_to_str_for_dedup_no_time(self):
        """去重字符串测试（无时间）"""
        memory = Memory(user_id="user_001", content="测试内容")
        result = memory.to_str_for_dedup()

        parsed = json.loads(result)
        assert parsed["content"] == "测试内容"
        assert parsed["occurred_string"] is None

    def test_get_keywords(self):
        """获取关键词测试"""
        memory = Memory(
            user_id="user_001",
            content="测试",
            keywords=["a", "b", "c"],
        )
        result = memory.get_keywords()

        assert result == ["a", "b", "c"]
        # 确保返回的是副本
        result.append("d")
        assert memory.keywords == ["a", "b", "c"]


class TestScoredMemory:
    """ScoredMemory 测试"""

    def test_create(self):
        """创建测试"""
        memory = Memory(user_id="user_001", content="测试")
        scored = ScoredMemory(memory=memory, score=0.95)

        assert scored.memory.content == "测试"
        assert scored.score == 0.95
