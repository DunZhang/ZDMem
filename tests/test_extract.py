"""记忆抽取测试（真实 LLM 调用）"""

import pytest
from memory_core.extract import extract_memories


class TestExtractMemories:
    """extract_memories 测试"""

    def test_basic_extraction(self, test_text, test_user_id):
        """基本抽取测试"""
        memories = extract_memories(
            text=test_text,
            user_id=test_user_id,
        )

        assert isinstance(memories, list)
        assert len(memories) > 0

        print(f"\n抽取的记忆数量: {len(memories)}")
        for i, m in enumerate(memories):
            print(f"\n记忆 {i + 1}:")
            print(f"  内容: {m.content}")
            print(f"  关键词: {m.keywords}")
            print(f"  时间: {m.occurred_string}")
            print(f"  引用对话ID: {m.ref_dial_ids}")
            print(f"  引用内容: {m.ref_contents}")

    def test_memory_structure(self, test_text, test_user_id):
        """记忆结构测试"""
        memories = extract_memories(
            text=test_text,
            user_id=test_user_id,
        )

        for m in memories:
            # 验证必需字段
            assert m.user_id == test_user_id
            assert isinstance(m.content, str)
            assert len(m.content) > 0
            assert isinstance(m.keywords, list)
            assert m.id == ""  # 新记忆 id 为空

            # 验证引用
            assert len(m.ref_dial_ids) > 0
            assert len(m.ref_contents) > 0
            assert len(m.ref_dial_ids) == len(m.ref_contents)

    def test_time_extraction(self, test_text, test_user_id):
        """时间抽取测试"""
        memories = extract_memories(
            text=test_text,
            user_id=test_user_id,
        )

        # 检查是否有记忆包含时间
        has_time = [m for m in memories if m.occurred_string]
        print(f"\n包含时间的记忆: {len(has_time)}")
        for m in has_time:
            print(f"  时间: {m.occurred_string}, 内容: {m.content[:50]}...")

        # 测试文本中有明确的日期，应该至少有一条记忆包含时间
        assert len(has_time) >= 1

    def test_empty_text(self, test_user_id):
        """空文本"""
        memories = extract_memories(
            text="",
            user_id=test_user_id,
        )
        assert isinstance(memories, list)
