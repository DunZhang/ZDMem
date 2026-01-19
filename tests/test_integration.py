"""集成测试"""

import tempfile
import pytest
from memory_core.main import process_memories
from memory_core.local_manager import LocalFileMemoryManager


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def manager(self):
        """创建临时目录管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalFileMemoryManager(data_dir=tmpdir)

    def test_full_pipeline(self, test_text, test_user_id, test_text_id, manager):
        """完整流程测试"""
        # 第一次处理
        memories = process_memories(
            text=test_text,
            text_id=test_text_id,
            user_id=test_user_id,
            manager=manager,
        )

        print(f"\n第一次抽取: {len(memories)} 条记忆")
        for m in memories:
            print(f"  - {m.content[:50]}...")

        assert len(memories) > 0

        # 保存记忆
        for m in memories:
            manager.add_memory(test_user_id, m)

        # 验证保存
        all_keywords = manager.get_all_keywords(test_user_id)
        print(f"\n保存后的关键词: {all_keywords}")
        assert len(all_keywords) > 0

    def test_search_after_save(self, test_text, test_user_id, test_text_id, manager):
        """保存后检索测试"""
        # 处理并保存
        memories = process_memories(
            text=test_text,
            text_id=test_text_id,
            user_id=test_user_id,
            manager=manager,
        )

        for m in memories:
            manager.add_memory(test_user_id, m)

        # 混合检索
        results = manager.hybrid_search(test_user_id, "小美喜欢吃什么", score_threshold=0.3)

        print(f"\n检索 '小美喜欢吃什么' 结果:")
        for r in results:
            print(f"  - 分数: {r.score:.3f}, 内容: {r.memory.content[:50]}...")

        assert len(results) > 0

    def test_incremental_processing(self, test_user_id, manager):
        """增量处理测试"""
        # 第一段文本
        text1 = "2024年12月10日，我和老婆小美去吃海底捞，她喜欢番茄锅底。"
        memories1 = process_memories(
            text=text1,
            text_id="text_001",
            user_id=test_user_id,
            manager=manager,
        )

        for m in memories1:
            manager.add_memory(test_user_id, m)

        print(f"\n第一次: 抽取 {len(memories1)} 条")

        # 第二段文本（有重叠信息）
        text2 = "小美今天又说想吃火锅了，她还是最喜欢番茄锅底。"
        memories2 = process_memories(
            text=text2,
            text_id="text_002",
            user_id=test_user_id,
            manager=manager,
        )

        print(f"第二次: 抽取 {len(memories2)} 条")
        for m in memories2:
            print(f"  - {m.content}")

        # 由于去重，第二次应该抽取较少的新信息
        # 注意：这不是硬性要求，取决于 LLM 的判断

    def test_synonym_workflow(self, test_user_id, manager):
        """同义词工作流测试"""
        from memory_core.synonym import generate_synonyms
        from memory_core.models import Memory

        # 添加记忆
        m = Memory(
            user_id=test_user_id,
            content="我老婆喜欢吃火锅",
            keywords=["老婆", "火锅"],
        )
        manager.add_memory(test_user_id, m)

        # 生成同义词
        all_keywords = manager.get_all_keywords(test_user_id)
        synonyms = generate_synonyms(all_keywords)

        print(f"\n生成的同义词: {synonyms}")

        # 保存同义词
        manager.save_synonyms(test_user_id, synonyms)

        # 使用同义词搜索
        if "妻子" in synonyms.get("老婆", []):
            results = manager.search_by_keywords(test_user_id, ["妻子"])
            print(f"用 '妻子' 搜索结果: {len(results)} 条")
            assert len(results) > 0

    def test_empty_database(self, test_text, test_user_id, test_text_id, manager):
        """空数据库处理测试"""
        # 空数据库应该能正常处理
        memories = process_memories(
            text=test_text,
            text_id=test_text_id,
            user_id=test_user_id,
            manager=manager,
        )

        assert len(memories) > 0

    def test_multiple_texts(self, test_user_id, manager):
        """多文本处理测试"""
        texts = [
            ("text_001", "我今天去了北京出差，见了客户张总。"),
            ("text_002", "明天要和张总一起去上海开会。"),
            ("text_003", "张总是我们最重要的客户之一。"),
        ]

        all_memories = []
        for text_id, text in texts:
            memories = process_memories(
                text=text,
                text_id=text_id,
                user_id=test_user_id,
                manager=manager,
            )
            for m in memories:
                manager.add_memory(test_user_id, m)
            all_memories.extend(memories)

        print(f"\n处理 {len(texts)} 段文本，共抽取 {len(all_memories)} 条记忆")

        # 检索张总相关
        results = manager.hybrid_search(test_user_id, "张总是谁", score_threshold=0.3)
        print(f"检索 '张总是谁' 结果: {len(results)} 条")

        assert len(results) > 0
