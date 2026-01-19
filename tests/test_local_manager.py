"""本地文件管理器测试"""

import tempfile
import pytest
from memory_core.local_manager import LocalFileMemoryManager
from memory_core.models import Memory


class TestLocalFileMemoryManager:
    """LocalFileMemoryManager 测试"""

    @pytest.fixture
    def manager(self):
        """创建临时目录管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalFileMemoryManager(data_dir=tmpdir)

    @pytest.fixture
    def sample_memory(self):
        """示例记忆"""
        return Memory(
            user_id="user1",
            content="小美喜欢吃火锅，特别是番茄锅底",
            keywords=["小美", "火锅", "番茄锅底"],
            occurred_string="2024-12-20",
            ref_dial_ids=["text1"],
            ref_contents=["小美喜欢吃火锅"],
        )

    def test_add_and_get_memory(self, manager, sample_memory):
        """添加和获取记忆"""
        saved = manager.add_memory("user1", sample_memory)

        # 检查 ID 被赋值
        assert saved.id != ""

        # 获取记忆
        loaded = manager.get_memory("user1", saved.id)
        assert loaded is not None
        assert loaded.content == sample_memory.content
        assert loaded.keywords == sample_memory.keywords

    def test_get_nonexistent_memory(self, manager):
        """获取不存在的记忆"""
        result = manager.get_memory("user1", "nonexistent_id")
        assert result is None

    def test_search_by_vector(self, manager, sample_memory):
        """向量检索"""
        manager.add_memory("user1", sample_memory)

        results = manager.search_by_vector("user1", "小美喜欢什么食物", score_threshold=0.3)

        assert len(results) > 0
        assert results[0].memory.content == sample_memory.content
        print(f"\n向量检索分数: {results[0].score}")

    def test_search_by_keywords(self, manager, sample_memory):
        """关键词检索"""
        manager.add_memory("user1", sample_memory)

        results = manager.search_by_keywords("user1", ["小美", "火锅"])

        assert len(results) > 0
        assert results[0].memory.content == sample_memory.content
        print(f"\n关键词检索分数: {results[0].score}")

    def test_search_by_keywords_no_match(self, manager, sample_memory):
        """关键词检索无匹配"""
        manager.add_memory("user1", sample_memory)

        results = manager.search_by_keywords("user1", ["不存在的关键词"])
        assert len(results) == 0

    def test_get_all_keywords(self, manager):
        """获取所有关键词"""
        m1 = Memory(user_id="user1", content="测试1", keywords=["a", "b"])
        m2 = Memory(user_id="user1", content="测试2", keywords=["b", "c"])

        manager.add_memory("user1", m1)
        manager.add_memory("user1", m2)

        keywords = manager.get_all_keywords("user1")
        assert set(keywords) == {"a", "b", "c"}

    def test_synonyms_crud(self, manager):
        """同义词表 CRUD"""
        synonyms = {
            "老婆": ["妻子", "太太"],
            "iPhone": ["苹果手机"],
        }

        manager.save_synonyms("user1", synonyms)
        loaded = manager.get_synonyms("user1")

        assert loaded == synonyms

    def test_get_synonyms_empty(self, manager):
        """获取不存在的同义词表"""
        result = manager.get_synonyms("user1")
        assert result == {}

    def test_hybrid_search(self, manager, sample_memory):
        """混合检索"""
        manager.add_memory("user1", sample_memory)

        results = manager.hybrid_search("user1", "小美喜欢吃什么", score_threshold=0.3)

        assert len(results) > 0
        print(f"\n混合检索分数: {results[0].score}")

    def test_keyword_search_with_synonyms(self, manager, sample_memory):
        """带同义词的关键词检索"""
        manager.add_memory("user1", sample_memory)

        # 保存同义词
        manager.save_synonyms("user1", {"老婆": ["小美"]})

        # 用同义词搜索
        results = manager.search_by_keywords("user1", ["老婆"])

        assert len(results) > 0
        print(f"\n同义词检索匹配: {results[0].memory.content}")

    def test_multiple_users(self, manager):
        """多用户隔离"""
        m1 = Memory(user_id="user1", content="用户1的记忆", keywords=["test"])
        m2 = Memory(user_id="user2", content="用户2的记忆", keywords=["test"])

        manager.add_memory("user1", m1)
        manager.add_memory("user2", m2)

        # 各自只能看到自己的记忆
        user1_keywords = manager.get_all_keywords("user1")
        user2_keywords = manager.get_all_keywords("user2")

        # 检索应该隔离
        results1 = manager.search_by_keywords("user1", ["test"])
        results2 = manager.search_by_keywords("user2", ["test"])

        assert results1[0].memory.content == "用户1的记忆"
        assert results2[0].memory.content == "用户2的记忆"
