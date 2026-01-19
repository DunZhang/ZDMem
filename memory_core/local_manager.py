"""基于本地 JSON 文件的 MemoryDatabaseManager 实现"""

import json
import uuid
from pathlib import Path
import numpy as np

from .models import Memory, ScoredMemory
from .config import (
    get_memory_data_dir,
    get_embedding_model,
    get_vector_score_threshold,
    get_rrf_k,
)
from .llm import get_embedding
from .keyword import extract_keywords


class LocalFileMemoryManager:
    """
    基于本地 JSON 文件的 MemoryDatabaseManager 实现

    存储结构：
    {MEMORY_DATA_DIR}/{user_id}/
    ├── memories.json      # Memory 列表
    ├── embeddings.json    # {memory_id: [float, ...]}
    └── synonyms.json      # 同义词表
    """

    def __init__(self, data_dir: str | None = None):
        """
        初始化本地文件管理器

        Args:
            data_dir: 数据存储根目录，None 则从 MEMORY_DATA_DIR 环境变量获取
        """
        self.data_dir = Path(data_dir or get_memory_data_dir())
        self.embedding_model = get_embedding_model()

    def _get_user_dir(self, user_id: str) -> Path:
        """获取用户数据目录，不存在则创建"""
        user_dir = self.data_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _load_memories(self, user_id: str) -> list[Memory]:
        """加载用户所有记忆"""
        memories_path = self._get_user_dir(user_id) / "memories.json"
        if not memories_path.exists():
            return []

        with open(memories_path, encoding="utf-8") as f:
            data = json.load(f)

        return [self._dict_to_memory(d) for d in data]

    def _save_memories(self, user_id: str, memories: list[Memory]) -> None:
        """保存用户所有记忆"""
        memories_path = self._get_user_dir(user_id) / "memories.json"
        data = [self._memory_to_dict(m) for m in memories]

        with open(memories_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_embeddings(self, user_id: str) -> dict[str, list[float]]:
        """加载用户所有 embedding"""
        embeddings_path = self._get_user_dir(user_id) / "embeddings.json"
        if not embeddings_path.exists():
            return {}

        with open(embeddings_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_embeddings(self, user_id: str, embeddings: dict[str, list[float]]) -> None:
        """保存用户所有 embedding"""
        embeddings_path = self._get_user_dir(user_id) / "embeddings.json"

        with open(embeddings_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)

    def _memory_to_dict(self, memory: Memory) -> dict:
        """Memory 转字典"""
        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "keywords": memory.keywords,
            "occurred_string": memory.occurred_string,
            "occurred_at": memory.occurred_at,
            "ref_dial_ids": memory.ref_dial_ids,
            "ref_contents": memory.ref_contents,
        }

    def _dict_to_memory(self, data: dict) -> Memory:
        """字典转 Memory"""
        return Memory(
            id=data["id"],
            user_id=data["user_id"],
            content=data["content"],
            keywords=data.get("keywords", []),
            occurred_string=data.get("occurred_string"),
            occurred_at=data.get("occurred_at"),
            ref_dial_ids=data.get("ref_dial_ids", []),
            ref_contents=data.get("ref_contents", []),
        )

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ===== Protocol 方法 =====

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
        threshold = get_vector_score_threshold(score_threshold)

        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        if not memories or not embeddings:
            return []

        query_embedding = get_embedding(query, self.embedding_model, input_type="query")

        scored = []
        for memory in memories:
            if memory.id not in embeddings:
                continue

            score = self._cosine_similarity(query_embedding, embeddings[memory.id])
            if score >= threshold:
                scored.append(ScoredMemory(memory=memory, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search_by_keywords(
        self,
        user_id: str,
        keywords: list[str],
        top_k: int = 10
    ) -> list[ScoredMemory]:
        """
        关键词精确匹配（含同义词扩展）

        Args:
            user_id: 用户标识
            keywords: 关键词列表
            top_k: 返回数量

        Returns:
            按相关性降序排列的 ScoredMemory 列表
        """
        if not keywords:
            return []

        memories = self._load_memories(user_id)
        synonyms = self.get_synonyms(user_id)

        # 扩展查询关键词
        expanded_keywords = set(keywords)
        for keyword in keywords:
            # 如果 keyword 是规范词，添加其同义词
            if keyword in synonyms:
                expanded_keywords.update(synonyms[keyword])
            # 反向查找：如果 keyword 是某个规范词的同义词
            for canonical, syns in synonyms.items():
                if keyword in syns:
                    expanded_keywords.add(canonical)
                    expanded_keywords.update(syns)

        scored = []
        for memory in memories:
            memory_keywords = set(memory.get_keywords())

            # 扩展记忆的关键词
            expanded_memory_keywords = set(memory_keywords)
            for mk in memory_keywords:
                if mk in synonyms:
                    expanded_memory_keywords.update(synonyms[mk])

            # 计算命中数
            hits = len(expanded_keywords & expanded_memory_keywords)
            if hits > 0:
                score = hits / len(keywords)
                scored.append(ScoredMemory(memory=memory, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def get_all_keywords(self, user_id: str) -> list[str]:
        """
        获取记忆库中所有关键词（去重）

        Args:
            user_id: 用户标识

        Returns:
            关键词列表
        """
        memories = self._load_memories(user_id)
        all_keywords = set()

        for memory in memories:
            all_keywords.update(memory.get_keywords())

        return list(all_keywords)

    def get_synonyms(self, user_id: str) -> dict[str, list[str]]:
        """
        获取同义词表

        Args:
            user_id: 用户标识

        Returns:
            格式：{canonical_keyword: [synonym1, synonym2, ...]}
        """
        synonyms_path = self._get_user_dir(user_id) / "synonyms.json"
        if not synonyms_path.exists():
            return {}

        with open(synonyms_path, encoding="utf-8") as f:
            return json.load(f)

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
        rrf_k = get_rrf_k()

        # 向量检索
        vector_results = self.search_by_vector(
            user_id, query, top_k=top_k * 2, score_threshold=score_threshold
        )

        # 关键词抽取并检索
        keywords = extract_keywords(query)
        keyword_results = self.search_by_keywords(user_id, keywords, top_k=top_k * 2)

        # RRF 融合
        rrf_scores: dict[str, float] = {}

        for rank, scored in enumerate(vector_results, start=1):
            memory_id = scored.memory.id
            rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + 1 / (rrf_k + rank)

        for rank, scored in enumerate(keyword_results, start=1):
            memory_id = scored.memory.id
            rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + 1 / (rrf_k + rank)

        # 构建结果
        memory_map: dict[str, Memory] = {}
        for scored in vector_results + keyword_results:
            memory_map[scored.memory.id] = scored.memory

        results = [
            ScoredMemory(memory=memory_map[mid], score=score)
            for mid, score in rrf_scores.items()
        ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ===== CRUD 方法 =====

    def add_memory(self, user_id: str, memory: Memory) -> Memory:
        """
        添加记忆

        Args:
            user_id: 用户标识
            memory: 记忆对象

        Returns:
            添加后的记忆（包含生成的 id）
        """
        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        # 分配 ID
        if not memory.id:
            memory.id = str(uuid.uuid4())

        # 生成 embedding
        embedding_vector = get_embedding(
            memory.to_str_for_dense_retrieval(),
            self.embedding_model,
            input_type="document"
        )

        memories.append(memory)
        embeddings[memory.id] = embedding_vector

        self._save_memories(user_id, memories)
        self._save_embeddings(user_id, embeddings)

        return memory

    def get_memory(self, user_id: str, memory_id: str) -> Memory | None:
        """
        获取单条记忆

        Args:
            user_id: 用户标识
            memory_id: 记忆 ID

        Returns:
            Memory 对象，不存在则返回 None
        """
        memories = self._load_memories(user_id)

        for memory in memories:
            if memory.id == memory_id:
                return memory

        return None

    def save_synonyms(self, user_id: str, synonyms: dict[str, list[str]]) -> None:
        """
        保存同义词表

        Args:
            user_id: 用户标识
            synonyms: 同义词表
        """
        synonyms_path = self._get_user_dir(user_id) / "synonyms.json"

        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)
