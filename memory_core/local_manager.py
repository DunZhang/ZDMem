"""基于本地 JSON 文件的 MemoryDatabaseManager 实现"""

import os
import json
import uuid
from pathlib import Path
import numpy as np

from .models import Memory, ScoredMemory
from .llm import get_embedding


class LocalFileMemoryManager:
    """
    基于本地 JSON 文件的 MemoryDatabaseManager 实现

    存储结构：
    {MEMORY_DATA_DIR}/{user_id}/
    ├── memories.json      # Memory 列表
    └── embeddings.json    # {memory_id: [float, ...]}
    """

    def __init__(self, data_dir: str | None = None):
        """
        初始化本地文件管理器

        Args:
            data_dir: 数据存储根目录，None 则从 MEMORY_DATA_DIR 环境变量获取
        """
        self.data_dir = Path(data_dir or os.getenv("MEMORY_DATA_DIR", "./data"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "voyage/voyage-3-large")

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
        threshold = score_threshold if score_threshold is not None else float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.5"))

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

    def get_all_memories(self, user_id: str) -> list[Memory]:
        """
        获取用户所有记忆

        Args:
            user_id: 用户标识

        Returns:
            Memory 列表
        """
        return self._load_memories(user_id)
