# 记忆系统设计 v5

## 1. 概述

本文档描述大模型记忆系统核心模块的详细设计，面向 vibe coding 场景。

**设计原则：**
- 核心模块只实现业务逻辑，不依赖具体数据库
- 调用方通过实现 `MemoryDatabaseManager` 接口对接自己的技术栈
- 依赖倒置：核心模块依赖抽象接口，而非具体实现

**技术选型：**
- 语言：Python 3.10+
- 包管理：uv
- LLM 调用：LiteLLM
- Prompt 管理：Jinja2 模板（本地 .j2 文件）
- 配置管理：python-dotenv
- 默认模型：`openrouter/google/gemini-3-flash-preview`

## 2. 模块划分

| 模块 | 职责 | 实现方 |
|------|------|--------|
| 检索字符串生成 | 文本 → 多个检索字符串 | 核心模块 |
| 记忆更新 | 文本 + 已有记忆 → 更新/删除操作 | 核心模块 |
| 记忆插入 | 文本 + 已有记忆 → 新记忆列表 | 核心模块 |
| 关键词抽取 | 文本 → 关键词列表 | 核心模块 |
| 同义词生成 | 关键词列表 → 同义词映射表 | 核心模块 |
| 记忆存储/检索 | 增删改查、向量索引、倒排索引、同义词管理 | 调用方 |

## 3. 数据结构

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json

class MemoryAction(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class MemoryReference:
    """记忆来源引用"""
    text_id: str
    lines: list[int]  # 行号列表（1-based），如 [4, 5, 6]

@dataclass
class Memory:
    """记忆实体"""
    user_id: str
    content: str
    id: str = ""                          # 新记忆为空，由调用方赋值
    keywords: list[str] = field(default_factory=list)  # 关键词列表，用于检索和索引
    occurred_string: str | None = None    # 可残缺的时间字符串，如 "2024-12-25"、"2024-12"、"2024-12-25T03"
    occurred_at: str | None = None        # ISO 8601 格式，如 "2024-12-25T06:30:45.123456+00:00"
    created_at: str = ""                  # ISO 8601 格式
    updated_at: str = ""                  # ISO 8601 格式
    references: list[MemoryReference] = field(default_factory=list)

    def to_str_for_dense_retrieval(self) -> str:
        """
        返回用于向量编码检索的字符串

        Returns:
            content 字段内容
        """
        return self.content

    def to_str_for_update(self) -> str:
        """
        返回用于更新判断时展示在 prompt 中的 JSON 字符串

        只包含 content、keywords、occurred_string 字段

        Returns:
            JSON 格式字符串
        """
        return json.dumps({
            "content": self.content,
            "keywords": self.keywords,
            "occurred_string": self.occurred_string,
        }, ensure_ascii=False, indent=2)

    def to_str_for_insert(self) -> str:
        """
        返回用于插入新记忆时展示在 prompt 中的 JSON 字符串

        只包含 content 和 occurred_string 字段，用于让 LLM 判断避免重复

        Returns:
            JSON 格式字符串
        """
        return json.dumps({
            "content": self.content,
            "occurred_string": self.occurred_string,
        }, ensure_ascii=False, indent=2)

    def get_keywords(self) -> list[str]:
        """
        返回与此记忆相关的所有关键词

        Returns:
            关键词列表
        """
        return self.keywords.copy()

@dataclass
class ScoredMemory:
    """带分数的记忆（用于检索结果）"""
    memory: Memory
    score: float

@dataclass 
class MemoryOperation:
    """记忆操作（统一表示创建、更新、删除三种操作）"""
    action: MemoryAction  # CREATE, UPDATE 或 DELETE
    memory_id: str = ""   # 被操作的记忆 ID，CREATE 时为空
    # CREATE 时使用
    created_memory: Memory | None = None
    # UPDATE 时使用
    update_dict: dict | None = None       # 更新字典，只包含需要更新的字段
    updated_memory: Memory | None = None  # 应用 update_dict 后的完整记忆

@dataclass
class MemoryOperationResult:
    """记忆操作结果"""
    created: list[MemoryOperation]  # action=CREATE 的操作列表
    updated: list[MemoryOperation]  # action=UPDATE 的操作列表
    deleted: list[MemoryOperation]  # action=DELETE 的操作列表
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 记忆唯一标识，新记忆为空，由调用方赋值 |
| user_id | str | 用户标识 |
| content | str | 记忆内容，自然语言文本 |
| keywords | list[str] | 从 content 中抽取的关键词列表，包括人物、地点、活动、物品、情感、概念等 |
| occurred_string | str \| None | 可残缺的时间字符串，由 LLM 根据实际情况填写，如 "2024-12-25"、"2024-12"、"2024-12-25T03"，这是最准确的时间表示 |
| occurred_at | str \| None | 完整 ISO 8601 时间字符串，CREATE 时由核心模块根据 occurred_string 自动补全；UPDATE 时由调用方根据 updated_memory 自行处理 |
| created_at | str | 记忆创建时间，ISO 8601 格式，由调用方在存储时填充 |
| updated_at | str | 记忆最后更新时间，ISO 8601 格式，由调用方在存储时填充 |
| references | list[MemoryReference] | 记忆来源，记录从哪段文本的哪些行抽取（行号列表） |

**MemoryOperation 字段说明：**

| action | 使用字段 | 说明 |
|--------|----------|------|
| CREATE | created_memory | 新创建的记忆，memory_id 和 created_memory.id 均为空，由调用方赋值后存储 |
| UPDATE | memory_id, update_dict, updated_memory | update_dict 只包含需要更新的字段；updated_memory 为应用更新后的完整记忆 |
| DELETE | memory_id | 只需要记忆 ID |

**时间格式说明：**

所有时间字段（除 `occurred_string` 外）均使用 ISO 8601 格式，默认时区为 UTC：

```
2024-12-25T06:30:45.123456+00:00
```

`occurred_string` 字段允许残缺格式，常见形式：
- `"2024"` - 只知道年份
- `"2024-12"` - 精确到月
- `"2024-12-25"` - 精确到日
- `"2024-12-25T03"` - 精确到小时
- `"2024-12-25T03:30"` - 精确到分钟
- `None` - 无法从文本中提取时间

## 4. 接口定义

调用方需实现 `MemoryDatabaseManager` 协议：

```python
from typing import Protocol

class MemoryDatabaseManager(Protocol):
    def search_by_vector(self, user_id: str, query: str, top_k: int = 10) -> list[ScoredMemory]:
        """
        向量检索
        
        Args:
            user_id: 用户标识
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            按相似度降序排列的 ScoredMemory 列表，包含分数用于后续过滤
        """
        ...
    
    def search_by_keywords(self, user_id: str, keywords: list[str], top_k: int = 10) -> list[ScoredMemory]:
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
            按相关性降序排列的 ScoredMemory 列表（任一关键词命中即返回）
            分数可基于命中关键词数量、权重等计算
        """
        ...

    def get_all_keywords(self, user_id: str) -> list[str]:
        """
        获取记忆库中所有关键词（去重）

        Args:
            user_id: 用户标识

        用于同义词生成
        """
        ...
    
    def get_synonyms(self, user_id: str) -> dict[str, list[str]]:
        """
        获取同义词表
        
        Args:
            user_id: 用户标识
        
        Returns:
            格式：{canonical_entity: [synonym1, synonym2, ...]}
            示例：{"老婆": ["妻子", "太太", "媳妇"]}
        """
        ...
    
    def hybrid_search(self, user_id: str, query: str, top_k: int = 10) -> list[ScoredMemory]:
        """
        混合检索：向量检索 + 关键词检索 + RRF 融合

        用于聊天时检索相关记忆，与 process_memories 中的检索流程不同：
        - 本方法用于查询场景，需要 RRF 融合排序并截断到 top_k
        - process_memories 中的检索用于记忆更新/插入，需要尽可能全面召回，不做 RRF 融合

        Args:
            user_id: 用户标识
            query: 查询文本
            top_k: 返回数量

        Returns:
            RRF 融合后的 ScoredMemory 列表，按分数降序排列

        实现要点：
            1. 调用 search_by_vector 获取向量检索结果
            2. 从 query 抽取关键词，调用 search_by_keywords 获取关键词检索结果
            3. 使用 RRF 公式融合两路结果：score = 1/(k+rank1) + 1/(k+rank2)，k 从 RRF_K 环境变量获取，默认 60
        """
        ...
```

## 5. 算法流程

### 5.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        process_memories                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: 生成检索字符串                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 输入: 用户文本                                                           │ │
│ │ 输出: search_strings = [s1, s2, ..., sn, 原始文本]                       │ │
│ │ 说明: LLM 生成多个检索字符串，必须覆盖全面，宁可冗余不可遗漏               │ │
│ │       最后将原始文本也加入作为兜底                                        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: 混合检索（对每个 search_string）                                     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ for each search_string:                                                 │ │
│ │   ├── Dense 检索: search_by_vector(search_string) → ScoredMemory[]     │ │
│ │   ├── 关键词抽取: extract_entities(search_string) → keywords[]         │ │
│ │   └── 关键词检索: search_by_keywords(keywords) → ScoredMemory[]        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: 结果合并与排序                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 1. 合并所有 search_string 的检索结果（向量检索 + 关键词检索）             │ │
│ │ 2. 按 memory.id 去重                                                    │ │
│ │ 3. 按 occurred_at 升序排序（早的在前，None 放最后）                       │ │
│ │ 输出: related_memories: list[Memory]                                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: 分批更新记忆                                                         │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ batch_size = 10 (默认)                                                  │ │
│ │ updated_ids = set()                                                     │ │
│ │ deleted_ids = set()                                                     │ │
│ │ all_update_ops = []                                                     │ │
│ │ all_delete_ops = []                                                     │ │
│ │                                                                         │ │
│ │ for batch in chunk(related_memories, batch_size):                       │ │
│ │   ├── 构建 prompt：                                                     │ │
│ │   │   - 用户文本                                                        │ │
│ │   │   - 每条记忆用 <memory_idx=i>{memory.to_str_for_update()}</memory>   │ │
│ │   ├── LLM 输出：                                                        │ │
│ │   │   [{"update_idx": i, "update_dict": {...}}, {"delete_idx": j}, ...] │ │
│ │   ├── 记录 updated_ids 和 deleted_ids                                   │ │
│ │   ├── 对于 UPDATE 操作，计算 updated_memory = apply_update(memory, update_dict) │ │
│ │   └── 收集 all_update_ops 和 all_delete_ops                             │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 5: 构建插入上下文                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ context_memories = []                                                   │ │
│ │ for memory in related_memories:                                         │ │
│ │   if memory.id in deleted_ids:                                          │ │
│ │     continue  # 跳过已删除的                                             │ │
│ │   if memory.id in updated_ids:                                          │ │
│ │     # 使用更新后的版本                                                    │ │
│ │     updated_memory = get_updated_memory(memory.id)                      │ │
│ │     context_memories.append(updated_memory)                             │ │
│ │   else:                                                                 │ │
│ │     context_memories.append(memory)                                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 6: 插入新记忆                                                           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 预处理：给用户文本每行添加行号前缀 "[行号]: {内容}"                        │ │
│ │ 构建 prompt：                                                           │ │
│ │   - 带行号的用户文本                                                     │ │
│ │   - 已有记忆列表（使用 memory.to_str_for_insert()）                      │ │
│ │ LLM 输出：                                                              │ │
│ │   新记忆列表，每条包含 content, keywords, occurred_string, lines 字段    │ │
│ │ 要求：生成的新记忆不能与已有记忆重复                                      │ │
│ │ 为每条新记忆构建 MemoryOperation(action=CREATE, created_memory=...)     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 7: 返回结果                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ return MemoryOperationResult(                                           │ │
│ │   created=all_create_ops,   # CREATE 操作列表                           │ │
│ │   updated=all_update_ops,   # UPDATE 操作列表                           │ │
│ │   deleted=all_delete_ops    # DELETE 操作列表                           │ │
│ │ )                                                                       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键设计说明

#### 5.2.1 为什么要生成检索字符串？

直接用用户原文做向量检索效果差，原因：
1. 用户文本可能很长，包含多个话题
2. 向量模型对长文本的表示能力有限
3. 不同话题混在一起会稀释语义

解决方案：让 LLM 分析用户文本，生成多个聚焦的检索字符串，每个覆盖一个话题或关键信息点。

#### 5.2.2 为什么要把原始文本也作为检索字符串？

作为兜底策略，防止 LLM 生成的检索字符串遗漏重要信息。

#### 5.2.3 为什么分批处理更新？

1. 避免 prompt 过长导致 LLM 效果下降
2. 控制单次 LLM 调用的复杂度
3. 每条记忆只出现在一个批次中，不会重复处理

#### 5.2.4 为什么插入要用到更新后的记忆？

防止重复抽取：
- 如果某条记忆被更新了，用更新后的版本做去重判断
- 如果某条记忆被删除了，不参与去重（可以重新抽取类似内容）

#### 5.2.5 为什么 UPDATE 操作要同时保留 update_dict 和 updated_memory？

- `update_dict`：记录本次更新了哪些字段，便于调用方进行增量更新、审计日志等
- `updated_memory`：提供更新后的完整记忆，便于调用方直接使用，无需自行合并

## 6. 核心模块 API

### 6.1 LLM 调用规范

所有调用大模型的地方遵循以下规范：

- **重试机制**：LLM 调用失败时自动重试，可配置重试次数
- **结构化输出**：使用 LiteLLM 的 `response_format={"type": "json_object"}` 确保输出 JSON
- **输出解析**：处理 \`\`\`json 或 \`\`\` 包裹的情况，使用 `json_repair` 库修复常见 JSON 格式问题

### 6.2 模型选择逻辑

所有需要 LLM 的函数都支持 `model` 参数，选择逻辑如下：

```
参数值 > 对应环境变量 > DEFAULT_MODEL 环境变量
```

例如 `generate_search_strings` 函数的模型选择：
1. 如果传入 `search_string_gen_model` 参数，使用该值
2. 否则读取 `SEARCH_STRING_GEN_MODEL` 环境变量
3. 如果环境变量未设置，使用 `DEFAULT_MODEL` 环境变量

### 6.3 记忆处理（主入口）

```python
def process_memories(
    text: str,
    text_id: str,
    user_id: str,
    manager: MemoryDatabaseManager,
    *,
    search_top_k: int | None = None,
    update_batch_size: int | None = None,
    search_string_gen_model: str | None = None,
    memory_update_model: str | None = None,
    memory_insert_model: str | None = None,
    entity_extract_model: str | None = None,
) -> MemoryOperationResult:
    """
    从文本处理记忆：创建新记忆、更新已有记忆、删除失效记忆
    
    Args:
        text: 输入文本（对话记录、用户笔记等）
        text_id: 文本唯一标识，用于记录记忆来源
        user_id: 用户标识
        manager: 记忆数据库管理器
        search_top_k: 每个检索字符串的返回数量，None 则从 SEARCH_TOP_K 环境变量获取，默认 10
        update_batch_size: 更新时每批处理的记忆数量，None 则从 UPDATE_BATCH_SIZE 环境变量获取，默认 10
        search_string_gen_model: 检索字符串生成模型，None 则从 SEARCH_STRING_GEN_MODEL 环境变量获取
        memory_update_model: 记忆更新模型，None 则从 MEMORY_UPDATE_MODEL 环境变量获取
        memory_insert_model: 记忆插入模型，None 则从 MEMORY_INSERT_MODEL 环境变量获取
        entity_extract_model: 关键词抽取模型，None 则从 ENTITY_EXTRACT_MODEL 环境变量获取
        
    Returns:
        MemoryOperationResult，包含：
        - created: CREATE 操作列表，每个操作包含 created_memory
        - updated: UPDATE 操作列表，每个操作包含 update_dict 和 updated_memory
        - deleted: DELETE 操作列表，每个操作只包含 memory_id
    """
```

### 6.4 检索字符串生成

```python
def generate_search_strings(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    从用户文本生成多个检索字符串
    
    要点：
    - 必须覆盖文本中所有可能相关的话题和信息点
    - 宁可冗余生成，也不能遗漏
    - 返回的列表不包含原始文本（由调用方自行添加）
    
    Args:
        text: 用户输入文本
        model: 使用的 LLM 模型，None 则从 SEARCH_STRING_GEN_MODEL 或 DEFAULT_MODEL 获取
        
    Returns:
        检索字符串列表
    """
```

### 6.5 关键词抽取

```python
def extract_entities(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    从文本抽取关键词

    关键词范围比传统实体更广，包括：
    - 人物、地点、组织、产品等命名实体
    - 活动、物品、情感词、概念等有意义的词汇

    支持两类输入：
    1. 长文本（对话记录、笔记等）- 用于记忆抽取流程
    2. 短文本（用户问题、检索字符串）- 用于检索时的关键词抽取

    LLM 需要根据文本特征自适应处理：
    - 长文本：全面抽取所有有意义的关键词
    - 短文本/问题：聚焦于查询意图相关的关键词

    Args:
        text: 输入文本
        model: 使用的 LLM 模型，None 则从 ENTITY_EXTRACT_MODEL 或 DEFAULT_MODEL 获取

    Returns:
        关键词列表（去重）
    """
```

### 6.6 同义词生成

```python
def generate_synonyms(
    keywords: list[str],
    *,
    batch_size: int | None = None,
    max_retries: int | None = None,
    model: str | None = None,
) -> dict[str, list[str]]:
    """
    批量生成同义词

    注意：本函数只负责生成同义词，不负责存储。调用方需要：
    1. 自行决定何时调用（如定期触发、新关键词达到一定数量时）
    2. 调用 manager.save_synonyms 存储结果
    3. 在 search_by_keywords 中使用同义词表进行扩展

    处理流程：
    1. 将 keywords 分批，每批 batch_size 个
    2. 逐批调用 LLM 生成同义词
    3. 纠错：检查结果中是否有关键词被遗漏，遗漏的重新生成
    4. 合并所有批次结果

    Args:
        keywords: 关键词列表
        batch_size: 每批处理数量，None 则从 SYNONYM_BATCH_SIZE 环境变量获取，默认 20
        max_retries: 纠错重试次数，None 则从 SYNONYM_MAX_RETRIES 环境变量获取，默认 2
        model: 使用的 LLM 模型，None 则从 SYNONYM_GEN_MODEL 或 DEFAULT_MODEL 获取

    Returns:
        同义词映射表，格式：{keyword: [synonym1, synonym2, ...]}
        - 没有同义词的关键词返回空数组
    """
```

## 7. Prompt 规范

使用 Jinja2 模板，存放于 `prompts/` 目录。Prompt 内容使用 XML + Markdown 结构化。

### 7.1 检索字符串生成 Prompt

文件：`prompts/generate_search_strings.j2`

**输入变量：**
- `text` - 用户输入文本

**输出要求：**
- JSON 数组，包含多个检索字符串
- 必须覆盖文本中所有可能相关的话题和信息点
- 宁可冗余，不可遗漏

**Prompt 结构：**
```xml
<role>检索字符串生成助手</role>
<task>
分析用户文本，生成多个用于记忆检索的搜索字符串。
</task>
<guidelines>
  - 每个检索字符串应聚焦于一个话题或信息点
  - 必须全面覆盖，不能遗漏任何可能相关的内容
  - 宁可生成冗余的检索字符串，也不能遗漏重要信息
  - 检索字符串应简洁、语义明确，适合向量检索
  - 人名、地名、专有名词等应单独生成检索字符串
</guidelines>
<output_format>
输出 JSON 数组：["检索字符串1", "检索字符串2", ...]
</output_format>
<input>
{{ text }}
</input>
```

### 7.2 记忆更新 Prompt

文件：`prompts/update_memories.j2`

**输入变量：**
- `text` - 用户输入文本
- `memories` - 已有记忆列表（XML 格式，每条用 `<memory_idx=N>` 包裹）

**输出要求：**
- JSON 数组，每个元素为更新或删除操作
- 更新操作：`{"update_idx": int, "update_dict": {...}}`
- 删除操作：`{"delete_idx": int}`
- update_dict 只包含需要更新的字段（content、keywords、occurred_string）

**Prompt 结构：**
```xml
<role>记忆更新助手</role>
<task>
分析用户文本，判断已有记忆中哪些需要更新或删除。
</task>
<guidelines>
  - 更新：当用户文本包含对已有记忆的补充、修正、更新时
  - 删除：当用户文本明确表示某记忆已失效或错误时
  - 不需要更新的记忆不要输出
  - update_dict 只包含需要更改的字段，不变的字段不要包含
  - occurred_string 应反映实际时间精度，如 "2024-12-25"、"2024-12" 等
</guidelines>
<output_format>
输出 JSON 数组：
[
  {"update_idx": 0, "update_dict": {"content": "新内容", "keywords": ["关键词1"]}},
  {"delete_idx": 2},
  ...
]
如果没有需要更新或删除的记忆，输出空数组：[]
</output_format>
<input>
<text>
{{ text }}
</text>
<existing_memories>
{{ memories }}
</existing_memories>
</input>
```

### 7.3 记忆插入 Prompt

文件：`prompts/insert_memories.j2`

**输入变量：**
- `text` - 用户输入文本（每行已添加行号前缀 `[行号]: 内容`）
- `text_id` - 文本标识
- `existing_memories` - 已有记忆列表（JSON 格式，使用 to_str_for_insert）

**输出要求：**
- JSON 数组，每个元素为新记忆
- 每条新记忆包含：content、keywords、occurred_string、lines
- lines 为行号数组 `[4, 5, 6]`，标记 content 对应的原文行号
- 生成的新记忆不能与已有记忆重复

**Prompt 结构：**
```xml
<role>记忆抽取助手</role>
<task>
从用户文本中抽取新的记忆，确保不与已有记忆重复。
</task>
<guidelines>
  - 一条记忆只记录一个事实/信息点
  - 宽松抽取：只要是有意义的信息都应该抽取
  - 不要抽取与已有记忆重复的内容
  - keywords 从 content 中抽取关键词，包括但不限于：人物、地点、组织、活动、物品、情感、概念等
  - occurred_string 根据文本中的时间信息填写，可以是残缺格式，无法确定时为 null
  - lines 标记 content 对应的原文行号，参考输入文本每行的 [行号] 前缀
</guidelines>
<additional_guidelines>
  - 图片消息：仔细阅读 <image_message> 内的文字描述，将图片内容抽取为记忆
  - 关系记忆：当文本出现 "we/us/our/我们" 时，明确标注涉及的所有人物
  - 相对时间转换：将 "last week/yesterday/last year" 等相对时间结合对话时间戳转换为绝对日期
  - 情感描述：保留原始情感词汇，避免过度概括（如 "felt tiny and in awe" 不要简化为 "amazing"）
  - 活动细节：具体内容完整保留，不要省略细节（如 "one-on-one mentoring and training, workshops and classes" 不要省略）
  - 时间+事件：确保事件和对应的时间信息同时被抽取到同一条记忆中
</additional_guidelines>
<output_format>
输出 JSON 数组：
[
  {
    "content": "记忆内容",
    "keywords": ["关键词1", "关键词2"],
    "occurred_string": "2024-12-25" 或 null,
    "lines": [4, 5]
  },
  ...
]
如果没有新记忆可抽取，输出空数组：[]
</output_format>
<input>
<text id="{{ text_id }}">
{{ text }}
</text>
<existing_memories>
{{ existing_memories }}
</existing_memories>
</input>
```

### 7.4 关键词抽取 Prompt

文件：`prompts/extract_entities.j2`

**输入变量：**
- `text` - 输入文本

**输出要求：**
- JSON 数组，包含所有关键词（去重）

**Prompt 结构：**
```xml
<role>关键词抽取助手</role>
<task>从文本中抽取有意义的关键词，用于记忆检索</task>
<guidelines>
  - 关键词类型：人物、地点、组织、活动、物品、情感词、概念、时间表达等
  - 示例：对于 "Melanie and her kids painted nature-inspired pictures together"
    应抽取：["Melanie", "kids", "painting", "pictures", "nature-inspired", "art", "together"]
  - 支持两类输入：长文本、短文本/问题
  - 长文本：全面抽取所有有意义的关键词
  - 短文本：聚焦查询意图相关的关键词
</guidelines>
<output_format>
输出 JSON 数组：["关键词1", "关键词2", ...]
</output_format>
<input>{{ text }}</input>
```

### 7.5 同义词生成 Prompt

文件：`prompts/generate_synonyms.j2`

**输入变量：**
- `keywords` - 关键词列表（JSON 数组）

**输出要求：**
- JSON 对象，key 为关键词，value 为同义词数组
- 无同义词的关键词返回空数组
- 必须覆盖输入的所有关键词

**Prompt 结构：**
```xml
<role>同义词生成助手</role>
<task>为关键词生成同义词、别名、口语化表达</task>
<guidelines>
  - 只生成真正的同义词，不要上位词/下位词
  - 考虑口语化、缩写、别名
  - 保持语言一致
  - 必须处理所有输入关键词
</guidelines>
<output_format>
输出 JSON 对象：
{
  "关键词1": ["同义词1", "同义词2"],
  "关键词2": [],
  ...
}
</output_format>
<input>{{ keywords }}</input>
```

## 8. 环境变量

文件：`.env`

```bash
# 默认模型（兜底）
DEFAULT_MODEL=openrouter/google/gemini-3-flash-preview

# 各模块专用模型（可选，未设置则使用 DEFAULT_MODEL）
SEARCH_STRING_GEN_MODEL=
ENTITY_EXTRACT_MODEL=
MEMORY_UPDATE_MODEL=
MEMORY_INSERT_MODEL=
SYNONYM_GEN_MODEL=

# Embedding 模型（Voyage）
EMBEDDING_MODEL=voyage/voyage-4
VOYAGE_API_KEY=your-voyage-api-key

# 数据存储目录（本地文件实现）
MEMORY_DATA_DIR=./data

# 参数默认值
SEARCH_TOP_K=10
UPDATE_BATCH_SIZE=10
SYNONYM_BATCH_SIZE=20
SYNONYM_MAX_RETRIES=2
RRF_K=60

# API Keys（根据使用的模型配置）
OPENROUTER_API_KEY=your-api-key

# 可选：其他模型提供商
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
```

## 9. 项目结构

```
memory-system/
├── pyproject.toml       # uv 项目配置
├── .env
├── prompts/
│   ├── generate_search_strings.j2
│   ├── update_memories.j2
│   ├── insert_memories.j2
│   ├── extract_entities.j2
│   └── generate_synonyms.j2
├── memory_core/
│   ├── __init__.py
│   ├── models.py            # 数据结构
│   ├── protocol.py          # Manager 接口
│   ├── config.py            # 环境变量和模型选择逻辑
│   ├── search.py            # 检索字符串生成
│   ├── update.py            # 记忆更新
│   ├── insert.py            # 记忆插入
│   ├── main.py              # 主入口 process_memories
│   ├── entity.py            # 关键词抽取（函数名保持 extract_entities 兼容）
│   ├── synonym.py           # 同义词生成
│   ├── llm.py               # LiteLLM 封装
│   └── local_manager.py     # 本地文件实现的 MemoryDatabaseManager
├── tests/
│   └── ...
└── examples/
    └── example_usage.py
```

## 10. 测试

使用 pytest 进行测试，环境变量从项目根目录 `.env` 文件加载。

**测试数据：**

编写一段真实的测试文本，包含多个话题、人物、时间等信息，例如：

```
2024年12月20日，我和老婆小美去深圳南山区的海底捞吃火锅，她特别喜欢番茄锅底。
第二天我们去了华强北买了一台 iPhone 16 Pro，花了 8999 元。
小美说她下周要去上海出差，大概待一周左右。
对了，我最近在学 Python 的 FastAPI 框架，感觉比 Flask 好用多了。
```

**测试范围：**
- 数据结构：Memory 类的各方法
- 核心模块：`generate_search_strings`、`extract_entities`、`generate_synonyms`、`process_memories`
- LocalFileMemoryManager：所有 Protocol 方法和 CRUD 方法

**完成标准：**

每个测试用例都必须通过，所有测试成功后才算最终完成。LLM 相关测试使用真实调用，不使用 mock。

## 11. 日志规范

使用 Python 标准 `logging` 模块，每个模块使用独立 logger：

```python
import logging

logger = logging.getLogger(__name__)

def some_function():
    logger.info("开始处理...")
    logger.debug("详细信息: %s", data)
    logger.warning("警告信息")
    logger.error("错误信息: %s", error)
```

**日志级别使用建议：**
- `DEBUG`：详细调试信息（如 LLM 输入输出、检索结果）
- `INFO`：关键流程节点（如开始处理、完成处理、找到 N 条记忆）
- `WARNING`：非预期但可恢复的情况（如重试、降级）
- `ERROR`：错误信息

调用方通过标准方式配置日志级别和输出格式。

## 12. 本地文件实现（LocalFileMemoryManager）

基于本地 JSON 文件的 `MemoryDatabaseManager` 实现，适用于小批量数据场景。

### 12.1 存储结构

按用户 ID 隔离，每个用户一个目录：

```
{MEMORY_DATA_DIR}/
├── user_001/
│   ├── memories.json      # Memory 列表
│   ├── embeddings.json    # {memory_id: [float, ...]}
│   └── synonyms.json      # 同义词表
├── user_002/
│   └── ...
```

### 12.2 环境变量

```bash
# 数据存储根目录
MEMORY_DATA_DIR=./data

# Embedding 模型（通过 LiteLLM 调用 Voyage）
EMBEDDING_MODEL=voyage/voyage-4
VOYAGE_API_KEY=your-api-key
```

### 12.3 需要实现的方法

**Protocol 方法：**
- `search_by_vector(user_id, query, top_k)` - 余弦相似度计算
- `search_by_keywords(user_id, keywords, top_k)` - 遍历匹配，分数基于命中关键词数量
- `get_all_keywords(user_id)` - 从所有 Memory 中收集
- `get_synonyms(user_id)` - 读取 synonyms.json
- `hybrid_search(user_id, query, top_k)` - 向量 + 关键词检索，RRF 融合（k 从 RRF_K 环境变量获取）

**CRUD 方法：**
- `add_memory(user_id, memory)` - 添加记忆，同时生成 embedding
- `update_memory(user_id, memory_id, update_dict)` - 更新记忆，若 content 变化则重新生成 embedding
- `delete_memory(user_id, memory_id)` - 删除记忆及其 embedding
- `get_memory(user_id, memory_id)` - 获取单条记忆
- `save_synonyms(user_id, synonyms)` - 保存同义词表

### 12.4 实现要点

- 根据 `user_id` 自动拼接用户目录路径，不存在则创建
- `add_memory` 时若 id 为空，使用 UUID 生成
- 向量检索：加载用户的所有 embedding 到内存，计算余弦相似度后排序
- 关键词检索：遍历用户的所有 Memory，同义词扩展后匹配，分数 = 命中关键词数 / 查询关键词数
- 文件读写：每次操作后立即持久化，保证数据一致性

## 13. 变更记录

### v5 (当前版本)

**数据结构变更：**
1. `MemoryReference.spans` → `MemoryReference.lines`
   - 从字符位置 `[[start, end], ...]` 改为行号列表 `[4, 5, 6]`
   - 输入文本预处理时添加行号前缀 `[行号]: 内容`
2. `Memory.entities` → `Memory.keywords`
   - 范围扩展：不仅限于人物、地点等命名实体，还包括活动、物品、情感、概念等有意义的关键词
3. 删除 `Memory.category` 字段
   - 该字段实际用途有限，予以移除

**Prompt 优化（针对 badcase 分析）：**
- 图片消息：增加对 `<image_message>` 内容的抽取指导
- 关系记忆：对 "we/us/our/我们" 表述明确标注涉及人物
- 相对时间：将 "last week/yesterday" 等转换为绝对日期
- 情感描述：保留原始情感词汇，避免过度概括
- 活动细节：完整保留具体内容，不省略细节

**向后兼容：**
- `Memory.from_dict()` 支持旧格式数据：
  - `entities` 自动映射到 `keywords`
  - `spans` 自动转换为 `lines`（取每个 span 的首元素）
  - `category` 字段被忽略
