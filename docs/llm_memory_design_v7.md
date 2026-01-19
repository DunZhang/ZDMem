# 记忆系统设计 v7

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
- 默认模型：`deepseek/deepseek-chat`

## 2. 模块划分

| 模块 | 职责 | 实现方 |
|------|------|--------|
| 检索字符串生成 | 文本 → 多个检索字符串 | 核心模块 |
| 记忆抽取 | 文本 + 已有记忆 → 新记忆列表 | 核心模块 |
| 关键词抽取 | 文本 → 关键词列表 | 核心模块 |
| 同义词生成 | 关键词列表 → 同义词映射表 | 核心模块 |
| 记忆存储/检索 | 增删改查、向量索引、倒排索引、同义词管理 | 调用方 |

## 3. 数据结构

```python
from dataclasses import dataclass, field
import json

@dataclass
class MemoryReference:
    """记忆来源引用"""
    text_id: str
    spans: list[tuple[int, int]]  # [(start, end), ...] 基于原始文本的字符级别位置
    contents: list[str]           # 每个 span 对应的文本内容，通过 span 从原始文本截取获得

@dataclass
class Memory:
    """记忆实体"""
    user_id: str
    content: str
    id: str = ""                          # 新记忆为空，由调用方赋值
    keywords: list[str] = field(default_factory=list)  # 服务于检索的关键词，来源和形式多样
    occurred_string: str | None = None    # 可残缺的时间字符串，如 "2024-12-25"、"2024-12"、"2024-12-25T03"
    occurred_at: str | None = None        # ISO 8601 格式，如 "2024-12-25T06:30:45.123456+00:00"
    references: list[MemoryReference] = field(default_factory=list)
    
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
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 记忆唯一标识，新记忆为空，由调用方赋值 |
| user_id | str | 用户标识 |
| content | str | 记忆内容，自然语言文本 |
| keywords | list[str] | 服务于检索的关键词列表。来源多样：既有直接来自原文的词，也有总结/推断获得的词；形式多样：既有具体实体（人名、地名、产品名），也有抽象概念。一切以提升检索召回为目标 |
| occurred_string | str \| None | 可残缺的时间字符串，由 LLM 根据实际情况填写，如 "2024-12-25"、"2024-12"、"2024-12-25T03"，这是最准确的时间表示 |
| occurred_at | str \| None | 完整 ISO 8601 时间字符串，由核心模块根据 occurred_string 自动补全 |
| references | list[MemoryReference] | 记忆来源，记录从哪段文本的哪个位置抽取，包含对应的文本内容 |

**时间格式说明：**

`occurred_string` 字段允许残缺格式，常见形式：
- `"2024"` - 只知道年份
- `"2024-12"` - 精确到月
- `"2024-12-25"` - 精确到日
- `"2024-12-25T03"` - 精确到小时
- `"2024-12-25T03:30"` - 精确到分钟
- `None` - 无法从文本中提取时间

`occurred_at` 使用完整 ISO 8601 格式：`2024-12-25T06:30:45.123456+00:00`

## 4. 接口定义

调用方需实现 `MemoryDatabaseManager` 协议：

```python
from typing import Protocol

class MemoryDatabaseManager(Protocol):
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
            score_threshold: 相似度阈值，低于此值的结果不返回，None 则从 VECTOR_SCORE_THRESHOLD 环境变量获取，默认 0.5
            
        Returns:
            按相似度降序排列的 ScoredMemory 列表，只包含分数 >= score_threshold 的结果
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
            分数可基于命中关键词数量、关键词权重等计算
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
            格式：{canonical_keyword: [synonym1, synonym2, ...]}
            示例：{"老婆": ["妻子", "太太", "媳妇"]}
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
        
        用于聊天时检索相关记忆。
        
        Args:
            user_id: 用户标识
            query: 查询文本
            top_k: 返回数量
            score_threshold: 向量检索的相似度阈值，低于此值的结果不参与融合，默认 0.5
            
        Returns:
            RRF 融合后的 ScoredMemory 列表，按分数降序排列
            
        实现要点：
            1. 调用 search_by_vector 获取向量检索结果（应用 score_threshold 过滤）
            2. 从 query 抽取关键词，调用 search_by_keywords 获取关键词检索结果
            3. 使用 RRF 公式融合：score = 1/(k+rank1) + 1/(k+rank2)，k 默认 60
        """
        ...
```

## 5. 算法流程

### 5.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        process_memories                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: 生成检索字符串                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 输入: 用户文本                                                           │ │
│ │ 输出: search_strings = [s1, s2, ..., sn, 原始文本]                       │ │
│ │ 说明: LLM 生成多个检索字符串（必须是完整句子），最后将原始文本也加入作为兜底 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: 混合检索【可并行】                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ for each search_string (并行):                                          │ │
│ │   ├── 向量检索: search_by_vector(search_string, score_threshold)       │ │
│ │   ├── 关键词抽取: extract_keywords(search_string) → keywords[]         │ │
│ │   └── 关键词检索: search_by_keywords(keywords) → ScoredMemory[]        │ │
│ │                                                                         │ │
│ │ 合并去重，按 occurred_at 升序排序 → related_memories                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: 抽取新记忆                                                           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 构建 prompt：                                                           │ │
│ │   - 带行号的用户文本                                                     │ │
│ │   - 已有记忆列表（用于去重）                                              │ │
│ │ LLM 输出：                                                              │ │
│ │   新记忆列表，每条包含 content, keywords, occurred_string, line_numbers │ │
│ │ 后处理：                                                                │ │
│ │   - 根据 line_numbers 计算 spans                                        │ │
│ │   - 构建 Memory 对象                                                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: 返回结果                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ return list[Memory]  # 新抽取的记忆列表                                  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 文本分句与行号处理

为了方便 LLM 输出 reference，需要对输入文本进行分句并添加行号。

**基本思路：**

1. **分句**：将文本按换行符分割成多行（后续可扩展为更复杂的分句逻辑）

2. **添加行号**：为每行添加 `[idx]: ` 前缀，同时记录每行在原始文本中的字符位置
   - 输入：`"第一行\n第二行\n第三行"`
   - 输出：`"[0]: 第一行\n[1]: 第二行\n[2]: 第三行"` + 位置映射 `[(0, 3), (4, 7), (8, 11)]`

3. **行号转 spans**：将 LLM 输出的行号列表转换为原始文本的字符位置，相邻行合并为连续 span

4. **获取 contents**：根据 spans 从原始文本截取对应的文本内容，生成 contents 列表

**需要实现的函数：**

| 函数 | 输入 | 输出 |
|------|------|------|
| `split_into_lines(text)` | 原始文本 | 行列表 |
| `add_line_numbers(text)` | 原始文本 | (带行号文本, 行位置映射) |
| `line_numbers_to_spans(line_numbers, line_positions)` | 行号列表, 位置映射 | spans 列表 |
| `get_contents_from_spans(spans, text)` | spans 列表, 原始文本 | contents 列表 |

### 5.3 关键设计说明

#### 5.3.1 为什么要生成检索字符串？

直接用用户原文做向量检索效果差，原因：
1. 用户文本可能很长，包含多个话题
2. 向量模型对长文本的表示能力有限
3. 不同话题混在一起会稀释语义

解决方案：让 LLM 分析用户文本，生成多个聚焦的检索字符串，每个覆盖一个话题或关键信息点。

#### 5.3.2 为什么要把原始文本也作为检索字符串？

作为兜底策略，防止 LLM 生成的检索字符串遗漏重要信息。

#### 5.3.3 并行化设计

**可并行的操作：**
1. **Step 2**：对多个 search_string 的检索和关键词抽取可以并行
2. **同义词生成**：`generate_synonyms` 中各批次的 LLM 调用可以并行

## 6. 核心模块 API

### 6.1 LLM 调用规范

所有调用大模型的地方遵循以下规范：

- **重试机制**：LLM 调用失败时自动重试，可配置重试次数
- **结构化输出**：使用 LiteLLM 的 `response_format={"type": "json_object"}` 确保输出 JSON
- **输出解析**：处理 \`\`\`json 或 \`\`\` 包裹的情况，使用 `json_repair` 库修复常见 JSON 格式问题
- **语言一致性**：LLM 输出语言必须与用户文本语言保持一致（在 prompt 中明确指定）

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
    score_threshold: float | None = None,
    search_string_gen_model: str | None = None,
    memory_extract_model: str | None = None,
    keyword_extract_model: str | None = None,
) -> list[Memory]:
    """
    从文本抽取新记忆
    
    Args:
        text: 输入文本（对话记录、用户笔记等）
        text_id: 文本唯一标识，用于记录记忆来源
        user_id: 用户标识
        manager: 记忆数据库管理器
        search_top_k: 每个检索字符串的返回数量，默认 10
        score_threshold: 向量检索的相似度阈值，默认 0.5
        search_string_gen_model: 检索字符串生成模型
        memory_extract_model: 记忆抽取模型
        keyword_extract_model: 关键词抽取模型
        
    Returns:
        新抽取的 Memory 列表（id 为空，由调用方赋值后存储）
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
    - 检索字符串必须是完整的句子或问题，不能过短（如单个词或短语）
    - 好的例子："小美喜欢吃番茄锅底"、"在华强北购买 iPhone"
    - 坏的例子："小美"、"番茄"、"iPhone"
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
def extract_keywords(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    从文本抽取关键词，**仅用于检索场景**
    
    注意：此函数不用于记忆抽取流程。记忆的 keywords 字段由 extract_memories 时 LLM 一并生成。
    
    典型使用场景：
    - 检索时从用户问题中抽取关键词
    - 从检索字符串中抽取关键词用于关键词检索
    
    关键词类型包括：
    - 人名、地名、组织名、产品名等传统实体
    - 具体的地点描述（如"深圳南山区"）
    - 品牌、店铺名称（如"海底捞"）
    - 技术术语、概念
    - 其他有意义的名词短语
    
    Args:
        text: 输入文本（用户问题、检索字符串等）
        model: 使用的 LLM 模型，None 则从 KEYWORD_EXTRACT_MODEL 或 DEFAULT_MODEL 获取
        
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
    批量生成同义词【各批次可并行】
    
    注意：本函数只负责生成同义词，不负责存储。调用方需要：
    1. 自行决定何时调用（如定期触发、新关键词达到一定数量时）
    2. 调用 manager.save_synonyms 存储结果
    3. 在 search_by_keywords 中使用同义词表进行扩展
    
    处理流程：
    1. 将 keywords 分批，每批 batch_size 个
    2. 并行调用 LLM 为各批次生成同义词
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

使用 Jinja2 模板，存放于 `prompts/` 目录。

**重要：所有 Prompt 都必须包含语言一致性要求，确保 LLM 输出语言与用户文本语言一致。**

### 7.1 检索字符串生成 Prompt

文件：`prompts/generate_search_strings.j2`

**输入变量：**
- `text` - 用户输入文本

**输出格式：**
```json
["检索字符串1", "检索字符串2", ...]
```

### 7.2 记忆抽取 Prompt

文件：`prompts/extract_memories.j2`

**输入变量：**
- `text` - 带行号的用户输入文本
- `existing_memories` - 已有记忆列表（用于去重）

**输出格式：**
```json
[
  {
    "content": "记忆内容",
    "keywords": ["关键词1", "关键词2"],
    "occurred_string": "2024-12-25",
    "line_numbers": [0, 1]
  }
]
```

**keywords 定义：** 服务于检索召回，来源和形式多样。来源：既可直接来自原文，也可通过总结/推断获得；形式：既有具体实体（人名、地名、组织、产品、品牌、店铺），也有抽象概念和技术术语。一切以提升检索召回为目标，宁多勿少。

### 7.3 关键词抽取 Prompt

文件：`prompts/extract_keywords.j2`

**输入变量：**
- `text` - 输入文本

**输出格式：**
```json
["关键词1", "关键词2", ...]
```

**keywords 定义：** 与 7.2 记忆抽取中的定义保持一致。

### 7.4 同义词生成 Prompt

文件：`prompts/generate_synonyms.j2`

**输入变量：**
- `keywords` - 关键词列表（JSON 数组）

**输出格式：**
```json
{
  "关键词1": ["同义词1", "同义词2"],
  "关键词2": []
}
```

## 8. 环境变量

文件：`.env`

```bash
# 默认模型（兜底）
DEFAULT_MODEL=deepseek/deepseek-chat

# 各模块专用模型（可选，未设置则使用 DEFAULT_MODEL）
SEARCH_STRING_GEN_MODEL=
KEYWORD_EXTRACT_MODEL=
MEMORY_EXTRACT_MODEL=
SYNONYM_GEN_MODEL=

# Embedding 模型
EMBEDDING_MODEL=voyage/voyage-4
VOYAGE_API_KEY=your-voyage-api-key

# 数据存储目录
MEMORY_DATA_DIR=./data

# 参数默认值
SEARCH_TOP_K=10
SYNONYM_BATCH_SIZE=20
SYNONYM_MAX_RETRIES=2
RRF_K=60
VECTOR_SCORE_THRESHOLD=0.5

# API Keys
OPENROUTER_API_KEY=your-api-key
```

## 9. 项目结构

```
memory-system/
├── pyproject.toml
├── .env
├── prompts/
│   ├── generate_search_strings.j2
│   ├── extract_memories.j2
│   ├── extract_keywords.j2
│   └── generate_synonyms.j2
├── memory_core/
│   ├── __init__.py
│   ├── models.py            # 数据结构
│   ├── protocol.py          # Manager 接口
│   ├── config.py            # 环境变量和模型选择逻辑
│   ├── main.py              # 主入口 process_memories
│   ├── search.py            # 检索字符串生成
│   ├── extract.py           # 记忆抽取
│   ├── keyword.py           # 关键词抽取
│   ├── synonym.py           # 同义词生成
│   ├── text_utils.py        # 文本处理工具
│   ├── llm.py               # LiteLLM 封装
│   └── local_manager.py     # 本地文件实现
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
- 核心模块：`generate_search_strings`、`extract_keywords`、`generate_synonyms`、`process_memories`
- 文本工具：`split_into_lines`、`add_line_numbers`、`line_numbers_to_spans`、`get_contents_from_spans`
- LocalFileMemoryManager：所有 Protocol 方法和 CRUD 方法

**完成标准：**

每个测试用例都必须通过，所有测试成功后才算最终完成。LLM 相关测试使用真实调用，不使用 mock。

## 11. 日志规范

使用 Python 标准 `logging` 模块，每个模块使用独立 logger。

**日志级别使用建议：**
- `DEBUG`：详细调试信息（如 LLM 输入输出、检索结果）
- `INFO`：关键流程节点（如开始处理、完成处理、找到 N 条记忆）
- `WARNING`：非预期但可恢复的情况（如重试、降级）
- `ERROR`：错误信息

**LLM 调用日志规范：**

所有 LLM 调用必须在**返回结果后**一起输出 prompt 和 response，并明确标注是哪个函数的调用。

**日志格式要求：**

| 函数 | Prompt 日志 | Response 日志 |
|------|-------------|---------------|
| generate_search_strings | `"Generate search strings prompt:\n%s"` | `"Generate search strings response:\n%s"` |
| extract_memories | `"Extract memories prompt:\n%s"` | `"Extract memories response:\n%s"` |
| extract_keywords | `"Extract keywords prompt:\n%s"` | `"Extract keywords response:\n%s"` |
| generate_synonyms | `"Generate synonyms prompt:\n%s"` | `"Generate synonyms response:\n%s"` |

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
VOYAGE_API_KEY=your-voyage-api-key
```

### 12.3 需要实现的方法

**Protocol 方法：**
- `search_by_vector(user_id, query, top_k, score_threshold)` - 余弦相似度计算，过滤低于阈值的结果
- `search_by_keywords(user_id, keywords, top_k)` - 遍历匹配，分数基于命中关键词数量
- `get_all_keywords(user_id)` - 从所有 Memory 中收集
- `get_synonyms(user_id)` - 读取 synonyms.json
- `hybrid_search(user_id, query, top_k, score_threshold)` - 向量 + 关键词检索，RRF 融合（k 从 RRF_K 环境变量获取）

**CRUD 方法：**
- `add_memory(user_id, memory)` - 添加记忆，同时生成 embedding
- `get_memory(user_id, memory_id)` - 获取单条记忆
- `save_synonyms(user_id, synonyms)` - 保存同义词表

### 12.4 实现要点

- 根据 `user_id` 自动拼接用户目录路径，不存在则创建
- `add_memory` 时若 id 为空，使用 UUID 生成
- 向量检索：计算余弦相似度，**过滤掉分数 < score_threshold 的结果**
- 关键词检索：同义词扩展后匹配，分数 = 命中关键词数 / 查询关键词数
- 文件读写：每次操作后立即持久化
