"""
Example usage of the ZDMem memory system.

This script demonstrates:
1. Setting up the LocalFileMemoryManager
2. Processing text to create memories
3. Searching memories
4. Updating and deleting memories
"""

import logging
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

from memory_core import (
    Memory,
    extract_entities,
    generate_search_strings,
    generate_synonyms,
    process_memories,
)
from memory_core.local_manager import LocalFileMemoryManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    # Setup
    data_dir = Path("./example_data")
    manager = LocalFileMemoryManager(data_dir=str(data_dir))
    user_id = "example-user"

    print("=" * 60)
    print("ZDMem Memory System - Example Usage")
    print("=" * 60)

    # Sample text from design document
    sample_text = """
    2024年12月20日，我和老婆小美去深圳南山区的海底捞吃火锅，她特别喜欢番茄锅底。
    第二天我们去了华强北买了一台 iPhone 16 Pro，花了 8999 元。
    小美说她下周要去上海出差，大概待一周左右。
    对了，我最近在学 Python 的 FastAPI 框架，感觉比 Flask 好用多了。
    """

    # 1. Entity Extraction
    print("\n1. Entity Extraction")
    print("-" * 40)
    entities = extract_entities(sample_text)
    print(f"Extracted entities: {entities}")

    # 2. Search String Generation
    print("\n2. Search String Generation")
    print("-" * 40)
    search_strings = generate_search_strings(sample_text)
    print(f"Generated {len(search_strings)} search strings:")
    for i, ss in enumerate(search_strings, 1):
        print(f"  {i}. {ss}")

    # 3. Process Memories
    print("\n3. Process Memories (Main Entry Point)")
    print("-" * 40)
    result = process_memories(
        text=sample_text,
        text_id="example-001",
        user_id=user_id,
        manager=manager,
    )
    print(f"Created: {len(result.created)} memories")
    print(f"Updated: {len(result.updated)} memories")
    print(f"Deleted: {len(result.deleted)} memories")

    # Save created memories
    for op in result.created:
        if op.created_memory:
            saved = manager.add_memory(user_id, op.created_memory)
            print(f"  - Saved: {saved.content[:50]}...")

    # 4. Search Memories
    print("\n4. Search Memories")
    print("-" * 40)

    # Vector search
    print("\nVector Search: '小美喜欢吃什么'")
    results = manager.search_by_vector(user_id, "小美喜欢吃什么", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.memory.content[:50]}...")

    # Entity search
    print("\nEntity Search: ['小美']")
    results = manager.search_by_entities(user_id, ["小美"], top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.memory.content[:50]}...")

    # Hybrid search
    print("\nHybrid Search: 'Python 编程'")
    results = manager.hybrid_search(user_id, "Python 编程", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.memory.content[:50]}...")

    # 5. Synonym Generation
    print("\n5. Synonym Generation")
    print("-" * 40)
    all_entities = manager.get_all_entities(user_id)
    if all_entities:
        # Generate synonyms for first 5 entities
        sample_entities = all_entities[:5]
        synonyms = generate_synonyms(sample_entities)
        print(f"Generated synonyms for {len(sample_entities)} entities:")
        for entity, syns in synonyms.items():
            print(f"  {entity}: {syns}")

        # Save synonyms
        manager.save_synonyms(user_id, synonyms)
        print("Synonyms saved.")

    # 6. Memory Update Example
    print("\n6. Memory Update")
    print("-" * 40)
    memories = manager.get_all_memories(user_id)
    if memories:
        memory = memories[0]
        print(f"Original: {memory.content}")
        updated = manager.update_memory(
            user_id,
            memory.id,
            {"content": memory.content + " (已更新)"},
        )
        if updated:
            print(f"Updated:  {updated.content}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print(f"Data stored in: {data_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
