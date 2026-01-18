"""
Step 4 (简化版): 分析 Badcase

使用简单的文本匹配进行分类，不使用语义相似度。
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 问题类型
ISSUE_TYPE_EXTRACTION = "EXTRACTION_ISSUE"
ISSUE_TYPE_RETRIEVAL = "RETRIEVAL_ISSUE"
ISSUE_TYPE_OTHER = "OTHER_ISSUE"


def parse_evidence_id(evidence_id: str) -> tuple[int, str]:
    """解析 evidence ID"""
    match = re.match(r"D(\d+):(\d+)", evidence_id)
    if not match:
        return -1, evidence_id
    session_num = int(match.group(1))
    return session_num, evidence_id


def get_dialogue_by_evidence(evidence_id: str, user_pair_dir: Path) -> str:
    """根据 evidence ID 获取对话内容"""
    session_num, dialogue_id = parse_evidence_id(evidence_id)
    if session_num < 0:
        return ""

    session_file = user_pair_dir / f"session_{session_num}.txt"
    if not session_file.exists():
        return ""

    content = session_file.read_text(encoding="utf-8")
    pattern = rf"dialogue_id:\s*{re.escape(dialogue_id)},\s*(\w+):\s*(.+?)(?=\n\ndialogue_id:|\n*$)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        speaker = match.group(1)
        text = match.group(2).strip()
        return f"{speaker}: {text}"
    return ""


def simple_text_match(evidence_text: str, memories: list[dict]) -> dict:
    """
    使用简单文本匹配检查 evidence 是否在记忆中有覆盖

    匹配策略：
    1. 关键词匹配：提取 evidence 中的名词/实体
    2. 检查是否有记忆包含这些关键词
    """
    # 提取 evidence 中的关键词（简单实现）
    evidence_lower = evidence_text.lower()

    # 找最相似的记忆
    best_match = None
    best_score = 0

    for mem in memories:
        mem_content = mem.get("content", "").lower()

        # 简单的词重叠计算
        evidence_words = set(evidence_lower.split())
        mem_words = set(mem_content.split())

        overlap = len(evidence_words & mem_words)
        score = overlap / max(len(evidence_words), 1)

        if score > best_score:
            best_score = score
            best_match = mem

    return {
        "is_covered": best_score >= 0.3,  # 30% 词重叠视为覆盖
        "best_match": best_match,
        "match_score": best_score,
    }


def classify_badcase(
    result: dict,
    user_pair_dir: Path,
    memories: list[dict],
) -> tuple[str, dict]:
    """分类 badcase"""
    evidence_list = result.get("evidence", [])
    retrieved_memories = result.get("retrieved_memories", [])
    retrieved_ids = {m["id"] for m in retrieved_memories}

    # 获取 evidence 内容
    evidence_contents = []
    for ev in evidence_list:
        content = get_dialogue_by_evidence(ev, user_pair_dir)
        if content:
            evidence_contents.append({
                "evidence_id": ev,
                "content": content,
            })

    if not evidence_contents:
        return ISSUE_TYPE_OTHER, {
            "reason": "无法获取 evidence 内容",
        }

    # 检查是否被抽取
    missing_evidences = []
    matched_memories = []

    for ev in evidence_contents:
        match_result = simple_text_match(ev["content"], memories)
        if match_result["is_covered"]:
            matched_memories.append({
                "evidence": ev,
                "memory": match_result["best_match"],
                "score": match_result["match_score"],
            })
        else:
            missing_evidences.append({
                "evidence": ev,
                "best_score": match_result["match_score"],
            })

    if missing_evidences:
        return ISSUE_TYPE_EXTRACTION, {
            "missing_evidences": missing_evidences,
            "matched_count": len(matched_memories),
            "reason": "Evidence 对应的对话内容未被完全抽取为记忆",
        }

    # 检查是否被检索到
    missed_in_retrieval = []
    for m in matched_memories:
        if m["memory"] and m["memory"]["id"] not in retrieved_ids:
            missed_in_retrieval.append(m)

    if missed_in_retrieval:
        return ISSUE_TYPE_RETRIEVAL, {
            "missed_memories": [m["memory"]["content"][:100] for m in missed_in_retrieval if m["memory"]],
            "reason": "相关记忆存在但未被检索到",
        }

    return ISSUE_TYPE_OTHER, {
        "reason": "记忆已检索到但 LLM 回答错误",
    }


def analyze_badcases_for_user_pair(
    user_pair: str,
    data_dir: Path,
    output_dir: Path,
) -> dict:
    """分析一对对话人的 badcase"""
    user_pair_dir = data_dir / user_pair
    user_output_dir = output_dir / user_pair

    # 加载 QA 评估结果
    qa_results_file = user_output_dir / "qa_results.json"
    with open(qa_results_file, encoding="utf-8") as f:
        qa_data = json.load(f)

    # 加载抽取的记忆
    memories_file = user_output_dir / "extracted_memories.json"
    with open(memories_file, encoding="utf-8") as f:
        memories = json.load(f)

    # 筛选 badcase
    badcases = [r for r in qa_data["results"] if not r["is_correct"]]
    logger.info(f"{user_pair}: 共 {len(badcases)} 个 badcase")

    # 分类分析
    extraction_issues = []
    retrieval_issues = []
    other_issues = []

    for i, bc in enumerate(badcases):
        if i % 10 == 0:
            logger.info(f"  处理 {i+1}/{len(badcases)}...")

        issue_type, details = classify_badcase(bc, user_pair_dir, memories)

        badcase_info = {
            "id": bc["id"],
            "question": bc["question"],
            "ground_truth": bc["ground_truth"],
            "prediction": bc["prediction"],
            "evidence": bc["evidence"],
            "category": bc["category"],
            "judge_score": bc["judge_score"],
            "issue_details": details,
        }

        if issue_type == ISSUE_TYPE_EXTRACTION:
            extraction_issues.append(badcase_info)
        elif issue_type == ISSUE_TYPE_RETRIEVAL:
            retrieval_issues.append(badcase_info)
        else:
            other_issues.append(badcase_info)

    logger.info(f"  抽取问题: {len(extraction_issues)}, "
               f"检索问题: {len(retrieval_issues)}, "
               f"其他问题: {len(other_issues)}")

    return {
        "user_pair": user_pair,
        "total_badcases": len(badcases),
        "extraction_issues": extraction_issues,
        "retrieval_issues": retrieval_issues,
        "other_issues": other_issues,
    }


def generate_report(analysis: dict, output_dir: Path):
    """生成分析报告"""
    badcase_dir = output_dir / "badcase_analysis"
    badcase_dir.mkdir(exist_ok=True)

    # 保存分类结果
    with open(badcase_dir / "extraction_issues.json", "w", encoding="utf-8") as f:
        json.dump(analysis["extraction_issues"], f, ensure_ascii=False, indent=2)

    with open(badcase_dir / "retrieval_issues.json", "w", encoding="utf-8") as f:
        json.dump(analysis["retrieval_issues"], f, ensure_ascii=False, indent=2)

    with open(badcase_dir / "other_issues.json", "w", encoding="utf-8") as f:
        json.dump(analysis["other_issues"], f, ensure_ascii=False, indent=2)

    # 统计
    total = analysis["total_badcases"]
    ext_count = len(analysis["extraction_issues"])
    ret_count = len(analysis["retrieval_issues"])
    other_count = len(analysis["other_issues"])

    # 按 category 统计
    category_stats = defaultdict(lambda: {"extraction": 0, "retrieval": 0, "other": 0})
    for issue in analysis["extraction_issues"]:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["extraction"] += 1
    for issue in analysis["retrieval_issues"]:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["retrieval"] += 1
    for issue in analysis["other_issues"]:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["other"] += 1

    # 生成 Markdown 报告
    report = f"""# Locomo 评估 Badcase 分析报告

## 概览

| 指标 | 数量 | 占比 |
|------|------|------|
| 总 Badcase 数 | {total} | 100% |
| 记忆抽取问题 | {ext_count} | {ext_count/total*100:.1f}% |
| 记忆检索问题 | {ret_count} | {ret_count/total*100:.1f}% |
| 其他问题 | {other_count} | {other_count/total*100:.1f}% |

## 按问题类别统计

| 类别 | 抽取问题 | 检索问题 | 其他问题 |
|------|----------|----------|----------|
"""
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        report += f"| {cat} | {stats['extraction']} | {stats['retrieval']} | {stats['other']} |\n"

    # 典型案例
    report += """
## 记忆抽取问题典型案例

"""
    for i, issue in enumerate(analysis["extraction_issues"][:5], 1):
        report += f"""### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **Evidence**: {issue['evidence']}
- **分析**: {issue['issue_details'].get('reason', '')}

"""

    report += """
## 记忆检索问题典型案例

"""
    for i, issue in enumerate(analysis["retrieval_issues"][:5], 1):
        missed = issue['issue_details'].get('missed_memories', [])
        report += f"""### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **未检索到的记忆**: {missed[:2] if missed else '无'}

"""

    report += """
## 其他问题典型案例

"""
    for i, issue in enumerate(analysis["other_issues"][:5], 1):
        report += f"""### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **Judge 评分**: {issue['judge_score']}

"""

    report += """
## 优化建议（待深入分析后补充）

### 针对记忆抽取问题
- 检查抽取 prompt 是否遗漏了关键信息类型

### 针对记忆检索问题
- 检查实体抽取和同义词覆盖

### 针对其他问题
- 检查 QA 回答 prompt 是否需要优化
"""

    with open(badcase_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"报告已保存至: {badcase_dir / 'analysis_report.md'}")


def main(user_pairs: list[str] | None = None):
    script_dir = Path(__file__).parent
    data_dir = script_dir / "processed_locomo_test_data"
    output_dir = script_dir / "evaluation_results"

    # 获取所有已评估的对话人
    all_user_pairs = sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "qa_results.json").exists()
    ])

    if user_pairs:
        all_user_pairs = [u for u in all_user_pairs if u in user_pairs]

    if not all_user_pairs:
        logger.error("未找到已评估的对话人")
        return

    logger.info(f"待分析的对话人数量: {len(all_user_pairs)}")

    all_analyses = {
        "total_badcases": 0,
        "extraction_issues": [],
        "retrieval_issues": [],
        "other_issues": [],
    }

    for user_pair in all_user_pairs:
        analysis = analyze_badcases_for_user_pair(user_pair, data_dir, output_dir)
        all_analyses["total_badcases"] += analysis["total_badcases"]
        all_analyses["extraction_issues"].extend(analysis["extraction_issues"])
        all_analyses["retrieval_issues"].extend(analysis["retrieval_issues"])
        all_analyses["other_issues"].extend(analysis["other_issues"])

    generate_report(all_analyses, output_dir)
    logger.info("\n=== 分析完成 ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-pairs", nargs="+")
    args = parser.parse_args()
    main(user_pairs=args.user_pairs)
