"""
Step 4: 分析 Badcase

利用 evidence 字段判断问题来源:
- 记忆抽取问题: evidence 对应对话未被抽取为记忆
- 记忆检索问题: 记忆存在但未被检索到
- 其他问题: 记忆已检索但回答错误

保存分析结果到 badcase_analysis/ 目录
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import litellm
litellm.suppress_debug_info = True

from memory_core.llm import get_embedding

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 问题类型
ISSUE_TYPE_EXTRACTION = "EXTRACTION_ISSUE"
ISSUE_TYPE_RETRIEVAL = "RETRIEVAL_ISSUE"
ISSUE_TYPE_OTHER = "OTHER_ISSUE"


def parse_evidence_id(evidence_id: str) -> tuple[int, str]:
    """
    解析 evidence ID

    Args:
        evidence_id: 如 "D10:18"

    Returns:
        (session_num, full_dialogue_id)
    """
    match = re.match(r"D(\d+):(\d+)", evidence_id)
    if not match:
        return -1, evidence_id
    session_num = int(match.group(1))
    return session_num, evidence_id


def get_dialogue_by_evidence(
    evidence_id: str,
    user_pair_dir: Path,
) -> str:
    """
    根据 evidence ID 获取对话内容

    Args:
        evidence_id: 如 "D10:18"
        user_pair_dir: 对话人目录

    Returns:
        对话内容
    """
    session_num, dialogue_id = parse_evidence_id(evidence_id)
    if session_num < 0:
        return ""

    session_file = user_pair_dir / f"session_{session_num}.txt"
    if not session_file.exists():
        return ""

    content = session_file.read_text(encoding="utf-8")

    # 查找对应 dialogue_id 的内容
    # 格式: "dialogue_id: D10:18,  Melanie: ..."
    pattern = rf"dialogue_id:\s*{re.escape(dialogue_id)},\s*(\w+):\s*(.+?)(?=\n\ndialogue_id:|\n*$)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        speaker = match.group(1)
        text = match.group(2).strip()
        return f"{speaker}: {text}"
    return ""


def compute_similarity(text1: str, text2: str) -> float:
    """计算两段文本的语义相似度"""
    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))
    except Exception as e:
        logger.warning(f"计算相似度时出错: {e}")
        return 0.0


def check_extraction_coverage(
    evidence_contents: list[dict],
    memories: list[dict],
    similarity_threshold: float = 0.65,
) -> dict:
    """
    检查 evidence 是否被抽取为记忆

    Args:
        evidence_contents: [{"evidence_id": "D1:2", "content": "..."}]
        memories: 抽取的记忆列表
        similarity_threshold: 相似度阈值

    Returns:
        覆盖情况
    """
    if not memories:
        return {
            "is_covered": False,
            "matched": [],
            "missing": evidence_contents,
        }

    # 预计算记忆的 embedding
    memory_contents = [m["content"] for m in memories]

    matched = []
    missing = []

    for ev in evidence_contents:
        ev_content = ev["content"]
        if not ev_content:
            continue

        # 计算与所有记忆的相似度
        best_sim = 0.0
        best_match_idx = -1

        for idx, mem_content in enumerate(memory_contents):
            sim = compute_similarity(ev_content, mem_content)
            if sim > best_sim:
                best_sim = sim
                best_match_idx = idx

        if best_sim >= similarity_threshold:
            matched.append({
                "evidence": ev,
                "matched_memory": memories[best_match_idx],
                "similarity": best_sim,
            })
        else:
            missing.append({
                "evidence": ev,
                "best_similarity": best_sim,
                "best_match": memories[best_match_idx] if best_match_idx >= 0 else None,
            })

    return {
        "is_covered": len(missing) == 0,
        "matched": matched,
        "missing": missing,
    }


def check_retrieval_coverage(
    matched_memories: list[dict],
    retrieved_memory_ids: set[str],
) -> dict:
    """
    检查相关记忆是否被检索到

    Args:
        matched_memories: 与 evidence 匹配的记忆
        retrieved_memory_ids: 检索到的记忆 ID 集合

    Returns:
        覆盖情况
    """
    retrieved = []
    missed = []

    for m in matched_memories:
        memory = m["matched_memory"]
        if memory["id"] in retrieved_memory_ids:
            retrieved.append(m)
        else:
            missed.append(m)

    return {
        "is_covered": len(missed) == 0,
        "retrieved": retrieved,
        "missed": missed,
    }


def classify_badcase(
    result: dict,
    user_pair_dir: Path,
    memories: list[dict],
) -> tuple[str, dict]:
    """
    分类 badcase 的问题来源

    Args:
        result: QA 评估结果
        user_pair_dir: 原始数据目录
        memories: 抽取的所有记忆

    Returns:
        (问题类型, 详细信息)
    """
    evidence_list = result.get("evidence", [])
    retrieved_memories = result.get("retrieved_memories", [])
    retrieved_ids = {m["id"] for m in retrieved_memories}

    # Step 1: 解析 evidence 获取原始对话
    evidence_contents = []
    for ev in evidence_list:
        content = get_dialogue_by_evidence(ev, user_pair_dir)
        evidence_contents.append({
            "evidence_id": ev,
            "content": content,
        })

    # Step 2: 检查 evidence 是否被抽取为记忆
    extraction_result = check_extraction_coverage(evidence_contents, memories)

    if not extraction_result["is_covered"]:
        return ISSUE_TYPE_EXTRACTION, {
            "missing_evidences": extraction_result["missing"],
            "matched_evidences": extraction_result["matched"],
            "reason": "Evidence 对应的对话内容未被完全抽取为记忆",
        }

    # Step 3: 检查相关记忆是否被检索到
    retrieval_result = check_retrieval_coverage(
        extraction_result["matched"],
        retrieved_ids,
    )

    if not retrieval_result["is_covered"]:
        return ISSUE_TYPE_RETRIEVAL, {
            "missed_memories": retrieval_result["missed"],
            "retrieved_memories": retrieval_result["retrieved"],
            "reason": "相关记忆存在但未被检索到",
        }

    # Step 4: 记忆已检索但回答错误
    return ISSUE_TYPE_OTHER, {
        "retrieved_relevant": True,
        "matched_evidences": extraction_result["matched"],
        "reason": "记忆已检索到但 LLM 回答错误，可能是推理问题或问题本身较难",
    }


def analyze_badcases_for_user_pair(
    user_pair: str,
    data_dir: Path,
    output_dir: Path,
) -> dict:
    """
    分析一对对话人的 badcase

    Args:
        user_pair: 对话人 ID
        data_dir: 原始数据目录
        output_dir: 评估结果目录

    Returns:
        分析结果
    """
    user_pair_dir = data_dir / user_pair
    user_output_dir = output_dir / user_pair

    # 加载 QA 评估结果
    qa_results_file = user_output_dir / "qa_results.json"
    if not qa_results_file.exists():
        raise FileNotFoundError(f"未找到评估结果: {qa_results_file}")

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

    for bc in badcases:
        issue_type, details = classify_badcase(bc, user_pair_dir, memories)

        badcase_info = {
            "id": bc["id"],
            "question": bc["question"],
            "ground_truth": bc["ground_truth"],
            "prediction": bc["prediction"],
            "evidence": bc["evidence"],
            "category": bc["category"],
            "judge_score": bc["judge_score"],
            "judge_explanation": bc["judge_explanation"],
            "issue_details": details,
        }

        if issue_type == ISSUE_TYPE_EXTRACTION:
            extraction_issues.append(badcase_info)
        elif issue_type == ISSUE_TYPE_RETRIEVAL:
            retrieval_issues.append(badcase_info)
        else:
            other_issues.append(badcase_info)

    return {
        "user_pair": user_pair,
        "total_badcases": len(badcases),
        "extraction_issues": extraction_issues,
        "retrieval_issues": retrieval_issues,
        "other_issues": other_issues,
        "summary": {
            "extraction_count": len(extraction_issues),
            "retrieval_count": len(retrieval_issues),
            "other_count": len(other_issues),
        },
    }


def generate_analysis_report(
    all_analyses: list[dict],
    output_dir: Path,
):
    """
    生成分析报告

    Args:
        all_analyses: 所有对话人的分析结果
        output_dir: 输出目录
    """
    badcase_dir = output_dir / "badcase_analysis"
    badcase_dir.mkdir(exist_ok=True)

    # 汇总统计
    total_badcases = sum(a["total_badcases"] for a in all_analyses)
    total_extraction = sum(a["summary"]["extraction_count"] for a in all_analyses)
    total_retrieval = sum(a["summary"]["retrieval_count"] for a in all_analyses)
    total_other = sum(a["summary"]["other_count"] for a in all_analyses)

    # 收集所有问题
    all_extraction_issues = []
    all_retrieval_issues = []
    all_other_issues = []

    for a in all_analyses:
        for issue in a["extraction_issues"]:
            issue["user_pair"] = a["user_pair"]
            all_extraction_issues.append(issue)
        for issue in a["retrieval_issues"]:
            issue["user_pair"] = a["user_pair"]
            all_retrieval_issues.append(issue)
        for issue in a["other_issues"]:
            issue["user_pair"] = a["user_pair"]
            all_other_issues.append(issue)

    # 保存分类结果
    with open(badcase_dir / "extraction_issues.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(all_extraction_issues),
            "issues": all_extraction_issues,
        }, f, ensure_ascii=False, indent=2)

    with open(badcase_dir / "retrieval_issues.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(all_retrieval_issues),
            "issues": all_retrieval_issues,
        }, f, ensure_ascii=False, indent=2)

    with open(badcase_dir / "other_issues.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(all_other_issues),
            "issues": all_other_issues,
        }, f, ensure_ascii=False, indent=2)

    # 按 category 统计
    category_stats = defaultdict(lambda: {"extraction": 0, "retrieval": 0, "other": 0, "total": 0})
    for issue in all_extraction_issues:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["extraction"] += 1
        category_stats[cat]["total"] += 1
    for issue in all_retrieval_issues:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["retrieval"] += 1
        category_stats[cat]["total"] += 1
    for issue in all_other_issues:
        cat = issue.get("category", "Unknown").strip() or "Unknown"
        category_stats[cat]["other"] += 1
        category_stats[cat]["total"] += 1

    # 生成 Markdown 报告
    report = f"""# Locomo 评估 Badcase 分析报告

## 概览

| 指标 | 数量 |
|------|------|
| 总 Badcase 数 | {total_badcases} |
| 记忆抽取问题 | {total_extraction} ({total_extraction/total_badcases*100:.1f}%) |
| 记忆检索问题 | {total_retrieval} ({total_retrieval/total_badcases*100:.1f}%) |
| 其他问题 | {total_other} ({total_other/total_badcases*100:.1f}%) |

## 按问题类别统计

| 类别 | 抽取问题 | 检索问题 | 其他问题 | 总计 |
|------|----------|----------|----------|------|
"""
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        report += f"| {cat} | {stats['extraction']} | {stats['retrieval']} | {stats['other']} | {stats['total']} |\n"

    report += """
## 记忆抽取问题分析

记忆抽取问题是指 evidence 对应的对话内容未被抽取为记忆。

### 典型案例

"""
    for i, issue in enumerate(all_extraction_issues[:5], 1):
        report += f"""#### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **Evidence**: {issue['evidence']}
- **缺失信息**: {json.dumps(issue['issue_details'].get('missing_evidences', []), ensure_ascii=False, indent=2)[:500]}

"""

    report += """
## 记忆检索问题分析

记忆检索问题是指相关记忆已被抽取，但未被检索到。

### 典型案例

"""
    for i, issue in enumerate(all_retrieval_issues[:5], 1):
        missed = issue['issue_details'].get('missed_memories', [])
        missed_info = [{"content": m.get("matched_memory", {}).get("content", ""), "similarity": m.get("similarity", 0)} for m in missed]
        report += f"""#### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **未检索到的记忆**: {json.dumps(missed_info, ensure_ascii=False, indent=2)[:500]}

"""

    report += """
## 其他问题分析

其他问题是指记忆已被检索到，但 LLM 回答错误。

### 典型案例

"""
    for i, issue in enumerate(all_other_issues[:5], 1):
        report += f"""#### 案例 {i}
- **问题**: {issue['question']}
- **标准答案**: {issue['ground_truth']}
- **预测答案**: {issue['prediction']}
- **Judge 评分**: {issue['judge_score']}
- **Judge 说明**: {issue['judge_explanation']}

"""

    report += """
## 优化建议

### 针对记忆抽取问题

1. **[待分析]** 检查 prompt 是否遗漏了某类信息
2. **[待分析]** 检查是否存在信息粒度问题

### 针对记忆检索问题

1. **[待分析]** 检查实体抽取是否完整
2. **[待分析]** 检查同义词覆盖是否充分
3. **[待分析]** 检查向量检索相似度阈值

### 针对其他问题

1. **[待分析]** 检查是否为 Multi-hop 推理问题
2. **[待分析]** 检查是否为时间推理问题
3. **[待分析]** 检查 LLM 回答 prompt 是否需要优化

---

*报告生成完成，请基于具体案例进一步分析优化方向*
"""

    with open(badcase_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"分析报告已保存至: {badcase_dir / 'analysis_report.md'}")


def main(user_pairs: list[str] | None = None):
    """
    主函数

    Args:
        user_pairs: 要分析的对话人列表
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "processed_locomo_test_data"
    output_dir = script_dir / "evaluation_results"

    # 获取所有已评估的对话人
    all_user_pairs = sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "qa_results.json").exists()
    ])

    # 过滤
    if user_pairs:
        all_user_pairs = [u for u in all_user_pairs if u in user_pairs]

    if not all_user_pairs:
        logger.error("未找到已评估的对话人，请先运行 step3_evaluate_qa.py")
        return

    logger.info(f"待分析的对话人数量: {len(all_user_pairs)}")

    all_analyses = []
    for user_pair in all_user_pairs:
        try:
            analysis = analyze_badcases_for_user_pair(user_pair, data_dir, output_dir)
            all_analyses.append(analysis)
            logger.info(f"  抽取问题: {analysis['summary']['extraction_count']}, "
                       f"检索问题: {analysis['summary']['retrieval_count']}, "
                       f"其他问题: {analysis['summary']['other_count']}")
        except Exception as e:
            logger.error(f"分析 {user_pair} 时出错: {e}")
            raise

    # 生成报告
    generate_analysis_report(all_analyses, output_dir)

    logger.info("\n=== 分析完成 ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析 Badcase")
    parser.add_argument(
        "--user-pairs",
        nargs="+",
        help="要分析的对话人列表",
    )
    args = parser.parse_args()

    main(user_pairs=args.user_pairs)
