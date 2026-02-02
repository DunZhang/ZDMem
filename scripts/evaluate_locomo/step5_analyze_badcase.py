"""Step 5: Badcase 分析脚本

分析错误原因，区分记忆抽取问题、检索问题、其他问题
"""

import sys
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_core.local_manager import LocalFileMemoryManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "processed_locomo_test_data"
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate_results"
BADCASE_DIR = SCRIPT_DIR / "badcase_analysis"


def load_session_files(user_pair: str) -> dict[str, str]:
    """加载所有 session 文件内容"""
    user_data_dir = DATA_DIR / user_pair
    sessions = {}
    for f in user_data_dir.glob("session_*.txt"):
        sessions[f.stem] = f.read_text(encoding="utf-8")
    return sessions


def parse_evidence(evidence_ids: list[str], sessions: dict[str, str]) -> list[dict]:
    """解析 evidence ID 对应的原文

    evidence 格式: "D{session_num}:{dialogue_id}"
    例如 "D1:2" 表示 session_1.txt 中 dialogue_id 为 D1:2 的对话
    """
    results = []
    for eid in evidence_ids:
        # 解析格式 D{session}:{line}
        match = re.match(r'D(\d+):(\d+)', eid)
        if not match:
            continue

        session_num = int(match.group(1))
        session_key = f"session_{session_num}"

        if session_key not in sessions:
            continue

        text = sessions[session_key]
        # 查找 dialogue_id 对应的行
        pattern = rf'dialogue_id: {re.escape(eid)},\s*(.+?)(?=\n\ndia|$)'
        found = re.search(pattern, text, re.DOTALL)

        if found:
            results.append({
                "evidence_id": eid,
                "session": session_key,
                "content": found.group(0).strip()
            })

    return results


def check_memory_coverage(
    evidence_texts: list[dict],
    all_memories: list
) -> dict:
    """检查 evidence 是否被记忆覆盖

    通过关键词匹配和语义相似度来判断
    """
    if not evidence_texts:
        return {
            "is_covered": True,
            "coverage_details": [],
            "relevant_memories": []
        }

    coverage_details = []
    relevant_memories = []

    for ev in evidence_texts:
        ev_content = ev["content"].lower()
        ev_covered = False
        covering_memories = []

        for mem in all_memories:
            mem_content = mem.content.lower()
            # 简单的关键词匹配
            # 检查记忆是否包含 evidence 中的关键内容
            ev_words = set(re.findall(r'\w+', ev_content))
            mem_words = set(re.findall(r'\w+', mem_content))

            # 计算重叠率
            if len(ev_words) > 0:
                overlap = len(ev_words & mem_words) / len(ev_words)
                if overlap > 0.3:  # 30% 重叠认为覆盖
                    ev_covered = True
                    covering_memories.append({
                        "memory_id": mem.id,
                        "memory_content": mem.content,
                        "overlap": overlap
                    })

        coverage_details.append({
            "evidence_id": ev["evidence_id"],
            "evidence_content": ev["content"][:200],
            "is_covered": ev_covered,
            "covering_memories": covering_memories
        })

        if covering_memories:
            for cm in covering_memories:
                # 找到对应的完整 memory 对象
                for mem in all_memories:
                    if mem.id == cm["memory_id"]:
                        if mem not in relevant_memories:
                            relevant_memories.append(mem)
                        break

    all_covered = all(d["is_covered"] for d in coverage_details)

    return {
        "is_covered": all_covered,
        "coverage_details": coverage_details,
        "relevant_memories": relevant_memories
    }


def classify_error(
    prediction: dict,
    all_memories: list,
    sessions: dict[str, str]
) -> dict:
    """分类错误类型"""
    evidence_ids = prediction.get("evidence", [])
    retrieved_memories = prediction.get("retrieved_memories", [])

    # 解析 evidence
    evidence_texts = parse_evidence(evidence_ids, sessions)

    # 检查记忆覆盖
    coverage = check_memory_coverage(evidence_texts, all_memories)

    # 检查检索结果
    retrieved_ids = {m["id"] for m in retrieved_memories}
    relevant_in_retrieved = [
        m for m in coverage["relevant_memories"]
        if m.id in retrieved_ids
    ]

    # 分类
    if not coverage["is_covered"]:
        error_type = "extraction"
        error_detail = "关键信息未被抽取为记忆"
        missing_info = [
            d for d in coverage["coverage_details"]
            if not d["is_covered"]
        ]
    elif coverage["relevant_memories"] and not relevant_in_retrieved:
        error_type = "retrieval"
        error_detail = "记忆存在但未被检索到"
        missing_info = [
            {"memory_id": m.id, "memory_content": m.content}
            for m in coverage["relevant_memories"]
        ]
    else:
        error_type = "reasoning"
        error_detail = "记忆已检索到但回答错误，可能是LLM理解/推理问题"
        missing_info = []

    return {
        "error_type": error_type,
        "error_detail": error_detail,
        "missing_info": missing_info,
        "evidence_texts": evidence_texts,
        "coverage": coverage,
        "retrieved_count": len(retrieved_memories),
        "relevant_retrieved_count": len(relevant_in_retrieved)
    }


def analyze_badcases_for_user(user_pair: str) -> dict:
    """分析一对用户的 badcase"""
    # 加载数据
    judgement_file = INTERMEDIATE_DIR / "judgements" / f"{user_pair}.jsonl"
    prediction_file = INTERMEDIATE_DIR / "predictions" / f"{user_pair}.jsonl"
    memories_dir = INTERMEDIATE_DIR / "memories"

    if not judgement_file.exists():
        logger.error(f"评估结果不存在，请先运行 step4")
        return None

    # 加载评估结果
    judgements = []
    with open(judgement_file, encoding="utf-8") as f:
        for line in f:
            judgements.append(json.loads(line))

    # 加载预测结果（包含 retrieved_memories）
    predictions = {}
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred["qa_id"]] = pred

    # 加载所有记忆
    manager = LocalFileMemoryManager(data_dir=str(memories_dir))
    all_memories = manager._load_memories(user_pair)

    # 加载 session 文件
    sessions = load_session_files(user_pair)

    logger.info(f"开始分析 {user_pair}，共 {len(judgements)} 条评估")
    logger.info(f"  - 记忆数: {len(all_memories)}")
    logger.info(f"  - Session 数: {len(sessions)}")

    # 分析 badcase
    badcases = []
    stats = {
        "total": len(judgements),
        "correct": 0,
        "incorrect": 0,
        "by_type": defaultdict(int),
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0})
    }

    for j in judgements:
        category = j.get("category", "Unknown").strip()
        stats["by_category"][category]["total"] += 1

        if j["is_correct"]:
            stats["correct"] += 1
            stats["by_category"][category]["correct"] += 1
            continue

        stats["incorrect"] += 1

        # 获取对应的预测结果
        pred = predictions.get(j["qa_id"], {})
        pred["evidence"] = j.get("evidence", pred.get("evidence", []))

        # 分类错误
        error_info = classify_error(pred, all_memories, sessions)

        stats["by_type"][error_info["error_type"]] += 1

        badcase = {
            "qa_id": j["qa_id"],
            "question": j["question"],
            "ground_truth": j["ground_truth"],
            "predicted_answer": j["predicted_answer"],
            "category": category,
            "judge_reasoning": j.get("reasoning", ""),
            **error_info
        }
        badcases.append(badcase)

    return {
        "user_pair": user_pair,
        "stats": stats,
        "badcases": badcases
    }


def generate_report(analysis: dict, output_dir: Path):
    """生成分析报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    user_pair = analysis["user_pair"]
    stats = analysis["stats"]
    badcases = analysis["badcases"]

    # 生成 summary.md
    accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    summary_lines = [
        f"# Locomo 评估 Badcase 分析报告",
        f"",
        f"## 概览",
        f"",
        f"- **用户对**: {user_pair}",
        f"- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **总问题数**: {stats['total']}",
        f"- **正确数**: {stats['correct']}",
        f"- **错误数**: {stats['incorrect']}",
        f"- **正确率**: {accuracy:.2%}",
        f"",
        f"## 错误类型分布",
        f"",
        f"| 错误类型 | 数量 | 占比 |",
        f"|----------|------|------|",
    ]

    for error_type, count in sorted(stats["by_type"].items()):
        pct = count / stats["incorrect"] * 100 if stats["incorrect"] > 0 else 0
        type_name = {
            "extraction": "记忆抽取问题",
            "retrieval": "检索问题",
            "reasoning": "推理/其他问题"
        }.get(error_type, error_type)
        summary_lines.append(f"| {type_name} | {count} | {pct:.1f}% |")

    summary_lines.extend([
        f"",
        f"## 分类别统计",
        f"",
        f"| Category | 总数 | 正确 | 错误 | 正确率 |",
        f"|----------|------|------|------|--------|",
    ])

    for cat, cat_stats in sorted(stats["by_category"].items()):
        cat_acc = cat_stats["correct"] / cat_stats["total"] * 100 if cat_stats["total"] > 0 else 0
        incorrect = cat_stats["total"] - cat_stats["correct"]
        summary_lines.append(
            f"| {cat} | {cat_stats['total']} | {cat_stats['correct']} | {incorrect} | {cat_acc:.1f}% |"
        )

    summary_lines.extend([
        f"",
        f"## Badcase 详情",
        f"",
        f"详细的 badcase 分析请查看 `detailed_cases/` 目录。",
        f"",
        f"### 按错误类型汇总",
        f"",
    ])

    # 按错误类型分组
    by_type = defaultdict(list)
    for bc in badcases:
        by_type[bc["error_type"]].append(bc)

    for error_type in ["extraction", "retrieval", "reasoning"]:
        cases = by_type.get(error_type, [])
        if not cases:
            continue

        type_name = {
            "extraction": "记忆抽取问题",
            "retrieval": "检索问题",
            "reasoning": "推理/其他问题"
        }.get(error_type, error_type)

        summary_lines.extend([
            f"#### {type_name} ({len(cases)} 条)",
            f"",
        ])

        for bc in cases[:5]:  # 只显示前5条
            gt = str(bc['ground_truth'])
            pred = str(bc['predicted_answer'])
            summary_lines.extend([
                f"- **Q{bc['qa_id']}**: {bc['question'][:60]}...",
                f"  - 预期: {gt[:50]}...",
                f"  - 预测: {pred[:50]}...",
                f"",
            ])

        if len(cases) > 5:
            summary_lines.append(f"... 共 {len(cases)} 条，详见 detailed_cases/")
        summary_lines.append("")

    # 写入 summary.md
    summary_file = output_dir / "summary.md"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    # 保存详细 badcase 数据
    detailed_dir = output_dir / "detailed_cases"
    detailed_dir.mkdir(exist_ok=True)

    # 保存所有 badcase 为 JSON
    all_badcases_file = output_dir / "all_badcases.json"
    with open(all_badcases_file, "w", encoding="utf-8") as f:
        json.dump(badcases, f, ensure_ascii=False, indent=2, default=str)

    # 为每个 badcase 生成详细 markdown
    for bc in badcases:
        case_file = detailed_dir / f"case_{bc['qa_id']:03d}.md"

        type_name = {
            "extraction": "记忆抽取问题",
            "retrieval": "检索问题",
            "reasoning": "推理/其他问题"
        }.get(bc["error_type"], bc["error_type"])

        lines = [
            f"# Case #{bc['qa_id']:03d} - {type_name}",
            f"",
            f"## 基本信息",
            f"",
            f"- **Category**: {bc['category']}",
            f"- **Error Type**: {type_name}",
            f"- **Error Detail**: {bc['error_detail']}",
            f"",
            f"## 问题与答案",
            f"",
            f"**Question**: {bc['question']}",
            f"",
            f"**Ground Truth**: {bc['ground_truth']}",
            f"",
            f"**Predicted Answer**: {bc['predicted_answer']}",
            f"",
            f"**Judge Reasoning**: {bc['judge_reasoning']}",
            f"",
            f"## Evidence 分析",
            f"",
        ]

        if bc["evidence_texts"]:
            for ev in bc["evidence_texts"]:
                lines.extend([
                    f"### {ev['evidence_id']} ({ev['session']})",
                    f"",
                    f"```",
                    f"{ev['content']}",
                    f"```",
                    f"",
                ])
        else:
            lines.append("*No evidence texts found*\n")

        lines.extend([
            f"## 记忆覆盖检查",
            f"",
            f"- **是否覆盖**: {'是' if bc['coverage']['is_covered'] else '否'}",
            f"- **检索到的记忆数**: {bc['retrieved_count']}",
            f"- **相关且被检索到的数**: {bc['relevant_retrieved_count']}",
            f"",
        ])

        if bc["missing_info"]:
            lines.append("### 缺失信息")
            lines.append("")
            for info in bc["missing_info"][:5]:
                if isinstance(info, dict):
                    if "evidence_content" in info:
                        lines.append(f"- Evidence: {info['evidence_content'][:100]}...")
                    elif "memory_content" in info:
                        lines.append(f"- Memory: {info['memory_content'][:100]}...")
            lines.append("")

        case_file.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"报告已生成: {output_dir}")


def extract_dialogue_ids_from_memory(memory: dict) -> set[str]:
    """从单个记忆的references中提取所有dialogue_id

    references 格式: ["D16:10", "D16:11", "D16:12"]
    """
    references = memory.get("references", [])
    return set(ref.strip().lower() for ref in references)


def calculate_recall_at_k(user_pair: str, max_k: int = 12) -> dict:
    """计算记忆检索的Recall@K

    对于每个问题：
    - evidence中包含需要的dialogue_id列表
    - 检查这些dialogue_id是否在前k个检索到的记忆中

    Args:
        user_pair: 用户对名称
        max_k: 最大K值

    Returns:
        包含各K值召回率的字典
    """
    prediction_file = INTERMEDIATE_DIR / "predictions" / f"{user_pair}.jsonl"

    if not prediction_file.exists():
        logger.error(f"预测文件不存在: {prediction_file}")
        return None

    # 加载预测结果
    predictions = []
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))

    logger.info(f"加载 {user_pair} 共 {len(predictions)} 条预测")

    # 统计每个K值的召回情况
    recall_hits = {k: 0 for k in range(1, max_k + 1)}
    valid_count = 0  # 有evidence的问题数

    for pred in predictions:
        evidence_ids = set(e.strip().lower() for e in pred.get("evidence", []))
        if not evidence_ids:
            # 没有evidence的问题跳过
            continue

        valid_count += 1
        retrieved_memories = pred.get("retrieved_memories", [])

        # 对于每个K值，检查evidence是否都在前K个记忆中
        for k in range(1, max_k + 1):
            top_k_memories = retrieved_memories[:k]

            # 收集前K个记忆中所有的dialogue_id
            retrieved_dialogue_ids = set()
            for mem in top_k_memories:
                retrieved_dialogue_ids.update(extract_dialogue_ids_from_memory(mem))

            # 检查所有evidence是否都被召回
            if evidence_ids.issubset(retrieved_dialogue_ids):
                recall_hits[k] += 1

    # 计算召回率
    recall_at_k = {}
    for k in range(1, max_k + 1):
        recall_at_k[k] = recall_hits[k] / valid_count if valid_count > 0 else 0

    return {
        "user_pair": user_pair,
        "valid_questions": valid_count,
        "recall_hits": recall_hits,
        "recall_at_k": recall_at_k
    }


def analyze_recall_all_users(max_k: int = 12) -> dict:
    """分析所有用户对的Recall@K并汇总"""
    user_pairs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

    all_results = []
    total_hits = {k: 0 for k in range(1, max_k + 1)}
    total_valid = 0

    for user_pair in sorted(user_pairs):
        result = calculate_recall_at_k(user_pair, max_k)
        if result is None:
            continue

        all_results.append(result)
        total_valid += result["valid_questions"]
        for k in range(1, max_k + 1):
            total_hits[k] += result["recall_hits"][k]

    # 计算总体召回率
    overall_recall = {k: total_hits[k] / total_valid if total_valid > 0 else 0 for k in range(1, max_k + 1)}

    return {
        "user_results": all_results,
        "total_valid_questions": total_valid,
        "total_hits": total_hits,
        "overall_recall_at_k": overall_recall
    }


def generate_full_report(output_dir: Path = None):
    """生成完整的评估报告，包含所有指标"""
    if output_dir is None:
        output_dir = BADCASE_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    user_pairs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

    logger.info(f"开始生成完整报告，共 {len(user_pairs)} 个用户对")

    # ========== 收集所有数据 ==========
    all_analyses = []
    global_stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "by_type": defaultdict(int),
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
    }
    all_badcases = []

    # Token 统计
    total_prompt_tokens = 0
    total_completion_tokens = 0
    qa_count = 0

    for user_pair in sorted(user_pairs):
        analysis = analyze_badcases_for_user(user_pair)
        if analysis is None:
            continue

        all_analyses.append(analysis)
        stats = analysis["stats"]

        # 汇总统计
        global_stats["total"] += stats["total"]
        global_stats["correct"] += stats["correct"]
        global_stats["incorrect"] += stats["incorrect"]

        for etype, count in stats["by_type"].items():
            global_stats["by_type"][etype] += count

        for cat, cat_stats in stats["by_category"].items():
            global_stats["by_category"][cat]["total"] += cat_stats["total"]
            global_stats["by_category"][cat]["correct"] += cat_stats["correct"]

        # 收集 badcases
        for bc in analysis["badcases"]:
            bc["user_pair"] = user_pair
            all_badcases.append(bc)

        # 收集 token 统计
        prediction_file = INTERMEDIATE_DIR / "predictions" / f"{user_pair}.jsonl"
        if prediction_file.exists():
            with open(prediction_file, encoding="utf-8") as f:
                for line in f:
                    pred = json.loads(line)
                    total_prompt_tokens += pred.get("prompt_tokens", 0)
                    total_completion_tokens += pred.get("completion_tokens", 0)
                    qa_count += 1

    # ========== Recall@K 分析 ==========
    recall_results = analyze_recall_all_users(max_k=12)

    # ========== 生成报告 ==========
    accuracy = global_stats["correct"] / global_stats["total"] if global_stats["total"] > 0 else 0

    # 计算平均 token
    avg_prompt_tokens = total_prompt_tokens / qa_count if qa_count > 0 else 0
    avg_completion_tokens = total_completion_tokens / qa_count if qa_count > 0 else 0

    report_lines = [
        f"# Locomo 评估完整报告",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## 1. 总体概览",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 用户对数量 | {len(all_analyses)} |",
        f"| 总问题数 | {global_stats['total']} |",
        f"| 正确数 | {global_stats['correct']} |",
        f"| 错误数 | {global_stats['incorrect']} |",
        f"| **正确率 (Accuracy)** | **{accuracy:.2%}** |",
        f"| 总 Prompt Tokens | {total_prompt_tokens:,} |",
        f"| 总 Completion Tokens | {total_completion_tokens:,} |",
        f"| 平均 Prompt Tokens/问答 | {avg_prompt_tokens:.1f} |",
        f"| 平均 Completion Tokens/问答 | {avg_completion_tokens:.1f} |",
        f"",
        f"---",
        f"",
        f"## 2. 记忆检索 Recall@K",
        f"",
        f"衡量检索到的前 K 个记忆是否包含回答问题所需的全部 evidence。",
        f"",
        f"| K | 命中数 | Recall@K |",
        f"|---|--------|----------|",
    ]

    for k in [1, 3, 5, 10, 12]:
        hits = recall_results["total_hits"].get(k, 0)
        recall = recall_results["overall_recall_at_k"].get(k, 0)
        report_lines.append(f"| {k} | {hits} | {recall:.2%} |")

    report_lines.extend([
        f"",
        f"*有效问题数（有 evidence 标注）: {recall_results['total_valid_questions']}*",
        f"",
        f"### 各用户对 Recall@K",
        f"",
        f"| 用户对 | 有效问题数 | R@1 | R@3 | R@5 | R@10 |",
        f"|--------|-----------|-----|-----|-----|------|",
    ])

    for result in recall_results["user_results"]:
        r = result["recall_at_k"]
        report_lines.append(
            f"| {result['user_pair']} | {result['valid_questions']} | "
            f"{r.get(1, 0):.1%} | {r.get(3, 0):.1%} | {r.get(5, 0):.1%} | {r.get(10, 0):.1%} |"
        )

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. 错误类型分布",
        f"",
        f"对错误进行分类分析，识别系统瓶颈。",
        f"",
        f"| 错误类型 | 数量 | 占错误比例 | 占总体比例 |",
        f"|----------|------|-----------|-----------|",
    ])

    for error_type in ["extraction", "retrieval", "reasoning"]:
        count = global_stats["by_type"].get(error_type, 0)
        pct_of_error = count / global_stats["incorrect"] * 100 if global_stats["incorrect"] > 0 else 0
        pct_of_total = count / global_stats["total"] * 100 if global_stats["total"] > 0 else 0
        type_name = {
            "extraction": "记忆抽取问题",
            "retrieval": "检索问题",
            "reasoning": "推理/其他问题"
        }.get(error_type, error_type)
        report_lines.append(f"| {type_name} | {count} | {pct_of_error:.1f}% | {pct_of_total:.1f}% |")

    report_lines.extend([
        f"",
        f"**错误类型说明**:",
        f"- **记忆抽取问题**: 关键信息未被抽取成记忆，导致无法回答",
        f"- **检索问题**: 记忆已抽取但检索时未被召回",
        f"- **推理/其他问题**: 记忆已检索到但 LLM 回答错误（理解/推理问题）",
        f"",
        f"---",
        f"",
        f"## 4. 分类别统计 (Category)",
        f"",
        f"按问题类型分析正确率。",
        f"",
        f"| Category | 总数 | 正确 | 错误 | 正确率 |",
        f"|----------|------|------|------|--------|",
    ])

    sorted_categories = sorted(
        global_stats["by_category"].items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )
    for cat, cat_stats in sorted_categories:
        cat_acc = cat_stats["correct"] / cat_stats["total"] * 100 if cat_stats["total"] > 0 else 0
        incorrect = cat_stats["total"] - cat_stats["correct"]
        report_lines.append(
            f"| {cat} | {cat_stats['total']} | {cat_stats['correct']} | {incorrect} | {cat_acc:.1f}% |"
        )

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. 各用户对统计",
        f"",
        f"| 用户对 | 总数 | 正确 | 错误 | 正确率 | 抽取错误 | 检索错误 | 推理错误 |",
        f"|--------|------|------|------|--------|---------|---------|---------|",
    ])

    for analysis in all_analyses:
        stats = analysis["stats"]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        report_lines.append(
            f"| {analysis['user_pair']} | {stats['total']} | {stats['correct']} | {stats['incorrect']} | "
            f"{acc:.1f}% | {stats['by_type'].get('extraction', 0)} | "
            f"{stats['by_type'].get('retrieval', 0)} | {stats['by_type'].get('reasoning', 0)} |"
        )

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 6. Badcase 样例",
        f"",
        f"### 6.1 记忆抽取问题样例",
        f"",
    ])

    extraction_cases = [bc for bc in all_badcases if bc["error_type"] == "extraction"][:3]
    if extraction_cases:
        for bc in extraction_cases:
            gt = str(bc['ground_truth'])
            pred = str(bc['predicted_answer'])
            report_lines.extend([
                f"**[{bc['user_pair']}] Q{bc['qa_id']}: {bc['question']}**",
                f"- 预期答案: {gt[:100]}{'...' if len(gt) > 100 else ''}",
                f"- 预测答案: {pred[:100]}{'...' if len(pred) > 100 else ''}",
                f"- 问题: {bc['error_detail']}",
                f"",
            ])
    else:
        report_lines.append("*无*\n")

    report_lines.extend([
        f"### 6.2 检索问题样例",
        f"",
    ])

    retrieval_cases = [bc for bc in all_badcases if bc["error_type"] == "retrieval"][:3]
    if retrieval_cases:
        for bc in retrieval_cases:
            gt = str(bc['ground_truth'])
            pred = str(bc['predicted_answer'])
            report_lines.extend([
                f"**[{bc['user_pair']}] Q{bc['qa_id']}: {bc['question']}**",
                f"- 预期答案: {gt[:100]}{'...' if len(gt) > 100 else ''}",
                f"- 预测答案: {pred[:100]}{'...' if len(pred) > 100 else ''}",
                f"- 检索到记忆数: {bc['retrieved_count']}，相关被检索到: {bc['relevant_retrieved_count']}",
                f"",
            ])
    else:
        report_lines.append("*无*\n")

    report_lines.extend([
        f"### 6.3 推理问题样例",
        f"",
    ])

    reasoning_cases = [bc for bc in all_badcases if bc["error_type"] == "reasoning"][:3]
    if reasoning_cases:
        for bc in reasoning_cases:
            gt = str(bc['ground_truth'])
            pred = str(bc['predicted_answer'])
            judge_reason = str(bc.get('judge_reasoning', ''))
            report_lines.extend([
                f"**[{bc['user_pair']}] Q{bc['qa_id']}: {bc['question']}**",
                f"- 预期答案: {gt[:100]}{'...' if len(gt) > 100 else ''}",
                f"- 预测答案: {pred[:100]}{'...' if len(pred) > 100 else ''}",
                f"- Judge 评价: {judge_reason[:150]}{'...' if len(judge_reason) > 150 else ''}",
                f"",
            ])
    else:
        report_lines.append("*无*\n")

    report_lines.extend([
        f"---",
        f"",
        f"## 7. 结论与建议",
        f"",
    ])

    # 自动生成建议
    extraction_pct = global_stats["by_type"].get("extraction", 0) / global_stats["incorrect"] * 100 if global_stats["incorrect"] > 0 else 0
    retrieval_pct = global_stats["by_type"].get("retrieval", 0) / global_stats["incorrect"] * 100 if global_stats["incorrect"] > 0 else 0
    reasoning_pct = global_stats["by_type"].get("reasoning", 0) / global_stats["incorrect"] * 100 if global_stats["incorrect"] > 0 else 0

    report_lines.append("### 主要发现\n")

    if extraction_pct > 30:
        report_lines.append(f"- **记忆抽取是主要瓶颈** ({extraction_pct:.1f}% 的错误)：需优化记忆抽取策略，确保关键信息被正确提取")
    if retrieval_pct > 30:
        report_lines.append(f"- **检索效果需提升** ({retrieval_pct:.1f}% 的错误)：考虑优化 embedding 模型或检索策略")
    if reasoning_pct > 30:
        report_lines.append(f"- **LLM 推理能力有限** ({reasoning_pct:.1f}% 的错误)：可尝试更强的模型或优化 prompt")

    r1 = recall_results["overall_recall_at_k"].get(1, 0)
    r5 = recall_results["overall_recall_at_k"].get(5, 0)
    r10 = recall_results["overall_recall_at_k"].get(10, 0)

    report_lines.append(f"\n### Recall 分析\n")
    report_lines.append(f"- Recall@1: {r1:.2%}, Recall@5: {r5:.2%}, Recall@10: {r10:.2%}")

    if r1 < 0.3:
        report_lines.append(f"- Recall@1 较低，检索排序需优化")
    if r10 - r1 > 0.3:
        report_lines.append(f"- Recall@1 到 Recall@10 提升显著 (+{(r10-r1):.1%})，说明相关记忆存在但排序靠后")

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"*详细的 badcase 数据保存在 `all_badcases.json`*",
        f"",
    ])

    # 写入报告
    report_file = output_dir / "full_report.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    # 保存所有 badcases
    badcases_file = output_dir / "all_badcases.json"
    with open(badcases_file, "w", encoding="utf-8") as f:
        json.dump(all_badcases, f, ensure_ascii=False, indent=2, default=str)

    # 保存 recall 详情
    recall_file = output_dir / "recall_details.json"
    with open(recall_file, "w", encoding="utf-8") as f:
        json.dump(recall_results, f, ensure_ascii=False, indent=2)

    # 保存汇总统计
    summary_stats = {
        "accuracy": accuracy,
        "total": global_stats["total"],
        "correct": global_stats["correct"],
        "incorrect": global_stats["incorrect"],
        "by_type": dict(global_stats["by_type"]),
        "by_category": {k: dict(v) for k, v in global_stats["by_category"].items()},
        "recall_at_k": recall_results["overall_recall_at_k"],
        "token_stats": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "qa_count": qa_count,
        },
    }
    stats_file = output_dir / "summary_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    # 控制台输出摘要
    print(f"\n{'='*60}")
    print("Locomo 评估完整报告")
    print(f"{'='*60}")
    print(f"\n总体正确率: {accuracy:.2%} ({global_stats['correct']}/{global_stats['total']})")
    print(f"\nRecall@K: R@1={r1:.2%}  R@5={r5:.2%}  R@10={r10:.2%}")
    print(f"\nToken 统计 (QA 阶段):")
    print(f"  - 平均 prompt_tokens: {avg_prompt_tokens:.1f}")
    print(f"  - 平均 completion_tokens: {avg_completion_tokens:.1f}")
    print(f"  - 总计: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")
    print(f"\n错误分布:")
    print(f"  - 记忆抽取: {global_stats['by_type'].get('extraction', 0)} ({extraction_pct:.1f}%)")
    print(f"  - 检索问题: {global_stats['by_type'].get('retrieval', 0)} ({retrieval_pct:.1f}%)")
    print(f"  - 推理问题: {global_stats['by_type'].get('reasoning', 0)} ({reasoning_pct:.1f}%)")
    print(f"\n报告已保存至: {output_dir}")
    print(f"{'='*60}\n")

    logger.info(f"完整报告已生成: {output_dir}")
    return output_dir


def main():
    """主函数：生成完整评估报告"""
    import argparse

    parser = argparse.ArgumentParser(description="Locomo 评估分析与报告生成")
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help="仅输出 Recall@K 分析（旧模式）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告输出目录"
    )
    args = parser.parse_args()

    if args.recall_only:
        # 旧模式：仅输出 recall
        results = analyze_recall_all_users(max_k=12)
        print(f"\n{'='*60}")
        print("记忆检索 Recall@K 分析")
        print(f"{'='*60}")
        print(f"\n总体结果 (有效问题数: {results['total_valid_questions']})")
        print(f"\n{'K':<4} {'Hits':<8} {'Recall':<10}")
        print("-" * 24)
        for k in range(1, 13):
            print(f"{k:<4} {results['total_hits'][k]:<8} {results['overall_recall_at_k'][k]:.2%}")
    else:
        # 完整报告模式
        output_dir = Path(args.output_dir) if args.output_dir else None
        generate_full_report(output_dir)


if __name__ == "__main__":
    main()
