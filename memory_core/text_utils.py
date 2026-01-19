"""文本处理工具"""


def split_into_lines(text: str) -> list[str]:
    """
    按换行符分割文本

    Args:
        text: 原始文本

    Returns:
        行列表（保留空行）
    """
    return text.split('\n')


def add_line_numbers(text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    为文本添加行号

    Args:
        text: 原始文本

    Returns:
        (带行号文本, 行位置映射)
        行位置映射为 [(start, end), ...] 列表，记录每行在原始文本中的字符位置

    Example:
        输入: "第一行\\n第二行"
        输出: ("[0]: 第一行\\n[1]: 第二行", [(0, 3), (4, 7)])
    """
    lines = split_into_lines(text)
    numbered_lines = []
    positions = []
    current_pos = 0

    for idx, line in enumerate(lines):
        numbered_lines.append(f"[{idx}]: {line}")
        line_start = current_pos
        line_end = current_pos + len(line)
        positions.append((line_start, line_end))
        current_pos = line_end + 1  # +1 for '\n'

    return '\n'.join(numbered_lines), positions


def line_numbers_to_spans(
    line_numbers: list[int],
    line_positions: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    将行号列表转换为字符位置 spans，相邻行合并

    Args:
        line_numbers: 行号列表（可能不连续）
        line_positions: 行位置映射

    Returns:
        合并后的 spans 列表

    Example:
        line_numbers: [0, 1, 3]
        line_positions: [(0, 3), (4, 7), (8, 11), (12, 15)]
        输出: [(0, 7), (12, 15)]  # 0,1 合并, 3 单独
    """
    if not line_numbers:
        return []

    # 过滤有效行号并排序去重
    valid_lines = sorted(set(ln for ln in line_numbers if 0 <= ln < len(line_positions)))
    if not valid_lines:
        return []

    spans = []

    span_start = line_positions[valid_lines[0]][0]
    span_end = line_positions[valid_lines[0]][1]

    for i in range(1, len(valid_lines)):
        current_line = valid_lines[i]
        prev_line = valid_lines[i - 1]

        if current_line == prev_line + 1:
            # 相邻行，扩展 span
            span_end = line_positions[current_line][1]
        else:
            # 非相邻行，保存当前 span，开始新 span
            spans.append((span_start, span_end))
            span_start = line_positions[current_line][0]
            span_end = line_positions[current_line][1]

    spans.append((span_start, span_end))
    return spans


def get_contents_from_spans(
    spans: list[tuple[int, int]],
    text: str
) -> list[str]:
    """
    根据 spans 从文本截取内容

    Args:
        spans: 位置列表 [(start, end), ...]
        text: 原始文本

    Returns:
        内容列表
    """
    return [text[start:end] for start, end in spans]
