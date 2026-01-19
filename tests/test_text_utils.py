"""文本工具测试"""

from memory_core.text_utils import (
    split_into_lines,
    add_line_numbers,
    line_numbers_to_spans,
    get_contents_from_spans,
)


class TestSplitIntoLines:
    """split_into_lines 测试"""

    def test_multiple_lines(self):
        """多行文本"""
        assert split_into_lines("a\nb\nc") == ["a", "b", "c"]

    def test_single_line(self):
        """单行文本"""
        assert split_into_lines("single") == ["single"]

    def test_empty_string(self):
        """空字符串"""
        assert split_into_lines("") == [""]

    def test_with_empty_lines(self):
        """包含空行"""
        assert split_into_lines("a\n\nc") == ["a", "", "c"]

    def test_trailing_newline(self):
        """末尾换行"""
        assert split_into_lines("a\nb\n") == ["a", "b", ""]


class TestAddLineNumbers:
    """add_line_numbers 测试"""

    def test_basic(self):
        """基本测试"""
        text = "第一行\n第二行"
        numbered, positions = add_line_numbers(text)

        assert "[0]: 第一行" in numbered
        assert "[1]: 第二行" in numbered
        assert len(positions) == 2

    def test_positions(self):
        """位置映射测试"""
        text = "abc\ndefg\nhi"
        numbered, positions = add_line_numbers(text)

        assert positions[0] == (0, 3)    # "abc"
        assert positions[1] == (4, 8)    # "defg"
        assert positions[2] == (9, 11)   # "hi"

    def test_single_line(self):
        """单行文本"""
        text = "hello"
        numbered, positions = add_line_numbers(text)

        assert numbered == "[0]: hello"
        assert positions == [(0, 5)]

    def test_empty_lines(self):
        """包含空行"""
        text = "a\n\nb"
        numbered, positions = add_line_numbers(text)

        assert "[0]: a" in numbered
        assert "[1]: " in numbered
        assert "[2]: b" in numbered
        assert len(positions) == 3


class TestLineNumbersToSpans:
    """line_numbers_to_spans 测试"""

    def test_continuous_lines(self):
        """连续行应合并"""
        positions = [(0, 3), (4, 7), (8, 11), (12, 15)]
        result = line_numbers_to_spans([0, 1], positions)
        assert result == [(0, 7)]

    def test_non_continuous_lines(self):
        """非连续行"""
        positions = [(0, 3), (4, 7), (8, 11), (12, 15)]
        result = line_numbers_to_spans([0, 2], positions)
        assert result == [(0, 3), (8, 11)]

    def test_mixed_lines(self):
        """混合情况"""
        positions = [(0, 3), (4, 7), (8, 11), (12, 15)]
        result = line_numbers_to_spans([0, 1, 3], positions)
        assert result == [(0, 7), (12, 15)]

    def test_empty_list(self):
        """空列表"""
        positions = [(0, 3), (4, 7)]
        result = line_numbers_to_spans([], positions)
        assert result == []

    def test_single_line(self):
        """单行"""
        positions = [(0, 3), (4, 7)]
        result = line_numbers_to_spans([1], positions)
        assert result == [(4, 7)]

    def test_unsorted_input(self):
        """未排序输入"""
        positions = [(0, 3), (4, 7), (8, 11)]
        result = line_numbers_to_spans([2, 0, 1], positions)
        assert result == [(0, 11)]  # 0,1,2 连续，合并为一个 span

    def test_duplicate_lines(self):
        """重复行号"""
        positions = [(0, 3), (4, 7), (8, 11)]
        result = line_numbers_to_spans([0, 0, 1, 1], positions)
        assert result == [(0, 7)]


class TestGetContentsFromSpans:
    """get_contents_from_spans 测试"""

    def test_basic(self):
        """基本测试"""
        text = "AAABBBCCCDDD"
        spans = [(0, 3), (6, 9)]
        result = get_contents_from_spans(spans, text)
        assert result == ["AAA", "CCC"]

    def test_full_text(self):
        """完整文本"""
        text = "hello world"
        spans = [(0, 11)]
        result = get_contents_from_spans(spans, text)
        assert result == ["hello world"]

    def test_empty_spans(self):
        """空 spans"""
        text = "hello"
        result = get_contents_from_spans([], text)
        assert result == []

    def test_unicode(self):
        """Unicode 文本"""
        text = "你好世界"
        spans = [(0, 2), (2, 4)]
        result = get_contents_from_spans(spans, text)
        assert result == ["你好", "世界"]
