"""pytest 配置"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载项目根目录的 .env
load_dotenv(project_root / ".env")


@pytest.fixture
def test_text():
    """测试用对话文本"""
    return """A conversation between User and Assistant. This conversation takes place on 2024-12-20T18:30:00.

dialogue_id: D1:1
User: 今天我和老婆小美去深圳南山区的海底捞吃火锅，她特别喜欢番茄锅底。


dialogue_id: D1:2
Assistant: 听起来很棒！你们在海底捞点了什么菜？


dialogue_id: D1:3
User: 第二天我们去了华强北买了一台 iPhone 16 Pro，花了 8999 元。


dialogue_id: D1:4
User: 小美说她下周要去上海出差，大概待一周左右。


dialogue_id: D1:5
User: 对了，我最近在学 Python 的 FastAPI 框架，感觉比 Flask 好用多了。
"""


@pytest.fixture
def test_user_id():
    """测试用户 ID"""
    return "test_user_001"
