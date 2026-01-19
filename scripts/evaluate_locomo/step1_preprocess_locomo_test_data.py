import json
import os
import random
import re
from datetime import datetime
from os.path import join

"""
Category 1: Multi-hop
Category 2: Temporal
Category 3: Open-domain
Category 4: Single-hop
Category 5: Adversarial
"""
CATEGORY2STR = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop"
}


def message_to_string(message):
    role = message["speaker"]
    content = message["text"]
    if message.get("img_url") and message.get("blip_caption"):
        return f"<image_message>{role} send an image.\nImage description: {message.get('query')}\n{message.get('blip_caption')}</image_message> {content}"
    return content


def parse_date_string(date_str):
    """转换为puzle echo的时间"""
    # 月份映射字典
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    # 匹配字符串中的时间、AM/PM、日、月和年
    pattern = r"(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*(\d{1,2})\s*([a-zA-Z]+),\s*(\d{4})"
    match = re.search(pattern, date_str, re.IGNORECASE)

    if not match:
        raise ValueError(f"Invalid date format: {date_str}")

    hour = int(match.group(1))
    minute = int(match.group(2))
    am_pm = match.group(3).lower()
    day = int(match.group(4))
    month_str = match.group(5).capitalize()
    year = int(match.group(6))

    # 处理AM/PM转换
    if am_pm == "pm" and hour != 12:
        hour += 12
    elif am_pm == "am" and hour == 12:
        hour = 0

    # 转换月份字符串为数字
    month = month_map.get(month_str)
    if month is None:
        raise ValueError(f"Invalid month: {month_str}")

    return datetime(year, month, day, hour, minute)


def process_locomo_data(locomo_conv_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(locomo_conv_path, "r", encoding="utf8") as f:
        data = json.load(f)
    print(len(data))

    for sample_id, item in enumerate(data):
        # each item represents multiple conversations happened at multi date,
        # each item also contain test data(qa)
        conversation = item["conversation"]
        speaker_a, speaker_b = conversation["speaker_a"], conversation["speaker_b"]

        user_save_dir = join(save_dir, f"{sample_id}__{speaker_a}__{speaker_b}")
        os.makedirs(user_save_dir, exist_ok=True)

        with open(join(user_save_dir, "test_qa.json"), "w", encoding="utf8") as fw:
            qa = item["qa"]
            qa = [i for i in qa if int(i["category"]) in CATEGORY2STR]
            for i in qa:
                i["category"] = CATEGORY2STR[i["category"]]
            random.seed(1)
            random.shuffle(qa)
            json.dump(qa, fw, indent=2, ensure_ascii=False)

        # 先收集所有session信息
        sessions = []
        for sess_id in range(1, 9999999999):
            if not conversation.get(f"session_{sess_id}_date_time") or not conversation.get(f"session_{sess_id}"):
                break
            session_time = parse_date_string(conversation.get(f"session_{sess_id}_date_time"))
            sessions.append({
                "session_time": session_time,
                "messages": conversation.get(f"session_{sess_id}")
            })

        # 按session_time升序排序
        sessions.sort(key=lambda x: x["session_time"])

        # 生成文件和dial_id2content
        dial_id2content = {}
        for new_sess_id, session in enumerate(sessions, start=1):
            session_time = session["session_time"]
            dialogue_string = (f"A conversation between {speaker_a} and {speaker_b}. "
                               f"This conversation takes place on {session_time.isoformat(timespec='seconds')}."
                               f"\n\nThe specific content of the conversation is:\n")

            for message in session["messages"]:
                role = message["speaker"]
                dia_id = message["dia_id"]
                content = message_to_string(message)
                dialogue_string += f"dialogue_id: {dia_id}\n{role}: {content}\n\n\n\n"
                dial_id2content[dia_id] = f"{session_time.isoformat(timespec='seconds')},\t{role}: {content}"

            with open(join(user_save_dir, f"session_{new_sess_id}.txt"), "w", encoding="utf8") as fw:
                fw.writelines(dialogue_string)

        with open(join(user_save_dir, "dial_id2content.json"), "w", encoding="utf8") as fw:
            json.dump(dial_id2content, fw, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data_dir = "./processed_locomo_test_data"
    result = process_locomo_data(
        locomo_conv_path="./locomo10.json",
        save_dir=data_dir
    )
