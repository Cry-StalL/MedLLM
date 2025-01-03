import re


def Extract_closest_tail_uppercase_letter(text: str) -> str:
    """
    从文本中提取最接近尾部的第一个英文大写字母。

    参数：
        text (str): 包含答案的文本。

    返回：
        str: 如果找到，返回最接近尾部的英文大写字母；否则返回空字符串。
    """
    # 使用正则表达式匹配所有大写字母
    uppercase_letters = re.findall(r'[A-Z]', text)

    if not uppercase_letters:
        # 如果没有找到任何大写字母，返回空字符串
        return ""

    # 倒序查找，从尾部最接近的匹配开始
    for char in reversed(text):
        if char in uppercase_letters:
            return char

    # 理论上不会执行到这里，如果没有找到返回空字符串
    return ""


def Extract_answer(response, no_response_count):
    # 使用正则表达式提取字母
    match = re.search(r'\b[A-E]\b', response)  # 查找 'A', 'B', 'C', 'D', 'E'
    if match:
        return match.group(0)  # 返回匹配的字母
    else:
        print(f"{no_response_count} times: Count Do not find any response in {response}")
        no_response_count += 1
        return None  # 如果没有找到匹配的字母