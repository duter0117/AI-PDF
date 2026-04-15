# -*- coding: utf-8 -*-
"""
Patch: 把 table_extractor.py 中殘留的 \\uXXXX unicode escape 字面量
      還原成真正的 UTF-8 中文字元，與檔案其餘部分風格一致
"""
import re

TARGET = "core/table_extractor.py"

with open(TARGET, encoding="utf-8") as f:
    content = f.read()

def decode_unicode_escapes_in_str_literals(content):
    """
    把 Python 字串字面量中的 \\uXXXX 轉成真實 Unicode 字元。
    只處理出現在 "..." 字串字面量內的情況。
    """
    count = [0]
    def replacer(m):
        codepoint = int(m.group(1), 16)
        char = chr(codepoint)
        count[0] += 1
        return char

    # 找出所有出現在 " 或 ' 字串字面量中的 \uXXXX（不是 \\uXXXX 雙重轉義）
    # 正則：匹配單個 \u 後接4位十六進位（排除 \\u，即前面有偶數個反斜線的情況）
    pattern = re.compile(r'(?<!\\)\\u([0-9A-Fa-f]{4})')
    new_content = pattern.sub(replacer, content)
    return new_content, count[0]

before = len([l for l in content.split('\n') if r'\u' in l])
new_content, replaced = decode_unicode_escapes_in_str_literals(content)
after = len([l for l in new_content.split('\n') if r'\u' in l])

print(f"替換了 {replaced} 個 \\uXXXX 序列")
print(f"含 \\u 的行數：{before} → {after}")

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done.")
