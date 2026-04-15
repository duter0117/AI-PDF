# -*- coding: utf-8 -*-
"""
Patch: 把「左方/中央/右方」的抽象方位說明，改成對應淺藍線格子的具體說法
"""
TARGET = "core/table_extractor.py"

with open(TARGET, encoding="utf-8") as f:
    lines = f.readlines()

# 新版 Rule 1 for 主 prompt (縮排 36 spaces = "                                    ")
NEW_RULE1_MAIN = [
    '                                    "1. 以淺藍格線對應欄位（極度重要）：圖中 2 條垂直線將梁分為左段/中段/右段，2 條水平線將梁分為上緣/梁身/下緣。\\n"\n',
    '                                    "   - 文字位於「上緣左段」→ top_main_bars_left；「上緣中段」→ top_main_bars_mid；「上緣右段」→ top_main_bars_right。\\n"\n',
    '                                    "   - 文字位於「梁身左段」→ stirrups_left；「梁身中段」→ stirrups_middle；「梁身右段」→ stirrups_right。\\n"\n',
    '                                    "   - 文字位於「下緣左段」→ bottom_main_bars_left；「下緣中段」→ bottom_main_bars_mid；「下緣右段」→ bottom_main_bars_right。\\n"\n',
    '                                    "   - 上緣/下緣四個角落的純數字為搭接長度：左上→lap_length_top_left，右上→lap_length_top_right，左下→lap_length_bottom_left，右下→lap_length_bottom_right。\\n"\n',
]

# 新版 Rule 1 for retry prompt (縮排 44 spaces = "                                            ")
NEW_RULE1_RETRY = [
    '                                            "1. 以淺藍格線對應欄位（極度重要）：圖中 2 條垂直線將梁分為左段/中段/右段，2 條水平線將梁分為上緣/梁身/下緣。\\n"\n',
    '                                            "   - 文字位於「上緣左段」→ top_main_bars_left；「上緣中段」→ top_main_bars_mid；「上緣右段」→ top_main_bars_right。\\n"\n',
    '                                            "   - 文字位於「梁身左段」→ stirrups_left；「梁身中段」→ stirrups_middle；「梁身右段」→ stirrups_right。\\n"\n',
    '                                            "   - 文字位於「下緣左段」→ bottom_main_bars_left；「下緣中段」→ bottom_main_bars_mid；「下緣右段」→ bottom_main_bars_right。\\n"\n',
    '                                            "   - 上緣/下緣四個角落的純數字為搭接長度：左上→lap_length_top_left，右上→lap_length_top_right，左下→lap_length_bottom_left，右下→lap_length_bottom_right。\\n"\n',
]

def is_rule1_block_start(line):
    return '實體幾何定位' in line or ('1.' in line and '九宮格' in line and '極度重要' in line)

def is_rule1_sub(line):
    return ('包含「左方」' in line or '包含「中央」' in line or '包含「右方」' in line or
            '`_left`' in line or '`_mid`' in line or '`_right`' in line)

i = 0
patches = 0
while i < len(lines):
    if is_rule1_block_start(lines[i]):
        # 決定用哪個縮排版本
        indent_len = len(lines[i]) - len(lines[i].lstrip())
        is_retry = indent_len > 40
        new_block = NEW_RULE1_RETRY if is_retry else NEW_RULE1_MAIN

        # 把接下來連續的 rule1 sub-lines 找出來一起刪掉
        j = i + 1
        while j < len(lines) and is_rule1_sub(lines[j]):
            j += 1

        # 替換：用 new_block 取代 lines[i:j]
        lines[i:j] = new_block
        patches += 1
        print(f"[OK] patch #{patches} 替換了原始行 {i+1}~{j}（縮排={'retry' if is_retry else 'main'}）")
        i += len(new_block)
    else:
        i += 1

with open(TARGET, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"完成：共替換 {patches} 個 Rule 1 區塊")
