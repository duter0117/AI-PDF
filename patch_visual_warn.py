# -*- coding: utf-8 -*-
"""
Patch v3: 移除上次的 unicode-escape 廢行，重新用真實 UTF-8 字元插入
"""
TARGET = "core/table_extractor.py"

# ---- 真實 UTF-8 字元版本的警告段落 ----
VISUAL_WARN_MAIN = (
    '                                     "【⚠️ 數字視覺混淆警告 — 填寫前請核對筆劃！】\\n"\r\n'
    '                                     "下列字元因印刷/掃描精度，在工程圖面中極易混淆，請逐一比對特徵後再填入：\\n"\r\n'
    '                                     "  · 5 vs 6：5 頂部是「平直橫線」；6 底部有「封閉圓圈」。\\n"\r\n'
    '                                     "  · 0 vs 8：0 是單一橢圓；8 是上下兩個圓緊接，中間有交叉點。\\n"\r\n'
    '                                     "  · 1 vs 7：1 是單純垂直線；7 頂端有「水平橫線向右延伸」。\\n"\r\n'
    '                                     "  · 1 vs 2：1 只有垂直筆劃；2 頂部向右彎弧、底部有水平橫底。\\n"\r\n'
    '                                     "  · 3 vs 8：3 左側開口（如兩個 C 疊加）；8 是完全閉合的雙圓。\\n"\r\n'
    '                                     "  · 5 vs S：5 頂部有平直橫線；S 兩端均為弧線，無平直頂。\\n"\r\n'
    '                                     "  · 2 vs Z：2 頂部為弧形；Z 兩端皆為銳角直線，整體有稜角感。\\n"\r\n'
    '                                     "  · 8 vs B：8 是數字（對稱閉合雙圓）；B 是字母（左側有垂直直線）。\\n"\r\n'
    '                                     "  · 6 vs G：6 底部有封閉圓圈；G 是字母（右側內部有向左延伸橫線）。\\n"\r\n'
    '                                     "【主筋計數驗算】若根數恰好落在混淆對附近，請再數一次！\\n"\r\n'
    '                                     "  · 底部有封閉圓圈 → 極可能是 6（非 5）\\n"\r\n'
    '                                     "  · 右側有兩條橫線 → 極可能是 12（非 11）\\n"\r\n'
    '                                     "  · 台灣常用鋼筋號數：#3~#11（超過 #11 極罕見）。\\n"\r\n'
)

VISUAL_WARN_RETRY = (
    '                                             "【⚠️ 數字視覺混淆警告 — 本次重試請特別注意！】\\n"\r\n'
    '                                             "下列高混淆配對是誤判主因，請再看一眼後再填入：\\n"\r\n'
    '                                             "  · 5 vs 6：5 頂部平直橫線；6 底部有封閉圓圈。\\n"\r\n'
    '                                             "  · 1 vs 7：1 無橫線；7 頂端有水平橫線。\\n"\r\n'
    '                                             "  · 1 vs 2：1 只有垂直線；2 底部有水平橫底。\\n"\r\n'
    '                                             "  · 0 vs 8：0 是單圓；8 是雙圓中間有交叉。\\n"\r\n'
    '                                             "  · 3 vs 8：3 左側開口；8 完全閉合。\\n"\r\n'
    '                                             "  · 6 vs G：6 底有圓圈；G 右側內有橫線。\\n"\r\n'
    '                                             "  · 8 vs B：8 對稱雙圓；B 左側有垂直直線。\\n"\r\n'
    '                                             "【計數再確認】底部有圓→可能 6 非 5；右側兩橫→可能 12 非 11。\\n"\r\n'
)

with open(TARGET, encoding="utf-8") as f:
    content = f.read()
    lines = content.splitlines(keepends=True)

# Step 1: 移除所有含 \\u 字面的廢行（上次錯插的 unicode escape 字串）
before = len(lines)
lines = [l for l in lines if '\\\\u' not in l]
after = len(lines)
print(f"[清理] 移除 {before - after} 行廢棄的 unicode-escape 字串")

# Step 2: 找「請直接輸出 JSON 格式的 BeamList 資料」這行 → 在它前面插入主警告
content_tmp = "".join(lines)
ANCHOR_MAIN = '"請直接輸出 JSON 格式的 BeamList 資料。"'

if ANCHOR_MAIN in content_tmp:
    content_tmp = content_tmp.replace(
        ANCHOR_MAIN,
        VISUAL_WARN_MAIN + '                                     ' + ANCHOR_MAIN,
        1  # 只替換第一個出現
    )
    print("[OK] 主 prompt 視覺警告插入成功")
else:
    print("[FAIL] 主 prompt anchor 未找到")

# Step 3: 找 retry 的 anchor：clean_rule_beam + json.dumps 那一行
ANCHOR_RETRY = 'f"【已確定資料 (請保留，並填補遺失處)】:\\n{json.dumps(clean_rule_beam'
if ANCHOR_RETRY in content_tmp:
    content_tmp = content_tmp.replace(
        ANCHOR_RETRY,
        VISUAL_WARN_RETRY + '                                             ' + ANCHOR_RETRY,
        1
    )
    print("[OK] retry prompt 視覺警告插入成功")
else:
    print("[FAIL] retry prompt anchor 未找到，嘗試備用搜尋...")
    # 備用：找含 clean_rule_beam 且含 indent=2 的行
    ANCHOR_RETRY2 = "indent=2)}"
    idx = content_tmp.find(ANCHOR_RETRY2)
    if idx != -1:
        # 往回找行首
        line_start = content_tmp.rfind("\n", 0, idx) + 1
        the_line = content_tmp[line_start:content_tmp.find("\n", idx)+1]
        content_tmp = content_tmp[:line_start] + VISUAL_WARN_RETRY + the_line + content_tmp[line_start + len(the_line):]
        print("[OK] retry prompt 用備用 anchor 插入成功")
    else:
        print("[FAIL] 備用 anchor 也未找到")

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content_tmp)

print("Patch v3 complete.")
