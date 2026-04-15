"""
OCR Field Assigner — 規則引擎
==============================
根據 OCR 偵測到的文字格式 + 九宮格位置，直接分配到 BeamDetail JSON 欄位。
高信心度的項目直接填入，低信心度的收集起來交給 LLM 補位。
"""
import re
from enum import Enum
from typing import List, Dict, Tuple, Optional
from core.normalizer import normalize_text


# ================================================================
# 格式分類
# ================================================================
class FormatType(Enum):
    REBAR = "rebar"           # 主筋: 5-#8, 3-#10
    STIRRUP = "stirrup"       # 箍筋: 13-#4@15, #4@20
    LAP_LENGTH = "lap_length" # 搭接長度: 230, 150
    BEAM_ID = "beam_id"       # 梁編號: B3F G1-2
    DIMENSION = "dimension"   # 尺寸: 50x70
    FACE_BAR = "face_bar"     # 腰筋: 12-#5 (E.F.)
    UNKNOWN = "unknown"       # 無法辨識


# 主筋: N-#S 格式，不含 @，只接受 #3 發展至 #18，支援逗號組合
RE_REBAR = re.compile(r'^\d+-#(?:[3-9]|1[0-8])(?:\s*,\s*\d+-#(?:[3-9]|1[0-8]))*$')
# 箍筋: 含 @ (如 13-#4@15, #4@20)，鋼筋號數限 #3~#18，間距限間距 7~30
RE_STIRRUP = re.compile(r'^(\d+-)?#(?:[3-9]|1[0-8])@(?:[7-9]|[1-2]\d|30)$')
# 尺寸: NxM
RE_DIMENSION = re.compile(r'^\d+[xX×*]\d+$')
# 搭接長度: 2~3 位數純數字
RE_LAP_LENGTH = re.compile(r'^\d{2,3}$')
# 腰筋: 含 E.F.
RE_FACE_BAR = re.compile(r'[Ee]\.?[Ff]\.?')

# 梁前綴 (與 table_extractor._is_beam_id 一致)
RE_BEAM_PREFIX = re.compile(
    r'^(F?W?[BGCS]|FB|FG|RB|CB|SB|[BR]\d+F)\s*',
    re.IGNORECASE
)


def classify_text(text: str) -> FormatType:
    """根據文字格式分類成工程標註類型"""
    # 移除頭尾非括號雜訊
    s = text.strip().strip('-_.= \u2014\u2013~')
    # 成對括號才刪除：左括號類 + 右括號類任意配對（容許 OCR 混淆 ( 和 （）
    _LEFT_BRACKETS  = set('(（[【')
    _RIGHT_BRACKETS = set(')）]】')
    while len(s) >= 2 and s[0] in _LEFT_BRACKETS and s[-1] in _RIGHT_BRACKETS:
        s = s[1:-1].strip()
    if not s or len(s) < 1:
        return FormatType.UNKNOWN
    
    # 建立無空白版本供嚴格的正則比對，消除 OCR 內部隨機空白干擾
    s_test = s.replace(" ", "")
    
    # 含 @ → 一定是箍筋
    if '@' in s_test and RE_STIRRUP.match(s_test):
        return FormatType.STIRRUP
    
    # 含 # 但不含 @ → 主筋
    if '#' in s_test and RE_REBAR.match(s_test):
        return FormatType.REBAR
    
    # 腰筋 (含 E.F.)
    if RE_FACE_BAR.search(s_test):
        return FormatType.FACE_BAR
        
    # 九宮格角落/純數字 → 高機率是搭接長度
    if RE_LAP_LENGTH.match(s_test):
        return FormatType.LAP_LENGTH
        
    # 尺寸格式 NxM
    if RE_DIMENSION.match(s_test):
        return FormatType.DIMENSION
    
    # 梁編號
    if _is_beam_id(s):
        return FormatType.BEAM_ID
    
    return FormatType.UNKNOWN


def _is_beam_id(text: str) -> bool:
    """嚴格辨識梁編號 (與 TableExtractor._is_beam_id 一致)"""
    text = text.strip()
    if len(text) < 2:
        return False
    if '@' in text or '#' in text:
        return False
    # 純數字 / 單字母+純數字 (如 195, F11-11, E12) → 不是梁編號，是軸線或尺寸
    if re.match(r'^[A-Za-z]?\d[\d\-]*$', text):
        return False
    if RE_BEAM_PREFIX.match(text):
        return True
    return False


# ================================================================
# 位置→欄位映射表
# ================================================================
FIELD_MAP = {
    # 主筋
    (FormatType.REBAR, "左上方"): "top_main_bars_left",
    (FormatType.REBAR, "正上方"): "top_main_bars_mid",
    (FormatType.REBAR, "右上方"): "top_main_bars_right",
    (FormatType.REBAR, "左下方"): "bottom_main_bars_left",
    (FormatType.REBAR, "正下方"): "bottom_main_bars_mid",
    (FormatType.REBAR, "右下方"): "bottom_main_bars_right",

    # 箍筋 — 主要位置
    (FormatType.STIRRUP, "正左方"): "stirrups_left",
    (FormatType.STIRRUP, "正中央"): "stirrups_middle",
    (FormatType.STIRRUP, "正右方"): "stirrups_right",
    # 箍筋 — fallback (有時箍筋標註偏上/下)
    (FormatType.STIRRUP, "左上方"): "stirrups_left",
    (FormatType.STIRRUP, "左下方"): "stirrups_left",
    (FormatType.STIRRUP, "右上方"): "stirrups_right",
    (FormatType.STIRRUP, "右下方"): "stirrups_right",
    (FormatType.STIRRUP, "正上方"): "stirrups_middle",
    (FormatType.STIRRUP, "正下方"): "stirrups_middle",

    # 搭接長度 — 嚴格依據上下位置
    (FormatType.LAP_LENGTH, "左上方"): "lap_length_top_left",
    (FormatType.LAP_LENGTH, "右上方"): "lap_length_top_right",
    (FormatType.LAP_LENGTH, "左下方"): "lap_length_bottom_left",
    (FormatType.LAP_LENGTH, "右下方"): "lap_length_bottom_right",
    # 搭接長度在正上/正下方的 fallback (少見但可能)
    (FormatType.LAP_LENGTH, "正上方"): "lap_length_top_left",
    (FormatType.LAP_LENGTH, "正下方"): "lap_length_bottom_left",
}

# 若格式未知 (或無對應映射)，強制採用空間位置預設欄位 (用於 Fallback)
POS_TO_DEFAULT_FIELD = {
    "左上方": "top_main_bars_left",
    "正上方": "top_main_bars_mid",
    "右上方": "top_main_bars_right",
    "正左方": "stirrups_left",
    "正中央": "stirrups_middle",
    "正右方": "stirrups_right",
    "左下方": "bottom_main_bars_left",
    "正下方": "bottom_main_bars_mid",
    "右下方": "bottom_main_bars_right"
}

# 陣列欄位 (多筆同位置要合併成 list)
LIST_FIELDS = {
    "top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
    "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right"
}

# 信心度門檻
HIGH_CONF_THRESHOLD = 0.95


# ================================================================
# 主函數
# ================================================================
def assign_fields(ocr_items: list, ctx) -> Tuple[dict, list]:
    """
    規則引擎主函數。
    
    Args:
        ocr_items: OCR 結果列表 (每筆含 text, conf, cx, cy, pos_label 等)
        ctx: CropContext (含 grid 邊界資訊)
    
    Returns:
        (beam_dict, low_conf_items)
        - beam_dict: 高信心度已填好的欄位 dict
        - low_conf_items: 需要 LLM 補位的低信心度 OCR 項目列表
    """
    # 初始化空的 beam dict
    beam = {
        "beam_id": "",
        "dimensions": "",
        "top_main_bars_left": [],
        "top_main_bars_mid": [],
        "top_main_bars_right": [],
        "bottom_main_bars_left": [],
        "bottom_main_bars_mid": [],
        "bottom_main_bars_right": [],
        "stirrups_left": "",
        "stirrups_middle": "",
        "stirrups_right": "",
        "face_bars": "",
        "lap_length_top_left": "",
        "lap_length_top_right": "",
        "lap_length_bottom_left": "",
        "lap_length_bottom_right": "",
        "self_confidence": 0,
        "note": ""
    }
    
    low_conf_items = []
    assignments = []  # 用於 debug log
    borderline_records = [] # 暫存 90% ~ 95% 的邊緣信心項目
    
    # ==== Pre-scan: 找出有低信心項目的方位，同方位的高信心項目也要送 LLM ====
    # 如果同一個九宮格方位中有任何項目信心<95%，代表那個位置的圖面有不確定性
    # 此時即使該方位有其他 96%+ 的項目，也不應自動填入，而是全部送 LLM 看完整圖面後判斷
    contested_positions = set()
    for item in ocr_items:
        pos = _extract_core_position(item.get("pos_label", ""))
        conf = item.get("conf", 0.0)
        if conf < HIGH_CONF_THRESHOLD:
            contested_positions.add(pos)
    if contested_positions:
        print(f"  [Pre-scan] 以下方位存在低信心項目，將全數送 LLM: {contested_positions}")
    
    for item in ocr_items:
        text = item["text"].strip()
        conf = item.get("conf", 0.0)
        pos_label = item.get("pos_label", "")
        
        # 去除位置描述中的「極端邊緣」標記，保留核心方位
        core_pos = _extract_core_position(pos_label)
        
        # Step 1: 格式分類
        fmt = classify_text(text)
        
        # Step 2: 位置→欄位映射
        field_key = FIELD_MAP.get((fmt, core_pos))
        
        # Step 3: 特殊處理
        if fmt == FormatType.BEAM_ID:
            normalized = normalize_text(text)
            # 梁名(與黏著的尺寸)：只要位置在左下、正下、右下，就直接接受
            if core_pos in ["左下方", "正下方", "右下方"]:
                if conf >= 0.90:
                    if not beam["beam_id"]:
                        beam["beam_id"] = normalized
                        assignments.append(f"✅ beam_id = '{normalized}' (conf={conf:.0%})")
                        
                    # 順便提取可能黏在一起的尺寸 (例如 "B3F B5-5 (50x70)")
                    dim_match = re.search(r'(\d+\s*[xX×*]\s*\d+)', text)
                    if dim_match and not beam["dimensions"]:
                        beam["dimensions"] = normalize_text(dim_match.group(1))
                        assignments.append(f"✅ dimensions = '{beam['dimensions']}' (從 beam_id 拆解提取)")
                else:
                    low_conf_items.append(item)
                    assignments.append(f"⚠️ beam_id 信心不足 <90%: '{text}' (conf={conf:.0%}) → LLM")
            else:
                default_field = POS_TO_DEFAULT_FIELD.get(core_pos)
                if default_field:
                    item["predicted_field"] = default_field
                    item["fallback_suffix"] = "(未匹配格式)"
                low_conf_items.append(item)
                assignments.append(f"⚠️ beam_id 位置不在邊線: '{text}' @ {core_pos} → LLM")
                
            continue
        
        if fmt == FormatType.DIMENSION:
            normalized = normalize_text(text)
            # 梁尺寸同理，只要在下方就接受
            if core_pos in ["左下方", "正下方", "右下方"]:
                if conf >= 0.90:
                    beam["dimensions"] = normalized
                    assignments.append(f"✅ dimensions = '{normalized}' (conf={conf:.0%})")
                else:
                    low_conf_items.append(item)
                    assignments.append(f"⚠️ dimensions 信心不足 <90%: '{text}' (conf={conf:.0%}) → LLM")
            else:
                default_field = POS_TO_DEFAULT_FIELD.get(core_pos)
                if default_field:
                    item["predicted_field"] = default_field
                    item["fallback_suffix"] = "(未匹配格式)"
                low_conf_items.append(item)
                assignments.append(f"⚠️ dimensions 位置異常: '{text}' @ {core_pos} → LLM")
                
            continue
        
        if fmt == FormatType.FACE_BAR:
            normalized = normalize_text(text)
            if conf >= 0.85:
                beam["face_bars"] = normalized
                assignments.append(f"✅ face_bars = '{normalized}' (conf={conf:.0%})")
            elif conf >= 0.80:
                borderline_records.append((item, "face_bars", normalized, conf))
            else:
                item["predicted_field"] = "face_bars"
                low_conf_items.append(item)
                assignments.append(f"⚠️ face_bars 信心極弱 <80%: '{text}' (conf={conf:.0%}) → LLM")
            continue
        
        if fmt == FormatType.LAP_LENGTH:
            if "上" not in core_pos and "下" not in core_pos:
                default_field = POS_TO_DEFAULT_FIELD.get(core_pos)
                if default_field:
                    item["predicted_field"] = default_field
                    item["fallback_suffix"] = "(未匹配格式)"
                low_conf_items.append(item)
                assignments.append(f"⚠️ 疑似搭接長度但位置異常: '{text}' @ {core_pos} → LLM")
                continue
        
        # Step 4: 通用映射
        if field_key:
            # 如果這個方位被標記為「有爭議」，則無視信心度，全部送 LLM
            if core_pos in contested_positions and field_key in LIST_FIELDS:
                item["predicted_field"] = field_key
                low_conf_items.append(item)
                assignments.append(f"⚠️ {field_key} 同方位有低信心項目，全部送 LLM: '{text}' (conf={conf:.0%})")
            elif conf >= 0.95:
                if field_key in LIST_FIELDS:
                    for part in text.split(","):
                        beam[field_key].append(part.strip())
                    assignments.append(f"✅ {field_key} += '{text}' (conf={conf:.0%})")
                else:
                    if not beam.get(field_key):
                        beam[field_key] = text
                        assignments.append(f"✅ {field_key} = '{text}' (conf={conf:.0%})")
                    else:
                        assignments.append(f"⚠️ {field_key} 已有值，略過重複物件 '{text}' (conf={conf:.0%})")
            elif conf >= 0.90:
                # 信心落在 90~95%：暫存到邊緣區，等待最後裁決
                borderline_records.append((item, field_key, text, conf))
            else:
                # 信心低於 90%，強制要求 LLM 檢查
                item["predicted_field"] = field_key
                low_conf_items.append(item)
                assignments.append(f"⚠️ {field_key} 信心不足 <90%: '{text}' (conf={conf:.0%}) → LLM")
        elif fmt == FormatType.UNKNOWN:
            default_field = POS_TO_DEFAULT_FIELD.get(core_pos)
            if default_field:
                item["predicted_field"] = default_field
                item["fallback_suffix"] = "(未匹配格式)"
            low_conf_items.append(item)
            assignments.append(f"❌ 無法分類: '{text}' @ {core_pos} (conf={conf:.0%}) → LLM")
        else:
            # 有格式但沒有對應的映射位置
            default_field = POS_TO_DEFAULT_FIELD.get(core_pos)
            if default_field:
                item["predicted_field"] = default_field
                item["fallback_suffix"] = "(未匹配格式)"
            low_conf_items.append(item)
            assignments.append(f"⚠️ 有格式({fmt.value})但位置({core_pos})無映射 → LLM")
            
    # ==== 結算 LLM 觸發邏輯 ====
    # 如果已經有低於 90% 的渣渣，或者沒抓到梁名，代表 LLM 「絕對得出門上班」
    must_call_llm = len(low_conf_items) > 0 or not beam["beam_id"]
    
    if must_call_llm:
        # LLM 既然都要加班了，那就把 90~95% 安全區間的卡牌也全部丟給他順便檢查！
        for item, fk, t, c in borderline_records:
            item["predicted_field"] = fk
            low_conf_items.append(item)
            assignments.append(f"⚠️ {fk} 落入邊緣檢查區間 (90~95%): '{t}' (conf={c:.0%}) → LLM 順便檢查")
    else:
        # 大家都表現得很好 (>90%)，LLM 可以放假！
        # 直接把 90~95% 邊緣区的手工寫入正式欄位
        for item, field_key, text, conf in borderline_records:
            if field_key in LIST_FIELDS:
                for part in text.split(","):
                    beam[field_key].append(part.strip())
                assignments.append(f"✅ {field_key} += '{text}' (沒叫LLM，直接放行 conf={conf:.0%})")
            else:
                beam[field_key] = text
                assignments.append(f"✅ {field_key} = '{text}' (沒叫LLM，直接放行 conf={conf:.0%})")
    
    # 計算信心分數
    total_items = len(ocr_items)
    high_conf_count = total_items - len(low_conf_items)
    beam["self_confidence"] = int((high_conf_count / max(total_items, 1)) * 100)
    
    # 輸出 debug log
    print(f"[規則引擎] 共 {total_items} 筆 OCR → {high_conf_count} 筆免 LLM 直填, {len(low_conf_items)} 筆交給 LLM 確認")
    for a in assignments:
        print(f"  {a}")
    
    return beam, low_conf_items


def _extract_core_position(pos_label: str) -> str:
    """從含有邊緣標記的位置字串中提取核心方位
    例: '右上方 (極右邊緣)' → '右上方'
    """
    # 去掉括號內容
    core = re.sub(r'\s*\(.*?\)\s*', '', pos_label).strip()
    return core if core else pos_label
