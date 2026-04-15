import re
from core.debug_logger import debug_print

print = debug_print

def _sum_rebars(rebar_list):
    """計算鋼筋陣列的總根數，例如 ['5-#8', '2-#8'] -> 7"""
    total = 0
    if not isinstance(rebar_list, list):
        rebar_list = [rebar_list]
    for item in rebar_list:
        m = re.match(r'^(\d+)-', str(item))
        if m:
            total += int(m.group(1))
    return total

def apply_structural_rules(beam_dict: dict) -> dict:
    """
    執行結構工程的業務邏輯後處理 (Business Logic Inference)。
    在 LLM JSON 輸出並經過基礎 normalizer 後執行。
    """
    if not isinstance(beam_dict, dict):
        return beam_dict
        
    b = dict(beam_dict)  # 複製一份避免污染原始資料
    
    # === 規則 0: 格式防護 — 主筋欄位絕不能含有箍筋值 (@符號) ===
    main_bar_keys = [
        "top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
        "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right"
    ]
    for mk in main_bar_keys:
        vals = b.get(mk, [])
        if isinstance(vals, list):
            cleaned = [v for v in vals if '@' not in str(v)]
            if len(cleaned) != len(vals):
                print(f"[後處理] ⚠️ 格式防護：{mk} 中發現箍筋值 {[v for v in vals if '@' in str(v)]}，已移除")
                b[mk] = cleaned
    
    def process_main_bars(b_dict, k_l, k_m, k_r, is_top):
        v_l = b_dict.get(k_l, [])
        v_m = b_dict.get(k_m, [])
        v_r = b_dict.get(k_r, [])
        
        has_l = bool(v_l and v_l != [""])
        has_m = bool(v_m and v_m != [""])
        has_r = bool(v_r and v_r != [""])
        
        counts = sum([has_l, has_m, has_r])
        
        if counts == 1:
            val = v_l if has_l else (v_m if has_m else v_r)
            b_dict[k_l] = b_dict[k_m] = b_dict[k_r] = val
        elif counts == 2:
            if has_l and has_m and not has_r:
                b_dict[k_r] = v_m
            elif not has_l and has_m and has_r:
                b_dict[k_l] = v_m
            elif has_l and not has_m and has_r:
                if v_l == v_r:
                    b_dict[k_m] = v_l
                else:
                    sum_l = _sum_rebars(v_l)
                    sum_r = _sum_rebars(v_r)
                    if is_top:
                        # 上層筋：中間抄小的
                        b_dict[k_m] = v_l if sum_l <= sum_r else v_r
                    else:
                        # 下層筋：中間抄大的
                        b_dict[k_m] = v_l if sum_l >= sum_r else v_r

    # 處理上下層主筋填補推論
    process_main_bars(b, "top_main_bars_left", "top_main_bars_mid", "top_main_bars_right", is_top=True)
    process_main_bars(b, "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right", is_top=False)

    # 規則 3: 箍筋若只標示 middle(或唯一有值)，視為全段相同
    stirrup_keys = ["stirrups_left", "stirrups_middle", "stirrups_right"]
    stirrup_vals = [b.get(k, "") for k in stirrup_keys]
    non_empty_stirrup = [v for v in stirrup_vals if v.strip()]
    
    if len(non_empty_stirrup) == 1:
        val = non_empty_stirrup[0]
        for k in stirrup_keys:
            b[k] = val

    # === 規則 4: lap_length 格式防護 — 必須是 2~3 位純數字 ===
    # 作為最終防線：清除所有非數字的搭接長度值（如 "LL", "N/A", "LLcm" 等）
    LAP_LENGTH_KEYS = [
        "lap_length_top_left", "lap_length_top_right",
        "lap_length_bottom_left", "lap_length_bottom_right"
    ]
    for lk in LAP_LENGTH_KEYS:
        v = b.get(lk, "")
        if not isinstance(v, str) or not v:
            continue
        # 去掉常見單位後驗證
        cleaned = v.strip().replace("cm", "").replace("CM", "").replace("Cm", "").strip()
        if not re.match(r'^\d{2,3}$', cleaned):
            print(f"[後處理] ⚠️ 格式防護：{lk} 值 '{v}' 不是純數字，已清空")
            b[lk] = ""
        elif cleaned != v.strip():
            # 去掉單位後是合法數字，保留乾淨版本
            b[lk] = cleaned

    return b

def is_empty_beam(beam_dict: dict) -> bool:
    """判斷此筆資料是否為空梁（只有梁名稱和 note，其他欄位都沒東西）。"""
    if not isinstance(beam_dict, dict):
        return False
    
    # 這些欄位不算「有實質內容」
    skip_keys = {"beam_id", "note", "crop_index", "_ocr_text", "_crop_file", 
                 "_debug_lines", "self_confidence", "alignment_status",
                 "spatial_anchor_rect_x_y", "span_group", "_raw_span_idx", "span_order"}
    
    _empty_markers = {"LLM沒有東西", "LLM看不出來", ""}
    
    for k, v in beam_dict.items():
        if k in skip_keys:
            continue
        if isinstance(v, list):
            real = [x for x in v if str(x).strip() not in _empty_markers]
            if real:
                return False
        elif isinstance(v, str) and v.strip() not in _empty_markers:
            return False
    
    return True
