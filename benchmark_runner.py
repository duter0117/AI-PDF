"""AI-PDF Benchmark Runner
========================
自動化正確率評測 + HTML 視覺化比對報告。

使用方式：
  # 1. 先讓 AI 預跑一次，自動產生 ground truth 草稿
  python benchmark_runner.py --init

  # 2. 打開產生的 JSON 檔案，修正錯誤的值
  # 3. 跑正式評測
  python benchmark_runner.py

可選參數:
  --init                預跑模式：對 benchmarks/ 裡尚無配對 JSON 的 PDF 跑 pipeline，
                        自動產生 ground truth 草稿供人工校正
  --init-all            對所有 PDF 重新產生 ground truth (覆蓋既有)
  --filter test1        只跑名稱含 test1 的測試
  --voting 2            Self-Consistency 投票輪數
  --verbose             終端機顯示每個欄位比對細節
  --no-open             不自動開啟瀏覽器
"""
import os
import sys
import json
import glob
import time
import asyncio
import argparse
import re
import html as html_mod
import webbrowser
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from workflow.drawing_agent import build_graph

# ================================================================
# 文字正規化
# ================================================================
from core.normalizer import normalize_text, normalize_list

# ================================================================
# 欄位定義與權重
# ================================================================
FIELD_WEIGHTS = {
    "beam_id": 10, "dimensions": 8,
    "top_main_bars_left": 5, "top_main_bars_mid": 5, "top_main_bars_right": 5,
    "bottom_main_bars_left": 5, "bottom_main_bars_mid": 5, "bottom_main_bars_right": 5,
    "stirrups_left": 4, "stirrups_middle": 4, "stirrups_right": 4,
    "face_bars": 3,
    "lap_length_top_left": 2, "lap_length_top_right": 2,
    "lap_length_bottom_left": 2, "lap_length_bottom_right": 2,
}
COMPARE_FIELDS = list(FIELD_WEIGHTS.keys())
LIST_FIELDS = {"top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
               "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right"}

# ================================================================
# 欄位比對
# ================================================================
def analyze_set(exp_dict, pred_dict, fields, is_list=False):
    missing, wrong, misplaced, halluc, expected_cnt = 0, 0, 0, 0, 0
    exp_all, pred_all = [], []
    for f in fields:
        if is_list:
            e_ls = normalize_list((exp_dict or {}).get(f) if exp_dict else [])
            p_ls = normalize_list((pred_dict or {}).get(f) if pred_dict else [])
            expected_cnt += len(e_ls)
            exp_all.extend(e_ls)
            pred_all.extend(p_ls)
        else:
            e_val = normalize_text((exp_dict or {}).get(f) if exp_dict else "")
            p_val = normalize_text((pred_dict or {}).get(f) if pred_dict else "")
            if e_val:
                expected_cnt += 1
                exp_all.append(e_val)
            if p_val:
                pred_all.append(p_val)
                
    if not exp_all and not pred_all: return 0,0,0,0,0
    if not exp_all and pred_all: return 0,0,0,len(pred_all),0
    if exp_all and not pred_all: return len(exp_all),0,0,0,expected_cnt
        
    if sorted(exp_all) == sorted(pred_all):
        correct = 0
        for f in fields:
            if is_list:
                e_ls = normalize_list((exp_dict or {}).get(f) if exp_dict else [])
                p_ls = normalize_list((pred_dict or {}).get(f) if pred_dict else [])
                c = 0
                for item in e_ls:
                    if item in p_ls:
                        c += 1; p_ls.remove(item)
                correct += c
            else:
                e_val = normalize_text((exp_dict or {}).get(f) if exp_dict else "")
                p_val = normalize_text((pred_dict or {}).get(f) if pred_dict else "")
                if e_val and p_val and (e_val == p_val or e_val in p_val or p_val in e_val):
                    correct += 1
        return 0,0,expected_cnt - correct,0,expected_cnt
        
    for f in fields:
        if is_list:
            e_ls = normalize_list((exp_dict or {}).get(f) if exp_dict else [])
            p_ls = normalize_list((pred_dict or {}).get(f) if pred_dict else [])
            if not e_ls and p_ls: halluc += len(p_ls)
            elif e_ls and not p_ls: missing += len(e_ls)
            else:
                temp_p = p_ls.copy()
                for item in e_ls:
                    if item in temp_p: temp_p.remove(item)
                    else: wrong += 1
                halluc += len(temp_p)
        else:
            e_val = normalize_text((exp_dict or {}).get(f) if exp_dict else "")
            p_val = normalize_text((pred_dict or {}).get(f) if pred_dict else "")
            if e_val and not p_val: missing += 1
            elif not e_val and p_val: halluc += 1
            elif e_val and p_val:
                if not (e_val == p_val or e_val in p_val or p_val in e_val): wrong += 1
    return missing, wrong, 0, halluc, expected_cnt

def compare_field(field_name: str, expected, predicted) -> Tuple[float, str]:
    if field_name in LIST_FIELDS:
        exp_list = normalize_list(expected if isinstance(expected, list) else [])
        pred_list = normalize_list(predicted if isinstance(predicted, list) else [])
        if not exp_list and not pred_list:
            return 1.0, "both empty"
        if not exp_list and pred_list:
            return 0.0, f"expected empty, got {pred_list}"
        if exp_list and not pred_list:
            return 0.0, f"expected {exp_list}, got empty"
        matches = sum(1 for i in range(max(len(exp_list), len(pred_list)))
                      if i < len(exp_list) and i < len(pred_list) and exp_list[i] == pred_list[i])
        if matches == max(len(exp_list), len(pred_list)):
            return 1.0, "element-wise"
        # 如果逐位比對失敗，嘗試集合比 (忽略順序)
        set_matches = len(set(exp_list) & set(pred_list))
        set_score = set_matches / max(len(set(exp_list) | set(pred_list)), 1)
        return max(matches / max(len(exp_list), len(pred_list)), set_score), "element-wise"
    else:
        exp_str = normalize_text(str(expected) if expected else "")
        pred_str = normalize_text(str(predicted) if predicted else "")
        if exp_str == pred_str:
            return 1.0, "exact"
        if exp_str and pred_str and (exp_str in pred_str or pred_str in exp_str):
            return 0.5, "partial"
        return 0.0, "mismatch"

# ================================================================
# Beam 配對
# ================================================================
def match_beams(expected_beams, predicted_beams):
    def _clean_id(raw_id):
        """正規化 beam_id 並剝掉重名後綴 (重複-X)"""
        s = normalize_text(raw_id)
        s = re.sub(r'\s*\(重複-\d+\)\s*$', '', s).strip()
        return s
    
    matched = []
    used_pred = set()
    for exp in expected_beams:
        exp_id = _clean_id(exp.get("beam_id", ""))
        for j, pred in enumerate(predicted_beams):
            if j in used_pred: continue
            pred_id = _clean_id(pred.get("beam_id", ""))
            if exp_id and pred_id and exp_id == pred_id:
                matched.append((exp, pred))
                used_pred.add(j)
                break
    unmatched_exp = [e for e in expected_beams if not any(e is m[0] for m in matched)]
    for exp in unmatched_exp:
        exp_id = _clean_id(exp.get("beam_id", ""))
        if not exp_id: continue
        best_j, best_score = None, 0
        for j, pred in enumerate(predicted_beams):
            if j in used_pred: continue
            pred_id = _clean_id(pred.get("beam_id", ""))
            if not pred_id: continue
            if exp_id in pred_id or pred_id in exp_id:
                score = min(len(exp_id), len(pred_id)) / max(len(exp_id), len(pred_id))
                if score > best_score: best_score, best_j = score, j
        if best_j is not None and best_score > 0.4:
            matched.append((exp, predicted_beams[best_j]))
            used_pred.add(best_j)
    unmatched_exp = [e for e in expected_beams if not any(e is m[0] for m in matched)]
    for exp in unmatched_exp:
        matched.append((exp, None))
    for j in range(len(predicted_beams)):
        if j not in used_pred:
            matched.append((None, predicted_beams[j]))
    return matched

# ================================================================
# 單 PDF 評測
# ================================================================
async def evaluate_single(pdf_path, ground_truth, graph) -> dict:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    page_num = ground_truth.get("page_num", 0)
    cv_params = ground_truth.get("cv_params", {})

    t0 = time.time()
    result = await graph.ainvoke({
        "pdf_bytes": pdf_bytes, "page_num": page_num,
        "task_id": None, "cv_params": cv_params
    })
    elapsed = round(time.time() - t0, 1)

    final = result.get("final_output", {})
    aligned = final.get("aligned_beams", [])
    api_metrics = final.get("api_metrics", {})
    for b in aligned: b["_source"] = "split"
    from core.post_processor import apply_structural_rules
    expected_beams = [apply_structural_rules(b) for b in ground_truth.get("beams", [])]

    # === 建立 expected beam 的 ID 查詢表 ===
    exp_by_ids = defaultdict(list)
    for exp in expected_beams:
        eid = normalize_text(exp.get("beam_id", ""))
        if eid:
            exp_by_ids[eid].append(exp)

    # === 統一合併配對 (不區分母子了) ===
    # 為避免漏抓，我們使用模糊配對機制
    aligned_pairs = []
    unmatched_exp = expected_beams.copy()
    unmatched_pred = []
    
    # 追蹤已被精確匹配消耗的 expected (每筆最多配一次)
    # 但重名預測 (如 重複-1, 重複-2) 可以共享同一筆 expected
    used_exp_set = set()  # 使用 id() 追蹤
    
    # Pass 1: 基於 beam_id 的精確與模糊配對
    for pred in aligned:
        pred_id = normalize_text(pred.get("beam_id", ""))
        matched_exp = None
        
        if pred_id and pred_id in exp_by_ids:
            # 優先找尚未被配對過的 expected
            for exp in exp_by_ids[pred_id]:
                if id(exp) not in used_exp_set:
                    matched_exp = exp
                    break
            # 如果全部都已被配對過（重名情境），允許重用第一筆
            if not matched_exp:
                for exp in exp_by_ids[pred_id]:
                    matched_exp = exp
                    break
            
        # 如果精確比對失敗，且有值，嘗試模糊比對
        if not matched_exp and pred_id:
            for eid, exps in exp_by_ids.items():
                if (pred_id in eid or eid in pred_id):
                    ratio = min(len(pred_id), len(eid)) / max(len(pred_id), len(eid))
                    if ratio > 0.4:
                        for exp in exps:
                            if id(exp) not in used_exp_set:
                                matched_exp = exp
                                break
                if matched_exp:
                    break
                        
        if matched_exp:
            aligned_pairs.append((matched_exp, pred))
            used_exp_set.add(id(matched_exp))
            if matched_exp in unmatched_exp:
                unmatched_exp.remove(matched_exp)
        else:
            unmatched_pred.append(pred)

    # Pass 2: 內容相似度配對 (Content-based Fallback)
    # 用於解決「整支梁未偵測到」+「幻覺」，但其實內容極度吻合，只是梁名稱辨識失敗的情況
    if unmatched_exp and unmatched_pred:
        def generate_content_signature(beam):
            sig = []
            for k in COMPARE_FIELDS:
                if k == "beam_id": continue
                v = beam.get(k, "")
                if k in LIST_FIELDS:
                    norm = normalize_list(v)
                    for x in norm:
                        if x: sig.append(f"{k}:{x}")
                else:
                    norm = normalize_text(str(v))
                    if norm: sig.append(f"{k}:{norm}")
            return set(sig)

        for i in range(len(unmatched_exp) - 1, -1, -1):
            if not unmatched_pred: break
            e_beam = unmatched_exp[i]
            e_sig = generate_content_signature(e_beam)
            if not e_sig: continue
            
            best_p_idx = -1
            best_score = 0
            for j, p_beam in enumerate(unmatched_pred):
                p_sig = generate_content_signature(p_beam)
                common = len(e_sig.intersection(p_sig))
                total = len(e_sig.union(p_sig))
                if total == 0: continue
                # Dice coefficient
                score = (2.0 * common) / (len(e_sig) + len(p_sig))
                
                # 只有當相似度 > 0.5 且至少有 3 個相同具體欄位時才配對
                if score > best_score and score > 0.5 and common >= 3:
                    best_score = score
                    best_p_idx = j
                    
            if best_p_idx != -1:
                aligned_pairs.append((e_beam, unmatched_pred[best_p_idx]))
                unmatched_pred.pop(best_p_idx)
                unmatched_exp.pop(i)
            
    # 將遺漏與剩餘的幻覺補齊
    for pred in unmatched_pred:
        aligned_pairs.append((None, pred))
    for exp in unmatched_exp:
        aligned_pairs.append((exp, None))

    # === 計算結果 ===
    def _score_pair(exp, pred, source):
        def add_t(t1, t2): return tuple(a + b for a, b in zip(t1, t2))
        
        if exp is None:
            # 幻覺梁：只扣分梁名，內容欄位不重複扣分
            m_b = analyze_set(exp, pred, ["beam_id"], False)
            m_main = (0, 0, 0, 0, 0)
            m_stirrup = (0, 0, 0, 0, 0)
            m_face = (0, 0, 0, 0, 0)
            m_lap = (0, 0, 0, 0, 0)
        else:
            m_b = analyze_set(exp, pred, ["beam_id"], False)
            m_t_t = analyze_set(exp, pred, ["top_main_bars_left", "top_main_bars_mid", "top_main_bars_right"], True)
            m_t_b = analyze_set(exp, pred, ["bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right"], True)
            m_main = add_t(m_t_t, m_t_b)
            m_stirrup = analyze_set(exp, pred, ["stirrups_left", "stirrups_middle", "stirrups_right"], False)
            m_face = analyze_set(exp, pred, ["face_bars"], False)
            m_lap = analyze_set(exp, pred, ["lap_length_top_left", "lap_length_top_right", "lap_length_bottom_left", "lap_length_bottom_right"], False)
        
        tw, ew, fs = 0, 0, {}
        for f in COMPARE_FIELDS:
            s, _ = compare_field(f, (exp or {}).get(f), (pred or {}).get(f))
            w = FIELD_WEIGHTS[f]
            tw += w; ew += s * w
            fs[f] = s
            
        return {
            "status": "HALLUCINATION" if exp is None else ("MISSED" if pred is None else "MATCHED"),
            "expected": exp, "predicted": pred,
            "_source": (pred or {}).get("_source", source),
            "score": round(ew / tw * 100, 1) if tw else 0, "field_scores": fs,
            "metrics": {"beam": m_b, "main": m_main, "stirrup": m_stirrup, "face": m_face, "lap": m_lap}
        }

    beam_results = [_score_pair(e, p, "split") for e, p in aligned_pairs]


    # === 統計 (aligned 和 split 分別計算) ===
    def _calc_stats(results):
        matched = sum(1 for b in results if b["status"] == "MATCHED")
        missed = sum(1 for b in results if b["status"] == "MISSED")
        halluc = sum(1 for b in results if b["status"] == "HALLUCINATION")

        fscores = defaultdict(list)
        for b in results:
            if b["status"] == "MATCHED":
                for f in COMPARE_FIELDS:
                    fscores[f].append(b["field_scores"].get(f, 0))
            elif b["status"] == "MISSED":
                for f in COMPARE_FIELDS:
                    fscores[f].append(0.0)
            elif b["status"] == "HALLUCINATION":
                # 幻覺梁：傳統欄位準確率也只扣分 beam_id，其餘不計入分母
                fscores["beam_id"].append(0.0)
        avg_fs = {f: round(sum(fscores[f]) / len(fscores[f]) * 100, 1) if fscores[f] else 0 for f in COMPARE_FIELDS}
        
        agg = {"beam": [0]*5, "main": [0]*5, "stirrup": [0]*5, "face": [0]*5, "lap": [0]*5}
        for b in results:
            for k in agg:
                agg[k] = [x + y for x, y in zip(agg[k], b["metrics"][k])]
                
        def calc_cat(m):
            mis, wrg, mpl, hal, exp_cnt = m
            # If denominator is 0 but hallucinations exist, treat denominator as 1 for rate math, otherwise 1.
            denom = exp_cnt if exp_cnt > 0 else (1 if hal > 0 else 1)
            rate = max(0.0, 100.0 - (mis/denom*100) - (wrg/denom*100) - (mpl/denom*100) - (hal/denom*100))
            return {
                "acc": round(rate, 1),
                "mis": round(mis/denom*100, 1),
                "wrg": round(wrg/denom*100, 1), 
                "mpl": round(mpl/denom*100, 1),
                "hal": round(hal/denom*100, 1),
                "raw": m
            }
            
        m_breakdown = {k: calc_cat(agg[k]) for k in agg}
        overall = (m_breakdown["beam"]["acc"] + m_breakdown["main"]["acc"] + m_breakdown["stirrup"]["acc"] + 0.5 * m_breakdown["face"]["acc"] + 0.5 * m_breakdown["lap"]["acc"]) / 4.0

        return matched, missed, halluc, avg_fs, overall, m_breakdown

    a_matched, a_missed, a_halluc, a_fs, a_overall, m_breakdown = _calc_stats(beam_results)

    # 整體統計
    total_matched = a_matched
    prec = a_matched / (a_matched + a_halluc) if (a_matched + a_halluc) else 0
    rec = a_matched / (a_matched + a_missed) if (a_matched + a_missed) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "pdf_file": os.path.basename(pdf_path),
        "elapsed": elapsed,
        "expected_count": len(expected_beams),
        "predicted_count": len(aligned),
        "predicted_parent_count": 0,
        "predicted_split_count": len(aligned),
        "expected_split_count": ground_truth.get("expected_split_count", "?"),
        "matched": a_matched,
        "missed": a_missed,
        "hallucinated": a_halluc,
        "precision": round(prec * 100, 1),
        "recall": round(rec * 100, 1),
        "f1": round(f1 * 100, 1),
        "overall_accuracy": round(a_overall, 1),
        "aligned_accuracy": round(a_overall, 1),
        "split_accuracy": round(a_overall, 1),
        "field_accuracy": a_fs,
        "metrics_breakdown": m_breakdown,
        "beam_details": beam_results,
        "cv_params": ground_truth.get("cv_params", {}),
        "api_metrics": api_metrics,
    }

# ================================================================
# HTML 視覺化報告生成器
# ================================================================
def _val_display(val):
    """將值轉為顯示用字串"""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "(空)"
    s = str(val).strip()
    return s if s else "(空)"

def _match_class(field, exp_val, pred_val):
    """根據比對結果回傳 CSS class"""
    score, _ = compare_field(field, exp_val, pred_val)
    if score >= 1.0:
        return "match"
    elif score >= 0.5:
        return "partial"
    else:
        return "mismatch"

def _source_stats_html(beam_details):
    """產生分源統計 HTML (原始圖 vs Split圖)"""
    stats = {"aligned": {"total": 0, "matched": 0, "scores": []},
             "split": {"total": 0, "matched": 0, "scores": []}}
    for b in beam_details:
        src = b.get("_source", "?")
        if src not in stats:
            continue
        stats[src]["total"] += 1
        if b["status"] == "MATCHED":
            stats[src]["matched"] += 1
            stats[src]["scores"].append(b["score"])

    html = ""
    for src, label, icon in [("aligned", "原始圖", "🖼"), ("split", "Split圖", "✂️")]:
        s = stats[src]
        avg = round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else 0
        html += f'''<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px">
            <div style="font-size:.85rem;font-weight:600;color:#e2e8f0;margin-bottom:6px">{icon} {label}</div>
            <div style="font-size:.7rem;color:#94a3b8">偵測數量: {s["total"]}</div>
            <div style="font-size:.7rem;color:#94a3b8">配對成功: {s["matched"]}</div>
            <div style="font-size:1.1rem;font-weight:700;color:{"#22c55e" if avg >= 80 else ("#eab308" if avg >= 50 else "#ef4444")};margin-top:4px">平均正確率: {avg}%</div>
        </div>'''
    return html

def generate_html_report(reports: list, voting_rounds: int = 1) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_pdfs = len(reports)
    avg_acc = round(sum(r["overall_accuracy"] for r in reports) / total_pdfs, 1) if total_pdfs else 0
    avg_prec = round(sum(r["precision"] for r in reports) / total_pdfs, 1) if total_pdfs else 0
    avg_rec = round(sum(r["recall"] for r in reports) / total_pdfs, 1) if total_pdfs else 0
    avg_f1 = round(sum(r["f1"] for r in reports) / total_pdfs, 1) if total_pdfs else 0
    
    total_llm_calls = sum(r.get("api_metrics", {}).get("llm_calls", 0) for r in reports)
    total_prompt_tokens = sum(r.get("api_metrics", {}).get("prompt_tokens", 0) for r in reports)
    total_candidates_tokens = sum(r.get("api_metrics", {}).get("candidates_tokens", 0) for r in reports)

    # Helper: render a single beam card
    def _render_beam_card(b, idx, zebra_idx):
        exp = b.get("expected") or {}
        pred = b.get("predicted") or {}
        status = b["status"]
        zebra = "zebra-a" if zebra_idx % 2 == 0 else "zebra-b"

        if status == "MISSED":
            card_class = f"beam-card missed {zebra}"
            badge = '<span class="badge badge-red">⛔ 遺漏（未偵測到）</span>'
        elif status == "HALLUCINATION":
            card_class = f"beam-card hallucination {zebra}"
            badge = '<span class="badge badge-orange">👻 幻覺（多出的）</span>'
        else:
            card_class = f"beam-card {zebra}"
            badge = f'<span class="badge badge-blue">✓ 配對成功 — {b["score"]}%</span>'

        src = b.get("_source", "?")
        src_badge = '<span class="badge badge-src-aligned">🖼 原始圖</span>' if src == "aligned" else ('<span class="badge badge-src-split">✂️ Split圖</span>' if src == "split" else '')

        def cell(field, colspan=1):
            e = exp.get(field)
            p = pred.get(field)
            ev = html_mod.escape(_val_display(e))
            pv = html_mod.escape(_val_display(p))
            cls = _match_class(field, e, p) if status == "MATCHED" else ("missed-field" if status == "MISSED" else "halluc-field")
            span = f' style="grid-column: span {colspan}"' if colspan > 1 else ''
            label = field.replace("_", " ")
            return f'''<div class="field-cell {cls}"{span}>
                <div class="field-label">{label}</div>
                <div class="field-row"><span class="tag tag-pred">AI</span><span class="field-value">{pv}</span></div>
                <div class="field-row"><span class="tag tag-exp">答案</span><span class="field-value">{ev}</span></div>
            </div>'''

        conf = pred.get("self_confidence", "")
        note_val = html_mod.escape(str(pred.get("note", "") or ""))

        ocr_raw = str(pred.get("_ocr_text", "(無 OCR 資料)"))
        ocr_content_html = f'<pre class="ocr-text">{html_mod.escape(ocr_raw)}</pre>'
        
        # 嘗試解析為九宮格結構 (新版與舊版格式相容)
        cells = {}
        is_grid_format = False
        
        if "[" in ocr_raw and "]: " in ocr_raw:
            # 新版結構化格式
            for line in ocr_raw.split('\n'):
                line = line.strip()
                if line.startswith('[') and ']: ' in line:
                    b_end = line.find(']')
                    key = line[1:b_end]
                    val = line[line.find(']: ')+3:]
                    cells[key] = val
            if len(cells) == 9:
                is_grid_format = True
        elif "@" in ocr_raw and "(信心:" in ocr_raw:
            # 舊版扁平格式，使用正則表達式萃取
            import re
            matches = re.finditer(r'"([^"]+)"\s*@\s*([^\s\(]+)(.*?)\(信心:', ocr_raw)
            temp_cells = {k: [] for k in ["左上方", "正上方", "右上方", "正左方", "正中央", "正右方", "左下方", "正下方", "右下方"]}
            for m in matches:
                text = m.group(1)
                pos = m.group(2)
                hint = m.group(3).strip()
                clean_pos = pos.replace("端", "") # 支援舊的「左端上方」寫法
                if clean_pos in temp_cells:
                    temp_cells[clean_pos].append(f'"{text}" {hint}'.strip())
                    is_grid_format = True
            
            if is_grid_format:
                for k in temp_cells:
                    items = temp_cells[k]
                    cells[k] = ", ".join(items) if items else "(空)"

        if is_grid_format:
            grid_html = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 8px; font-family: monospace;">'
            for key in ["左上方", "正上方", "右上方", "正左方", "正中央", "正右方", "左下方", "正下方", "右下方"]:
                val = cells.get(key, "(空)")
                # 用淺灰色背景區分邊距，文字換行以避免過長
                grid_html += f'<div style="background: #1e293b; border: 1px solid #475569; padding: 6px; border-radius: 4px; font-size: 0.75rem; color: #cbd5e1; word-break: break-all;">'
                grid_html += f'<div style="color: #94a3b8; font-weight: bold; margin-bottom: 2px;">{key}</div>'
                grid_html += f'{html_mod.escape(val)}</div>'
            grid_html += '</div>'
            ocr_content_html = f'<pre class="ocr-text" style="margin-bottom: 12px;">{html_mod.escape(ocr_raw)}</pre>' + grid_html
        
        raw_llm_json = pred.get("_raw_llm", "")
        raw_llm_retry_json = pred.get("_raw_llm_retry", "")

        def _render_llm_grid(json_str, label, icon, bg, border, header_color, val_color):
            """將 LLM 回覆 JSON 渲染為九宮格 + NOTE 獨立欄位"""
            try:
                data = json_mod.loads(json_str)
            except Exception:
                return (
                    f"<div style='margin-top:10px;padding:10px;background:{bg};"
                    f"border:1px solid {border};border-radius:4px;'>"
                    f"<strong style='color:{header_color};font-size:0.8rem;'>{icon} {label}</strong>"
                    f"<pre style='margin:6px 0 0;font-size:0.75rem;color:{val_color};"
                    f"white-space:pre-wrap;word-break:break-all;'>{html_mod.escape(json_str)}</pre></div>"
                )

            def _fv(key):
                v = data.get(key, "")
                if isinstance(v, list):
                    clean = [x for x in v if x not in ("LLM沒有東西", "LLM看不出來")]
                    return ", ".join(clean) if clean else "(空)"
                s = str(v).strip() if v else ""
                return s if s and s not in ("LLM沒有東西", "LLM看不出來") else "(空)"

            def _cell(label_txt, key, extra_style=""):
                val = html_mod.escape(_fv(key))
                is_empty = val == "(空)"
                opacity = "0.45" if is_empty else "1"
                return (
                    f"<div style='background:#1e293b;border:1px solid #334155;border-radius:4px;"
                    f"padding:6px 8px;{extra_style};opacity:{opacity}'>"
                    f"<div style='font-size:0.6rem;color:#64748b;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px'>{label_txt}</div>"
                    f"<div style='font-size:0.78rem;color:{val_color};word-break:break-all'>{val}</div>"
                    f"</div>"
                )

            note_val = html_mod.escape(str(data.get("note", "") or ""))
            beam_id_val = html_mod.escape(_fv("beam_id"))
            dim_val = html_mod.escape(_fv("dimensions"))

            grid_html = (
                f"<div style='margin-top:6px;'>"
                # header row: beam_id + dimensions
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:4px'>"
                f"{_cell('beam_id', 'beam_id')}"
                f"{_cell('dimensions', 'dimensions')}"
                f"</div>"
                # 九宮格：上層主筋
                f"<div style='font-size:0.6rem;color:#475569;text-align:center;margin:4px 0 2px;"
                f"text-transform:uppercase;letter-spacing:1px'>▲ 上層主筋 TOP BARS</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:4px'>"
                f"{_cell('左', 'top_main_bars_left')}"
                f"{_cell('中', 'top_main_bars_mid')}"
                f"{_cell('右', 'top_main_bars_right')}"
                f"</div>"
                # 九宮格：箍筋
                f"<div style='font-size:0.6rem;color:#475569;text-align:center;margin:4px 0 2px;"
                f"text-transform:uppercase;letter-spacing:1px'>↔ 箍筋 STIRRUPS</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:4px'>"
                f"{_cell('左', 'stirrups_left')}"
                f"{_cell('中', 'stirrups_middle')}"
                f"{_cell('右', 'stirrups_right')}"
                f"</div>"
                # 九宮格：下層主筋
                f"<div style='font-size:0.6rem;color:#475569;text-align:center;margin:4px 0 2px;"
                f"text-transform:uppercase;letter-spacing:1px'>▼ 下層主筋 BOTTOM BARS</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:4px'>"
                f"{_cell('左', 'bottom_main_bars_left')}"
                f"{_cell('中', 'bottom_main_bars_mid')}"
                f"{_cell('右', 'bottom_main_bars_right')}"
                f"</div>"
                # face_bars + lap_length
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;margin-bottom:4px'>"
                f"{_cell('腰筋', 'face_bars')}"
                f"{_cell('搭接↑左', 'lap_length_top_left')}"
                f"{_cell('搭接↑右', 'lap_length_top_right')}"
                f"</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;margin-bottom:6px'>"
                f"<div></div>"
                f"{_cell('搭接↓左', 'lap_length_bottom_left')}"
                f"{_cell('搭接↓右', 'lap_length_bottom_right')}"
                f"</div>"
                # NOTE 獨立欄位
                f"<div style='background:#0f172a;border:1px solid #1e3a5f;border-radius:4px;"
                f"padding:6px 10px;'>"
                f"<div style='font-size:0.6rem;color:#64748b;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px'>📝 Note</div>"
                f"<div style='font-size:0.75rem;color:#94a3b8'>{note_val if note_val else '(無)'}</div>"
                f"</div>"
                f"</div>"  # end grid_html wrapper
            )

            # 原始 JSON 可展開查看
            raw_detail = (
                f"<details style='margin-top:4px'>"
                f"<summary style='font-size:0.65rem;color:#475569;cursor:pointer;padding:2px 4px'>"
                f"查看原始 JSON</summary>"
                f"<pre style='font-size:0.68rem;color:#64748b;white-space:pre-wrap;"
                f"word-break:break-all;padding:6px;margin:0'>{html_mod.escape(json_str)}</pre>"
                f"</details>"
            )

            return (
                f"<div style='margin-top:10px;padding:10px;background:{bg};"
                f"border:1px solid {border};border-radius:6px;'>"
                f"<strong style='color:{header_color};font-size:0.8rem;'>{icon} {label}</strong>"
                f"{grid_html}"
                f"{raw_detail}"
                f"</div>"
            )

        import json as json_mod

        raw_llm_html = ""
        if raw_llm_json:
            raw_llm_html += _render_llm_grid(
                raw_llm_json,
                "LLM 原始回覆 (第一次)",
                "🤖", "#0b1829", "#1e3a5f", "#94a3b8", "#cbd5e1"
            )
        if raw_llm_retry_json:
            raw_llm_html += _render_llm_grid(
                raw_llm_retry_json,
                "LLM 二次重試回覆 (針對缺漏補齊)",
                "🔄", "#1a0f00", "#78350f", "#fbbf24", "#fef3c7"
            )
        
        return f'''<div class="{card_class}">
                <div class="beam-header">
                    <span class="beam-idx">#{idx+1}</span>
                    {src_badge}
                    {badge}
                </div>
                <div class="beam-grid row-3">
                    {cell("beam_id")}
                    {cell("dimensions")}
                    <div class="field-cell info-field">
                        <div class="field-label">self confidence</div>
                        <div class="field-row"><span class="tag tag-pred">AI</span><span class="field-value">{conf}</span></div>
                    </div>
                </div>
                <div class="beam-grid row-1">
                    <div class="field-cell info-field" style="grid-column: span 3">
                        <div class="field-label">note</div>
                        <div class="field-row"><span class="tag tag-pred">AI</span><span class="field-value note-text">{note_val if note_val else "(無)"}</span></div>
                    </div>
                </div>

                <details class="ocr-panel">
                    <summary>🔍 OCR 預掃結果 — {html_mod.escape(str(pred.get("_crop_file", "")))}</summary>
                    {ocr_content_html}
                    {raw_llm_html}
                </details>
            <div class="section-divider"><span>上層主筋 TOP BARS</span></div>
            <div class="beam-grid row-3">
                {cell("top_main_bars_left")}
                {cell("top_main_bars_mid")}
                {cell("top_main_bars_right")}
            </div>
            <div class="section-divider"><span>下層主筋 BOTTOM BARS</span></div>
            <div class="beam-grid row-3">
                {cell("bottom_main_bars_left")}
                {cell("bottom_main_bars_mid")}
                {cell("bottom_main_bars_right")}
            </div>
            <div class="section-divider"><span>箍筋 STIRRUPS</span></div>
            <div class="beam-grid row-3">
                {cell("stirrups_left")}
                {cell("stirrups_middle")}
                {cell("stirrups_right")}
            </div>
            <div class="section-divider"><span>腰筋 FACE BARS</span></div>
            <div class="beam-grid row-1">
                {cell("face_bars", 3)}
            </div>
            <div class="section-divider"><span>搭接長度 LAP LENGTH</span></div>
            <div class="beam-grid row-2">
                {cell("lap_length_top_left")}
                {cell("lap_length_top_right")}
            </div>
            <div class="beam-grid row-2">
                {cell("lap_length_bottom_left")}
                {cell("lap_length_bottom_right")}
            </div>
        </div>'''

    # Build per-PDF sections
    pdf_sections = ""
    for ri, r in enumerate(reports):
        beams = r["beam_details"]

        def beam_sort_key(b):
            p = b.get("predicted")
            e = b.get("expected")
            pred = p if p else (e if e else {})
            group = pred.get("span_group")
            raw_idx = pred.get("_raw_span_idx")
            eid = (e.get("beam_id", "") if e else "") or (p.get("beam_id", "") if p else "")
            return (
                str(group) if group is not None else "Z_" + eid,
                raw_idx if raw_idx is not None else 0
            )
            
        beams.sort(key=beam_sort_key)

        beams_html = ""
        current_group = None
        for i, b in enumerate(beams):
            pred = b.get("predicted") or {}
            grp = pred.get("span_group")
            
            if grp and grp != current_group:
                if current_group:
                    beams_html += '</div></div>'
                current_group = grp
                beams_html += f'<div style="border: 2px dashed #94a3b8; border-radius: 8px; padding: 16px; margin-bottom: 24px; background: #f8fafc;"><h4 style="margin:0 0 12px 0; color: #334155; display:flex; align-items:center; gap:8px;">🔗 連續跨關聯群組: {grp}</h4><div style="display: flex; flex-direction: column; gap: 16px;">'
            elif not grp and current_group:
                beams_html += '</div></div>'
                current_group = None
                
            card_html = _render_beam_card(b, i, i)
            # 讓卡片在連續跨群組中可以正確垂直堆疊
            if current_group:
                card_html = card_html.replace('margin-bottom: 20px;', 'margin-bottom: 0; width: 100%;')
            beams_html += card_html
            
        if current_group:
            beams_html += '</div></div>'

        a_count = len(beams)
        a_acc = r.get("aligned_accuracy", 0)

        # Field accuracy bar chart
        field_bars = ""
        for f in COMPARE_FIELDS:
            acc = r["field_accuracy"].get(f, 0)
            color = "#22c55e" if acc >= 90 else ("#eab308" if acc >= 60 else "#ef4444")
            field_bars += f'''<div class="fbar-row">
                <span class="fbar-label">{f.replace("_"," ")}</span>
                <div class="fbar-track"><div class="fbar-fill" style="width:{acc}%;background:{color}"></div></div>
                <span class="fbar-val">{acc}%</span>
            </div>'''

        exp_split = r.get("expected_split_count", "?")
        tab_id = f"pdf{ri}"
        mbd = r.get("metrics_breakdown", {})

        pdf_sections += f'''
        <div class="pdf-section">
            <div class="pdf-header">
                <h2>📄 {html_mod.escape(r["pdf_file"])}</h2>
                <span class="elapsed">{r["elapsed"]}s</span>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">偵測到梁數量</div>
                    <div class="stat-value">{r["predicted_count"]}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">實際梁數量</div>
                    <div class="stat-value">{r["expected_count"]}</div>
                </div>
                <div class="stat-card accent">
                    <div class="stat-label">Precision</div>
                    <div class="stat-value">{r["precision"]}%</div>
                </div>
                <div class="stat-card accent">
                    <div class="stat-label">Recall</div>
                    <div class="stat-value">{r["recall"]}%</div>
                </div>
                <div class="stat-card accent">
                    <div class="stat-label">F1 Score</div>
                    <div class="stat-value">{r["f1"]}%</div>
                </div>
                <div class="stat-card highlight">
                    <div class="stat-label">⭐ 加權正確率</div>
                    <div class="stat-value">{r["overall_accuracy"]}%</div>
                </div>
            </div>
            
            <details style="margin-bottom: 16px; background: #0f172a; border: 1px solid #1e3a8a; border-radius: 8px; padding: 12px; cursor: pointer;">
                <summary style="font-size: .85rem; font-weight: 600; color: #60a5fa;">🔍 查看詳細五大分項正確率 (可點此展開)</summary>
                <div style="margin-top: 12px; font-size: .75rem; color: #cbd5e1; display: flex; flex-direction: column; gap: 8px;">
                    <div><b>1. 梁名準確率 ({mbd['beam']['acc']}%)</b> = 100% - 未讀取({mbd['beam']['mis']}%) - 填錯({mbd['beam']['wrg']}%) - 幻覺({mbd['beam']['hal']}%)</div>
                    <div><b>2. 主筋正確率 ({mbd['main']['acc']}%)</b> = 100% - 未讀取({mbd['main']['mis']}%) - 錯置({mbd['main']['mpl']}%) - 填錯({mbd['main']['wrg']}%) - 幻覺({mbd['main']['hal']}%)</div>
                    <div><b>3. 剪力筋正確率 ({mbd['stirrup']['acc']}%)</b> = 100% - 未讀取({mbd['stirrup']['mis']}%) - 錯置({mbd['stirrup']['mpl']}%) - 填錯({mbd['stirrup']['wrg']}%) - 幻覺({mbd['stirrup']['hal']}%)</div>
                    <div><b>4. 腰筋正確率 ({mbd['face']['acc']}%)</b> = 100% - 未讀取({mbd['face']['mis']}%) - 填錯({mbd['face']['wrg']}%) - 幻覺({mbd['face']['hal']}%)</div>
                    <div><b>5. 搭長正確率 ({mbd['lap']['acc']}%)</b> = 100% - 未讀取({mbd['lap']['mis']}%) - 填錯({mbd['lap']['wrg']}%) - 幻覺({mbd['lap']['hal']}%)</div>
                    <div style="color: #93c5fd; margin-top: 4px; border-top: 1px dashed #334155; padding-top: 8px;"><b>綜合加權公式</b>: ((1) + (2) + (3) + 0.5×(4) + 0.5×(5)) / 4.0 = <b>{r["overall_accuracy"]}%</b></div>
                </div>
            </details>

            {"" if len(reports) >= 2 else f"""
            <details class="field-acc-panel">
                <summary>📊 各細項欄位傳統得分比率</summary>
                <div class="fbar-container">{field_bars}</div>
            </details>

            <div id="{tab_id}-aligned" class="tab-panel active">
                {beams_html if beams_html else '<div style="padding:20px;color:#64748b;text-align:center">無資料</div>'}
            </div>
            """}
            {f"""
            <details class="field-acc-panel" style="margin-bottom: 0;">
                <summary style="font-size: .9rem; font-weight: 600; color: #fbbf24;">📋 展開完整報告（欄位分析 + 梁比對表）</summary>
                <div style="padding: 12px 0;">
                    <details class="field-acc-panel" style="margin-bottom: 12px;">
                        <summary>📊 各細項欄位傳統得分比率</summary>
                        <div class="fbar-container">{field_bars}</div>
                    </details>
                    <div id="{tab_id}-aligned" class="tab-panel active">
                        {beams_html if beams_html else '<div style="padding:20px;color:#64748b;text-align:center">無資料</div>'}
                    </div>
                </div>
            </details>
            """ if len(reports) >= 2 else ""}
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI-PDF Benchmark Report — {now_str}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI','Microsoft JhengHei',system-ui,sans-serif;padding:24px;min-height:100vh}}
h1{{font-size:1.5rem;font-weight:700}}
h2{{font-size:1.2rem;font-weight:600;color:#93c5fd}}
h3{{font-size:1rem;font-weight:600;color:#94a3b8;margin:20px 0 12px}}

.report-header{{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:12px;padding:24px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
.report-header .meta{{color:#94a3b8;font-size:.85rem}}

.summary-bar{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:28px}}
.summary-card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;text-align:center}}
.summary-card .label{{font-size:.75rem;color:#64748b;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}}
.summary-card .value{{font-size:1.4rem;font-weight:700;color:#f8fafc}}
.summary-card.gold .value{{color:#fbbf24}}

.pdf-section{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;margin-bottom:24px}}
.pdf-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}}
.elapsed{{font-size:.8rem;color:#64748b;background:#0f172a;padding:4px 10px;border-radius:6px}}

.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}}
.stat-card{{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;text-align:center}}
.stat-card .stat-label{{font-size:.7rem;color:#64748b;margin-bottom:2px}}
.stat-card .stat-value{{font-size:1.2rem;font-weight:700;color:#e2e8f0}}
.stat-card.accent .stat-value{{color:#60a5fa}}
.stat-card.highlight{{border-color:#fbbf24}}
.stat-card.highlight .stat-value{{color:#fbbf24;font-size:1.4rem}}

.field-acc-panel{{background:#0f172a;border:1px solid #334155;border-radius:8px;margin-bottom:16px;cursor:pointer}}
.field-acc-panel summary{{padding:12px 16px;font-size:.85rem;color:#94a3b8}}
.fbar-container{{padding:8px 16px 16px}}
.fbar-row{{display:flex;align-items:center;gap:8px;margin-bottom:4px}}
.fbar-label{{font-size:.7rem;color:#94a3b8;width:170px;text-align:right;flex-shrink:0}}
.fbar-track{{flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden}}
.fbar-fill{{height:100%;border-radius:4px;transition:width .3s}}
.fbar-val{{font-size:.7rem;color:#e2e8f0;width:42px;text-align:right;font-weight:600}}

.beams-title{{border-top:1px solid #334155;padding-top:16px}}

.beam-card{{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;margin-bottom:14px}}
.beam-card.missed{{border-color:#ef4444;border-width:2px;background:#1a0505}}
.beam-card.hallucination{{border-color:#f97316;border-width:2px;background:#1a0f05}}
.beam-header{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
.beam-idx{{font-size:.8rem;font-weight:700;color:#64748b;background:#1e293b;padding:2px 8px;border-radius:4px}}
.badge{{font-size:.75rem;padding:3px 10px;border-radius:6px;font-weight:600}}
.badge-blue{{background:#1e3a5f;color:#60a5fa}}
.badge-red{{background:#451a1a;color:#f87171}}
.badge-orange{{background:#451a00;color:#fb923c}}
.badge-src-aligned{{background:#1e3a5f;color:#93c5fd;font-size:.65rem}}
.badge-src-split{{background:#3b1f5e;color:#c084fc;font-size:.65rem}}

.beam-card.zebra-a{{background:#0c1a2e}}
.beam-card.zebra-b{{background:#111f38}}
.beam-card.zebra-a .field-cell{{background:#152240}}
.beam-card.zebra-b .field-cell{{background:#1a2a4a}}

.tab-bar{{display:flex;gap:0;margin:16px 0 0;border-bottom:2px solid #334155}}
.tab-btn{{background:transparent;border:none;color:#64748b;padding:10px 20px;font-size:.85rem;font-weight:600;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s}}
.tab-btn:hover{{color:#94a3b8}}
.tab-btn.active{{color:#60a5fa;border-bottom-color:#60a5fa;background:#1e293b}}
.tab-panel{{display:none;padding-top:12px}}
.tab-panel.active{{display:block}}

.ocr-panel{{margin:6px 0;border:1px solid #1e3a5f;border-radius:6px;background:#0c1a2e}}
.ocr-panel summary{{padding:6px 10px;font-size:.7rem;color:#60a5fa;cursor:pointer;font-weight:600}}
.ocr-panel summary:hover{{color:#93c5fd}}
.ocr-text{{padding:8px 12px;font-size:.7rem;color:#94a3b8;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;margin:0;font-family:'Consolas','Courier New',monospace}}

.beam-grid{{display:grid;gap:8px;margin-bottom:4px}}
.beam-grid.row-3{{grid-template-columns:1fr 1fr 1fr}}
.beam-grid.row-2{{grid-template-columns:1fr 1fr}}
.beam-grid.row-1{{grid-template-columns:1fr}}

.section-divider{{display:flex;align-items:center;gap:8px;margin:10px 0 6px;color:#475569;font-size:.65rem;text-transform:uppercase;letter-spacing:1px}}
.section-divider::before,.section-divider::after{{content:"";flex:1;height:1px;background:#1e293b}}

.field-cell{{background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 10px;position:relative}}
.field-cell.match{{border-color:#22c55e;background:#052e16}}
.field-cell.partial{{border-color:#eab308;background:#1a1505}}
.field-cell.mismatch{{border-color:#ef4444;background:#1a0505}}
.field-cell.missed-field{{border-color:#ef4444;opacity:.7}}
.field-cell.halluc-field{{border-color:#f97316;opacity:.7}}
.field-cell.info-field{{border-color:#334155;background:#1e293b}}

.field-label{{font-size:.6rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;font-weight:600}}
.field-row{{display:flex;align-items:center;gap:6px;margin-bottom:2px}}
.tag{{font-size:.55rem;padding:1px 5px;border-radius:3px;font-weight:700;flex-shrink:0;width:28px;text-align:center}}
.tag-pred{{background:#1e3a5f;color:#60a5fa}}
.tag-exp{{background:#14532d;color:#4ade80}}
.field-value{{font-size:.8rem;color:#e2e8f0;word-break:break-all}}
.note-text{{font-size:.75rem;color:#94a3b8}}

@media(max-width:640px){{
  .beam-grid.row-3{{grid-template-columns:1fr}}
  .beam-grid.row-2{{grid-template-columns:1fr}}
  .stats-grid{{grid-template-columns:repeat(2,1fr)}}
}}
</style>
</head>
<body>
<div class="report-header">
    <div>
        <h1>🔬 AI-PDF Benchmark Report</h1>
        <div class="meta">{now_str} &nbsp;|&nbsp; Self-Consistency x{voting_rounds} &nbsp;|&nbsp; {total_pdfs} 份 PDF</div>
    </div>
</div>

<div class="summary-bar">
    <div class="summary-card gold">
        <div class="label">平均加權正確率</div>
        <div class="value">{avg_acc}%</div>
    </div>
    <div class="summary-card">
        <div class="label">Precision</div>
        <div class="value">{avg_prec}%</div>
    </div>
    <div class="summary-card">
        <div class="label">Recall</div>
        <div class="value">{avg_rec}%</div>
    </div>
    <div class="summary-card">
        <div class="label">F1</div>
        <div class="value">{avg_f1}%</div>
    </div>
    <div class="summary-card" style="background: linear-gradient(135deg, #1e3a8a, #111827);">
        <div class="label" style="color: #60a5fa;">LLM Token 消耗 (輸入+輸出)</div>
        <div class="value" style="font-size: 1.2rem;">{total_prompt_tokens + total_candidates_tokens:,} <span style="font-size: 0.7rem; color: #94a3b8;">({total_llm_calls} API Calls)</span></div>
    </div>
</div>

{pdf_sections}

<div style="text-align:center;color:#334155;font-size:.7rem;padding:24px 0">
    Generated by AI-PDF Benchmark Runner
</div>
<script>
function switchTab(pdfId, tab) {{
    const parent = document.getElementById(pdfId + '-aligned').parentElement;
    parent.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(pdfId + '-' + tab).classList.add('active');
    event.target.classList.add('active');
}}
</script>
</body>
</html>'''

# ================================================================
# 終端報告
# ================================================================
def print_report(reports, verbose=False):
    print("\n" + "=" * 70)
    print(f"  AI-PDF Benchmark Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    for r in reports:
        print(f"\n{'─' * 55}")
        print(f"  📄 {r['pdf_file']}  ({r['elapsed']}s)")
        print(f"  梁: 預期{r['expected_count']} 預測{r['predicted_count']} "
              f"配對{r['matched']} 遺漏{r['missed']} 幻覺{r['hallucinated']}")
        print(f"  P:{r['precision']}% R:{r['recall']}% F1:{r['f1']}%  ⭐{r['overall_accuracy']}%")
        if verbose:
            for f in COMPARE_FIELDS:
                a = r["field_accuracy"].get(f, 0)
                bar = "█" * int(a / 10) + "░" * (10 - int(a / 10))
                print(f"    {f:<28} {a:>5.1f}% {bar}")
    if len(reports) > 1:
        avg = round(sum(r["overall_accuracy"] for r in reports) / len(reports), 1)
        print(f"\n{'═' * 55}")
        print(f"  總平均正確率: {avg}%")
    print()

# ================================================================
# --init: 預跑 pipeline + Web Editor
# ================================================================

EDITOR_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-TW"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Truth Editor — {pdf_file}</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI','Microsoft JhengHei',system-ui,sans-serif;padding:20px}
h1{font-size:1.3rem;font-weight:700;margin-bottom:4px}
.header{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:20px;margin-bottom:16px}
.meta{color:#94a3b8;font-size:.8rem;margin-top:4px}
.meta-inputs{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
.meta-inputs label{font-size:.7rem;color:#64748b}
.meta-inputs input{background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:4px 8px;border-radius:4px;width:100px;font-size:.85rem}
.beam-card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin-bottom:12px;position:relative}
.beam-card .beam-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.beam-card .beam-num{font-size:.8rem;font-weight:700;color:#60a5fa;background:#1e3a5f;padding:2px 10px;border-radius:4px}
.del-btn{background:#451a1a;color:#f87171;border:1px solid #7f1d1d;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:.7rem}
.del-btn:hover{background:#7f1d1d}
.beam-grid{display:grid;gap:6px;margin-bottom:4px}
.g3{grid-template-columns:1fr 1fr 1fr}
.g2{grid-template-columns:1fr 1fr}
.g1{grid-template-columns:1fr}
.fcell{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 8px}
.fcell label{display:block;font-size:.6rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;font-weight:600}
.fcell input,.fcell textarea{width:100%;background:#1e293b;border:1px solid #475569;color:#e2e8f0;padding:4px 6px;border-radius:4px;font-size:.8rem;font-family:inherit}
.fcell input:focus,.fcell textarea:focus{outline:none;border-color:#60a5fa}
.fcell textarea{resize:vertical;min-height:28px}
.sdiv{display:flex;align-items:center;gap:6px;margin:8px 0 4px;color:#475569;font-size:.6rem;text-transform:uppercase;letter-spacing:1px}
.sdiv::before,.sdiv::after{content:"";flex:1;height:1px;background:#1e293b}
.actions{position:sticky;bottom:0;background:#0f172a;border-top:1px solid #334155;padding:12px 0;display:flex;gap:10px;justify-content:center;z-index:10}
.btn{padding:10px 28px;border:none;border-radius:8px;font-weight:700;cursor:pointer;font-size:.9rem;transition:all .2s}
.btn-save{background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff}
.btn-save:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(34,197,94,.3)}
.btn-add{background:#1e3a5f;color:#60a5fa;border:1px solid #2563eb}
.btn-add:hover{background:#2563eb;color:#fff}
.toast{position:fixed;top:20px;right:20px;background:#22c55e;color:#fff;padding:12px 24px;border-radius:8px;font-weight:600;display:none;z-index:999;box-shadow:0 4px 12px rgba(0,0,0,.3)}
</style>
</head><body>
<div id="toast" class="toast"></div>
<div class="header">
    <h1>📝 Ground Truth Editor</h1>
    <div class="meta">{pdf_file} — AI 預跑結果，請校正後儲存</div>
    <div class="meta-inputs">
        <div><label>expected_beam_count</label><input id="metaBeamCount" type="number" value="{expected_beam_count}"></div>
        <div><label>expected_split_count</label><input id="metaSplitCount" type="number" value="{expected_split_count}"></div>
        <div><label>page_num</label><input id="metaPageNum" type="number" value="{page_num}"></div>
    </div>
</div>
<div id="beamContainer"></div>
<div class="actions">
    <button class="btn btn-add" onclick="addBeam()">＋ 新增一筆梁</button>
    <button class="btn btn-save" onclick="saveAll()">💾 儲存 Ground Truth</button>
</div>
<script>
const FIELDS = [
    {row:[{f:"beam_id",w:1},{f:"dimensions",w:1}],cols:2},
    {section:"上層主筋 TOP BARS"},
    {row:[{f:"top_main_bars_left",w:1,list:true},{f:"top_main_bars_mid",w:1,list:true},{f:"top_main_bars_right",w:1,list:true}],cols:3},
    {section:"下層主筋 BOTTOM BARS"},
    {row:[{f:"bottom_main_bars_left",w:1,list:true},{f:"bottom_main_bars_mid",w:1,list:true},{f:"bottom_main_bars_right",w:1,list:true}],cols:3},
    {section:"箍筋 STIRRUPS"},
    {row:[{f:"stirrups_left",w:1},{f:"stirrups_middle",w:1},{f:"stirrups_right",w:1}],cols:3},
    {section:"腰筋 FACE BARS"},
    {row:[{f:"face_bars",w:1}],cols:1},
    {section:"搭接長度 LAP LENGTH"},
    {row:[{f:"lap_length_top_left",w:1},{f:"lap_length_top_right",w:1}],cols:2},
    {row:[{f:"lap_length_bottom_left",w:1},{f:"lap_length_bottom_right",w:1}],cols:2},
];
let beams = BEAMS_DATA_PLACEHOLDER;
function valStr(v){if(Array.isArray(v))return v.join(", ");return v==null?"":String(v)}
function renderBeams(){
    const c=document.getElementById("beamContainer");
    c.innerHTML="";
    beams.forEach((b,i)=>{
        let h=`<div class="beam-card" id="beam${i}"><div class="beam-hdr"><span class="beam-num">#${i+1} ${b.beam_id||""}</span><button class="del-btn" onclick="delBeam(${i})">🗑 刪除</button></div>`;
        FIELDS.forEach(def=>{
            if(def.section){h+=`<div class="sdiv"><span>${def.section}</span></div>`;return}
            const cls=def.cols===3?"g3":def.cols===2?"g2":"g1";
            h+=`<div class="beam-grid ${cls}">`;
            def.row.forEach(fd=>{
                const v=valStr(b[fd.f]);
                const ph=fd.list?"逗號分隔, 例: 3-#8, 5-#10":"";
                h+=`<div class="fcell"><label>${fd.f.replace(/_/g," ")}</label><input data-beam="${i}" data-field="${fd.f}" data-list="${!!fd.list}" value="${v.replace(/"/g,"&quot;")}" placeholder="${ph}"></div>`;
            });
            h+=`</div>`;
        });
        h+=`</div>`;
        c.innerHTML+=h;
    });
}
function collectBeams(){
    beams.forEach((b,i)=>{
        document.querySelectorAll(`input[data-beam="${i}"]`).forEach(inp=>{
            const f=inp.dataset.field;
            const isList=inp.dataset.list==="true";
            if(isList){
                b[f]=inp.value?inp.value.split(",").map(s=>s.trim()).filter(s=>s):[];
            }else{
                b[f]=inp.value;
            }
        });
    });
}
function addBeam(){
    collectBeams();
    beams.push({beam_id:"",dimensions:"",top_main_bars_left:[],top_main_bars_mid:[],top_main_bars_right:[],bottom_main_bars_left:[],bottom_main_bars_mid:[],bottom_main_bars_right:[],stirrups_left:"",stirrups_middle:"",stirrups_right:"",face_bars:"",lap_length_top_left:"",lap_length_top_right:"",lap_length_bottom_left:"",lap_length_bottom_right:""});
    renderBeams();
    document.getElementById("metaBeamCount").value=beams.length;
    document.querySelector(".beam-card:last-child").scrollIntoView({behavior:"smooth"});
}
function delBeam(i){
    if(!confirm(`確定刪除 #${i+1} ${beams[i].beam_id||""}?`))return;
    collectBeams();
    beams.splice(i,1);
    renderBeams();
    document.getElementById("metaBeamCount").value=beams.length;
}
async function saveAll(){
    collectBeams();
    const data={
        _comment:"由 --init Web Editor 校正",
        pdf_file:PDF_FILE_PLACEHOLDER,
        page_num:parseInt(document.getElementById("metaPageNum").value)||0,
        expected_beam_count:parseInt(document.getElementById("metaBeamCount").value)||beams.length,
        expected_split_count:parseInt(document.getElementById("metaSplitCount").value)||0,
        cv_params:CV_PARAMS_PLACEHOLDER,
        beams:beams
    };
    try{
        const r=await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
        const j=await r.json();
        if(j.ok){showToast("✅ 已儲存 "+j.path)}else{showToast("❌ 儲存失敗: "+j.error)}
    }catch(e){showToast("❌ 網路錯誤: "+e.message)}
}
function showToast(msg){const t=document.getElementById("toast");t.innerText=msg;t.style.display="block";setTimeout(()=>t.style.display="none",3000)}
renderBeams();
</script>
</body></html>'''

def _build_editor_html(gt_data):
    """用 pipeline 結果建立 editor HTML"""
    html = EDITOR_HTML_TEMPLATE
    html = html.replace("{pdf_file}", gt_data["pdf_file"])
    html = html.replace("{expected_beam_count}", str(gt_data.get("expected_beam_count", 0)))
    html = html.replace("{expected_split_count}", str(gt_data.get("expected_split_count", 0)))
    html = html.replace("{page_num}", str(gt_data.get("page_num", 0)))
    html = html.replace("BEAMS_DATA_PLACEHOLDER", json.dumps(gt_data["beams"], ensure_ascii=False))
    html = html.replace("PDF_FILE_PLACEHOLDER", json.dumps(gt_data["pdf_file"], ensure_ascii=False))
    html = html.replace("CV_PARAMS_PLACEHOLDER", json.dumps(gt_data.get("cv_params", {}), ensure_ascii=False))
    return html

def _start_editor_server(gt_data, json_path, port=8765):
    """啟動臨時 HTTP 伺服器提供 Editor 介面 + 儲存端點"""
    import http.server
    import socketserver
    import threading
    from urllib.parse import urlparse

    editor_html = _build_editor_html(gt_data)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(editor_html.encode("utf-8"))

        def do_POST(self):
            if self.path == "/save":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {"ok": True, "path": os.path.basename(json_path), "beam_count": len(data.get("beams", []))}
                    self.wfile.write(json.dumps(resp).encode())
                    print(f"💾 已儲存 {json_path} ({len(data.get('beams', []))} beams)")
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # 靜音

    server = socketserver.TCPServer(("", port), Handler)
    print(f"\n🌐 Editor 已啟動: http://localhost:{port}")
    print(f"   修改完成後點擊「儲存」按鈕，JSON 會直接寫入 {json_path}")
    print(f"   按 Ctrl+C 關閉 Editor\n")
    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Editor 已關閉")
        server.server_close()

async def init_ground_truth(bench_dir, graph, filter_str="", overwrite=False, cv_override=None):
    """掃描 benchmarks/ 裡的 PDF，跑 pipeline，啟動 Web Editor"""
    pdf_files = glob.glob(os.path.join(bench_dir, "*.pdf"))
    if filter_str:
        pdf_files = [f for f in pdf_files if filter_str.lower() in os.path.basename(f).lower()]

    if not pdf_files:
        print("❌ benchmarks/ 找不到任何 PDF 檔案")
        return

    from core.archiver import create_run_dir, archive_item
    # 此處保留，改由外部傳入 run_dir
    print("🧹 預跑前已清空 crops/ 快取資料夾")

    all_gt = []  # (json_path, gt_data)

    for pdf_path in sorted(pdf_files):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = os.path.join(bench_dir, f"{base_name}.json")

        if os.path.exists(json_path) and not overwrite:
            print(f"⏭️  跳過 {base_name}.pdf (JSON 已存在，用 --init-all 覆蓋)")
            continue

        print(f"\n🔄 預跑 {base_name}.pdf ...")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        cv_params = cv_override or {
            "dilation_iterations": 2,
            "min_area": 100000,
            "padding_bottom": 160,
            "hough_threshold": 95,
            "enable_decomp": True,
            "voting_rounds": 1
        }

        t0 = time.time()
        try:
            result = await graph.ainvoke({
                "pdf_bytes": pdf_bytes, "page_num": 0,
                "task_id": None, "cv_params": cv_params
            })
        except Exception as e:
            print(f"❌ {base_name}.pdf pipeline 失敗: {e}")
            continue
        elapsed = round(time.time() - t0, 1)

        final = result.get("final_output", {})
        aligned = final.get("aligned_beams", [])
        all_beams = aligned
        
        print(f"✅ Pipeline 完成: {len(aligned)} 單梁 = {len(all_beams)} beams ({elapsed}s)")

        gt_fields = [
            "beam_id", "dimensions",
            "top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
            "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right",
            "stirrups_left", "stirrups_middle", "stirrups_right",
            "face_bars",
            "lap_length_top_left", "lap_length_top_right",
            "lap_length_bottom_left", "lap_length_bottom_right"
        ]
        clean_beams = []
        for b in all_beams:
            clean = {}
            for fld in gt_fields:
                val = b.get(fld, "" if fld not in LIST_FIELDS else [])
                clean[fld] = val
            clean_beams.append(clean)

        gt_data = {
            "_comment": f"由 --init 自動產生 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {elapsed}s)",
            "pdf_file": os.path.basename(pdf_path),
            "page_num": 0,
            "expected_beam_count": len(clean_beams),
            "expected_split_count": len(clean_beams),
            "cv_params": cv_params,
            "beams": clean_beams
        }

        # 先存一份原始版本到 benchmarks/ 供修改
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)
        print(f"📄 已儲存初始版本: {json_path}")
        
        # 自動存入 finished 封存區
        if getattr(graph, "run_dir", None):
            archive_item(graph.run_dir, base_name + ".pdf", pdf_bytes, gt_data)

        all_gt.append((json_path, gt_data))

    # 開啟最後一個的 Web Editor
    if all_gt:
        json_path, gt_data = all_gt[-1]
        _start_editor_server(gt_data, json_path)
    else:
        print("沒有需要處理的 PDF。")

# ================================================================
# 主程式
# ================================================================
async def main():
    parser = argparse.ArgumentParser(description="AI-PDF Benchmark Runner")
    parser.add_argument("--init", action="store_true", help="預跑模式：產生 ground truth 草稿")
    parser.add_argument("--init-all", action="store_true", help="預跑模式 (覆蓋既有 JSON)")
    parser.add_argument("--edit", type=str, default="", help="直接開啟指定 JSON 的 Web Editor (不跑 pipeline)")
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--voting", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="不自動開啟瀏覽器")
    args = parser.parse_args()

    bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)

    # === --edit 模式: 直接開 Editor (不跑 pipeline) ===
    if args.edit:
        edit_path = os.path.join(bench_dir, args.edit)
        if not edit_path.endswith(".json"):
            edit_path += ".json"
        if not os.path.exists(edit_path):
            print(f"❌ 找不到 {edit_path}")
            return
        with open(edit_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        _start_editor_server(gt_data, edit_path)
        return

    # === --init 模式 ===
    if args.init or args.init_all:
        from core.archiver import create_run_dir
        run_dir = create_run_dir("預跑")
        graph = build_graph()
        graph.run_dir = run_dir # 黑魔法把屬性偷傳過去
        await init_ground_truth(bench_dir, graph, filter_str=args.filter, overwrite=args.init_all)
        return

    # === 正常評測模式 ===
    json_files = [f for f in glob.glob(os.path.join(bench_dir, "*.json"))
                  if not os.path.basename(f).startswith("_")]
    if args.filter:
        json_files = [f for f in json_files if args.filter.lower() in os.path.basename(f).lower()]

    if not json_files:
        print("❌ benchmarks/ 找不到 ground truth JSON。")
        print("   提示：先跑 python benchmark_runner.py --init 讓 AI 預填")
        return

    # 執行前強制清空舊有的 crops 快取，避免被上一次的殘留污染
    import shutil
    crops_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crops")
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir, ignore_errors=True)
    os.makedirs(crops_dir, exist_ok=True)

    print(f"🔍 找到 {len(json_files)} 個測試案例")
    from core.archiver import create_run_dir, archive_item, archive_report
    run_dir = create_run_dir("評測")
    graph = build_graph()
    reports = []

    for json_path in sorted(json_files):
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        pdf_name = gt.get("pdf_file", "")
        pdf_path = os.path.join(bench_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"⚠️  跳過 {os.path.basename(json_path)}: 找不到 {pdf_name}")
            continue
        if "cv_params" not in gt: gt["cv_params"] = {}
        if args.voting > 1:
            gt["cv_params"]["voting_rounds"] = args.voting

        print(f"\n🚀 評測: {pdf_name} ({len(gt.get('beams',[]))} beams)...")
        try:
            r = await evaluate_single(pdf_path, gt, graph)
            reports.append(r)
            
            # 把當下的 crop/debug_full_pdf 等存進 finished
            with open(pdf_path, "rb") as f:
                pdf_bytes_tmp = f.read()
            archive_item(run_dir, pdf_name, pdf_bytes_tmp, r)
            
        except Exception as e:
            print(f"❌ {pdf_name} 失敗: {e}")
            import traceback; traceback.print_exc()

    if not reports:
        print("❌ 沒有成功完成任何評測。"); return

    print_report(reports, verbose=args.verbose)

    # 產生 HTML 報告
    results_dir = os.path.join(bench_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(results_dir, f"report_{ts}.html")
    html_content = generate_html_report(reports, voting_rounds=args.voting)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📊 HTML 報告已產生: {html_path}")
    archive_report(run_dir, html_path)

    # 同時儲存 JSON
    json_result_path = os.path.join(results_dir, f"benchmark_{ts}.json")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "voting": args.voting,
                   "avg_accuracy": round(sum(r["overall_accuracy"] for r in reports)/len(reports),1),
                   "reports": reports}, f, ensure_ascii=False, indent=2)

    if not args.no_open:
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")

if __name__ == "__main__":
    asyncio.run(main())

