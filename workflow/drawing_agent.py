import json
import re
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.vector_extractor import VectorExtractor
from core.table_extractor import TableExtractor
from core.task_manager import update_task_progress, complete_task, fail_task

def _find_closest_beam_id_downwards(anchor_rect: list, texts_data: list) -> str:
    """依據給定框，由梁中心往下看，找回漏掉的梁編號"""
    if not anchor_rect or not texts_data:
        return ""
    x0, y0, x1, y1 = anchor_rect
    cx = (x0 + x1) / 2.0
    by = y1
    
    candidates = []
    for t in texts_data:
        tx0, ty0, tx1, ty1 = t["rect"]
        txt = t["text"].strip()
        
        # 只尋找 y 坐標位於圖片下緣 (允許 20px 寬容度交疊) 附近的文字
        if ty0 >= by - 20:
            tcx = (tx0 + tx1) / 2.0
            dx = abs(tcx - cx)
            dy = ty0 - by
            
            # 若距離下方超過 300 像素，可能跳到別的區域了
            if dy > 300:
                continue
                
            # 水平必須靠近中心線 (寬容度 100 像素 或 與中心重疊)
            if dx < 100 or (tx0 - 50 <= cx <= tx1 + 50):
                # 只有帶有英文字母或數字的文字才像是編號或尺寸
                if re.search(r'[a-zA-Z0-9]', txt):
                    # 計算總距離分數 (優先找 y 軸最靠近的，其次才是 x 軸偏移)
                    score = dy + (dx * 0.3)
                    candidates.append((score, txt))
                    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        # 回傳最近的那一個文字
        return candidates[0][1]
    return ""

class GraphState(TypedDict):
    pdf_bytes: bytes
    page_num: int
    task_id: Optional[str]
    cv_params: Dict[str, Any]
    vector_data: Dict[str, Any]
    table_markdown: str # 這裡在 Phase 2 被改為裝載純 JSON 字串
    final_output: Dict[str, Any]
    confidence_scores: Dict[str, float]

async def extract_vectors_node(state: GraphState):
    tid = state.get("task_id")
    if tid:
        update_task_progress(tid, 10, "[Phase 1] PyMuPDF 解析啟動：提取底層幾何座標...")
        
    print("[Agent] 正在提取 PDF 向量幾何，並執行 OpenCV 自適應尋邊...")
    extractor = VectorExtractor(state["pdf_bytes"])
    data = extractor.extract_page_data(state["page_num"])
    
    if tid:
        update_task_progress(tid, 30, f"[Phase 1] OpenCV 降噪裁切執行中：已為大圖分析出潛力區塊...")
        
    # 執行 Phase 3 的 OpenCV 形狀辨識拆框演算法
    cv_params = state.get("cv_params", {})
    cv_bboxes, cv_metrics = extractor.extract_opencv_bboxes(state["page_num"], cv_params)
    data["cv_bboxes"] = cv_bboxes
    data["cv_metrics"] = cv_metrics
    
    # == 計算進階切割信心分數 ==
    num_bboxes = len(cv_bboxes)
    crop_conf = 100.0
    noise_rate = cv_metrics.get("noise_drop_rate", 0)
    nms_rate = cv_metrics.get("nms_drop_rate", 0)
    
    if num_bboxes == 0:
        crop_conf = 0.0
    else:
            
        # 根據 NMS 重疊丟棄率扣分 (丟太多代表高度重疊，通常是參數沒調好或切割破碎)
        if nms_rate > 20:
            crop_conf -= (nms_rate - 20) * 1.5  # 嚴重扣分
            
        # 3. 根據數量防呆
        if num_bboxes < 2:
            crop_conf -= 40.0
        elif num_bboxes > 60:
            crop_conf -= (num_bboxes - 60) * 2.0
            
    crop_conf = max(0.0, min(100.0, crop_conf))
        
    scores = state.get("confidence_scores") or {}
    scores["crop_confidence"] = round(crop_conf, 1)
    scores["noise_drop_rate"] = noise_rate
    scores["nms_drop_rate"] = nms_rate
    
    print(f"[Agent] OpenCV 成功框出 {num_bboxes} 個區域。切割信心: {crop_conf}% (NMS丟棄:{cv_metrics.get('nms_drop_rate')}%)")
    return {"vector_data": data, "confidence_scores": scores}

async def extract_tables_node(state: GraphState):
    tid = state.get("task_id")
    cv_bboxes = state["vector_data"].get("cv_bboxes", [])
    cv_metrics = state["vector_data"].get("cv_metrics", {})
    
    if tid:
        update_task_progress(tid, 50, f"[Phase 2] 打包 {len(cv_bboxes)} 張微觀圖塊，正在發送至 Gemini 2.5 神經網路進行多模態辨識 (約需15-40秒)...")
        
    def progress_callback(msg: str):
        if tid:
            update_task_progress(tid, 60, msg)

    print("[Agent] 正在組裝多模態輸入，呼叫 Gemini Vision 解構 JSON...")
    extractor = TableExtractor()
    # 將 OpenCV 框傳給 LLM 進行精準片段裁切閱讀
    json_str = await extractor.extract_tables(
        state["pdf_bytes"], 
        state["page_num"], 
        cv_bboxes, 
        progress_cb=progress_callback,
        cv_metrics=cv_metrics
    )
    return {"table_markdown": json_str}

async def llm_reasoning_node(state: GraphState):
    tid = state.get("task_id")
    if tid:
        update_task_progress(tid, 85, "[Phase 3] 雙向定位啟動：將 AI 語意映射回實體坐標...")
        
    print("[Agent] [Phase 3] OpenCV 實體定錨啟動：映射圖像片段 ID 回絕對物理坐標...")
    
    try:
        gemini_data = json.loads(state["table_markdown"])
        beams_list = gemini_data.get("beams", [])
    except Exception as e:
        beams_list = []
        print(f"[異常] JSON 解析失敗: {e}")

    cv_bboxes = state["vector_data"].get("cv_bboxes", [])
    texts_data = state["vector_data"].get("texts_data", [])
    parent_count = state["vector_data"].get("cv_metrics", {}).get("parent_count", len(cv_bboxes))
    
    aligned_entities = []
    split_beams = []
    
    for idx, beam in enumerate(beams_list):
        anchor_rect = None
        crop_idx = beam.get("crop_index")
        is_split = False
        
        # 1. 優先使用 Phase 3 的 Micro-Vision 裁塊索引定錨
        if crop_idx is not None and isinstance(crop_idx, int) and 1 <= crop_idx <= len(cv_bboxes):
            # Gemini 輸出的是 1-based (片段 1, 片段 2... )
            anchor_rect = cv_bboxes[crop_idx - 1]
            beam["spatial_anchor_rect_x_y"] = anchor_rect
            
            if crop_idx > parent_count:
                is_split = True
                beam["alignment_status"] = f"SUCCESS: Anchored to Split Child Frame #{crop_idx}."
            else:
                beam["alignment_status"] = f"SUCCESS: Anchored perfectly to Micro-Vision Crop Frame #{crop_idx}."
        else:
            beam["spatial_anchor_rect_x_y"] = None
            beam["alignment_status"] = "WARNING: Missing or invalid crop_index from LLM JSON."
            
        b_id = beam.get("beam_id", "") or beam.get("id", "") or beam.get("name", "")
        
        # == 反推估核心邏輯 ==
        # 如果是分離出來的子梁片段，且長得像沒有自己編號的「邊緣跨」，我們應該用其「母圖」的座標去往下找！
        search_anchor = anchor_rect
        if not b_id and is_split:
            child_to_parent_map = state["vector_data"].get("cv_metrics", {}).get("child_to_parent_map", {})
            child_idx_in_decomposed = crop_idx - 1 - parent_count
            parent_idx = child_to_parent_map.get(str(child_idx_in_decomposed)) or child_to_parent_map.get(child_idx_in_decomposed)
            if parent_idx is not None and 0 <= parent_idx < len(cv_bboxes):
                search_anchor = cv_bboxes[parent_idx]
                beam["alignment_status"] += f" (Using Parent #{parent_idx+1} for ID search)"
                
        if not b_id and search_anchor and texts_data:
            recovered_id = _find_closest_beam_id_downwards(search_anchor, texts_data)
            if recovered_id:
                b_id = recovered_id
                beam["alignment_status"] += f" [Reverse Estimated ID: {b_id}]"

        if not b_id:
            beam["beam_id"] = f"未命名_{idx+1}"
        else:
            beam["beam_id"] = b_id
            
        if is_split:
            split_beams.append(beam)
        else:
            aligned_entities.append(beam)

    # === Phase 4: 同名解抉與去重 (針對母圖) ===
    # 取消了 Phase 4 對所有資料的擴充，回歸單純對母圖同名狀態去重，保證不塞冗餘子圖
    import uuid
    from collections import defaultdict
    beam_groups = defaultdict(list)
    
    for beam in aligned_entities:
        b_id = beam.get("beam_id", "")
        if not b_id or b_id.startswith("未命名"):
            beam_groups[f"未命名_{uuid.uuid4().hex[:6]}"].append(beam)
        else:
            beam_groups[b_id].append(beam)
            
    def calculate_richness(b: dict) -> float:
        score = 0.0
        if b.get("dimensions") or b.get("尺寸"): score += 20
        top = (b.get("top_main_bars_left") or []) + (b.get("top_main_bars_mid") or []) + (b.get("top_main_bars_right") or [])
        bot = (b.get("bottom_main_bars_left") or []) + (b.get("bottom_main_bars_mid") or []) + (b.get("bottom_main_bars_right") or [])
        if top: score += 15
        if bot: score += 15
        stirrup_info = (b.get("stirrups_left") or "") + (b.get("stirrups_middle") or "") + (b.get("stirrups_right") or "")
        if stirrup_info: score += 15
        score += b.get("self_confidence", 0) * 0.1
        return score
        
    final_merged_entities = []
    
    for b_id, group in beam_groups.items():
        if len(group) == 1:
            final_merged_entities.append(group[0])
        else:
            group.sort(key=calculate_richness, reverse=True)
            best_beam = group[0]
            best_beam["alignment_status"] += f" [Phase 4: Multi-crop deduplicated]"
            final_merged_entities.append(best_beam)
            
    aligned_entities = final_merged_entities
    # =========================================================================

    beam_count = len(aligned_entities)
    
    # == 計算進階辨識信心分數 ==
    recog_conf = 100.0
    total_found = beam_count + len(split_beams)
    
    if total_found == 0:
        recog_conf = 0.0
    else:
        total_score = 0.0
        # 打分數包含兩者
        for beam in aligned_entities + split_beams:
            beam_score = 10.0 # 基礎分
            
            # 1. 有無梁編號 (非常重要)
            b_id = beam.get("beam_id", "")
            if b_id and not str(b_id).startswith("未命名"):
                beam_score += 40.0
                
            # 2. 有無梁尺寸 (寬 x 高)
            if beam.get("dimensions") or beam.get("尺寸"):
                beam_score += 20.0
                
            # 3. 有無上層與下層主筋細節
            top_bars = (beam.get("top_main_bars_left") or []) + (beam.get("top_main_bars_mid") or []) + (beam.get("top_main_bars_right") or [])
            bot_bars = (beam.get("bottom_main_bars_left") or []) + (beam.get("bottom_main_bars_mid") or []) + (beam.get("bottom_main_bars_right") or [])
            
            if len(top_bars) > 0:
                beam_score += 10.0
            if len(bot_bars) > 0:
                beam_score += 10.0
                
            # 4. 有無箍筋細節
            stirrups_info = (beam.get("stirrups_left") or "") + (beam.get("stirrups_middle") or "") + (beam.get("stirrups_right") or "")
            if stirrups_info:
                beam_score += 10.0
                
            # 4. 把 LLM 主觀的信心指數考量進去 (預設為 100)
            self_conf = beam.get("self_confidence", 100)
            if self_conf < 80:
                # LLM 對這根梁感到沒自信，扣除相近的倍率分
                beam_score -= (80.0 - self_conf) * 0.5
                
            total_score += min(100.0, max(0.0, beam_score))
            
        recog_conf = total_score / total_found
        
    scores = state.get("confidence_scores") or {}
    scores["recognition_confidence"] = round(recog_conf, 1)

    result = {
        "status": "success",
        "message": f"Phase 3 定位完成：成功將神經網路配筋物件映射回影像物理座標。共偵測到 {beam_count} 個主梁與 {len(split_beams)} 個附屬子梁片段。",
        "aligned_beams": aligned_entities,
        "split_beams": split_beams,
        "raw_json_string": state["table_markdown"],
        "confidence_scores": scores,
        "extracted_summary": {
            "total_beams_found": beam_count,
            "total_split_beams_found": len(split_beams),
            "total_vectors_in_page": state["vector_data"].get("vector_count", 0)
        }
    }
    
    if tid:
        update_task_progress(tid, 100, "分析圓滿完成！")
        
    return {"final_output": result}

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("extract_vectors", extract_vectors_node)
    workflow.add_node("extract_tables", extract_tables_node)
    workflow.add_node("llm_reason", llm_reasoning_node)
    
    workflow.set_entry_point("extract_vectors")
    workflow.add_edge("extract_vectors", "extract_tables")
    workflow.add_edge("extract_tables", "llm_reason")
    workflow.add_edge("llm_reason", END)
    
    return workflow.compile()
