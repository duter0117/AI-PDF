import os
import io
import re
import fitz
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from core.crop_context import CropContext, DetectedLine, ObjectStatus

# Suppress the deprecation warning for google.generativeai
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List
from collections import Counter

# 定義嚴格的 JSON 輸出規格
class BeamDetail(BaseModel):
    beam_id: str = Field(description="構件編號或名稱，例如 B1F FWB1。通常出現在【正下方】附近。")
    dimensions: str = Field(description="構件外觀尺寸，例如 100x380。通常伴隨在梁編號附近。")
    
    top_main_bars_left: List[str] = Field(description="上層主筋數量(左端)。格式為 N-#S (如 5-#8)，絕不含@符號。若有多排請存入陣列。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    top_main_bars_mid: List[str] = Field(description="上層主筋數量(中央)。格式為 N-#S。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    top_main_bars_right: List[str] = Field(description="上層主筋數量(右端)。格式為 N-#S。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    bottom_main_bars_left: List[str] = Field(description="下層主筋數量(左端)。格式為 N-#S。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    bottom_main_bars_mid: List[str] = Field(description="下層主筋數量(中央)。格式為 N-#S。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    bottom_main_bars_right: List[str] = Field(description="下層主筋數量(右端)。格式為 N-#S。若無內容或判斷為雜訊，絕對不要留空，請輸出 ['LLM沒有東西']。")
    
    stirrups_left: str = Field(description="箍筋數量(左側)。格式含@符號。若無內容或判斷為雜訊，絕對不要留空，請輸出字串「LLM沒有東西」。")
    stirrups_middle: str = Field(description="箍筋數量(中央)。如果整支梁只有標註一個箍筋，請統一填入此欄位。若無內容或判斷為雜訊，絕對不要留空，請輸出「LLM沒有東西」。")
    stirrups_right: str = Field(description="箍筋數量(右側)。格式含@符號。若無內容或判斷為雜訊，絕對不要留空，請輸出字串「LLM沒有東西」。")
    face_bars: str = Field(description="腰筋(又稱側邊鋼筋)，通常帶有 E.F. 字樣。若無內容或判斷為雜訊，絕對不要留空，請輸出「LLM沒有東西」。")
    
    lap_length_top_left: str = Field(description="上層鋼筋搭接長度(左)，純數字。若無內容或判斷為雜訊，請輸出「LLM沒有東西」。")
    lap_length_top_right: str = Field(description="上層鋼筋搭接長度(右)，純數字。若無內容或判斷為雜訊，請輸出「LLM沒有東西」。")
    lap_length_bottom_left: str = Field(description="下層鋼筋搭接長度(左)，純數字。若無內容或判斷為雜訊，請輸出「LLM沒有東西」。")
    lap_length_bottom_right: str = Field(description="下層鋼筋搭接長度(右)，純數字。若無內容或判斷為雜訊，請輸出「LLM沒有東西」。")
    
    self_confidence: int = Field(description="請給出本次辨識的信心分數(0-100)。畫面清晰完整則為60-100，有雜訊或字跡難辨則降低。")
    note: str = Field(description="若圖中有任何文字無法歸類(如工法說明)請抄錄至此。若無請填寫「LLM沒有東西」。")

class BeamList(BaseModel):
    beams: List[BeamDetail] = Field(description="此單圖片中所解析出的配筋詳圖清單 (若圖中全無配筋資訊，請回傳空陣列)")


class TableExtractor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        self._ocr = None          # Lazy init
        self._ocr_available = None # None = 未偵測, True/False = 已偵測
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # 修正 Google 最新的模型命名，沒有 2.5 lite。
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.model = genai.GenerativeModel(model_name)

    @staticmethod
    def _is_beam_id(text):
        """嚴格辨識梁編號。只有明確匹配工程梁命名慣例才算。"""
        text = text.strip()
        if len(text) < 2:
            return False
        if '@' in text or '#' in text:
            return False
        # 純數字 / 單字母+純數字 (如 195, F11-11, E12) → 不是梁編號，是軸線或尺寸
        if re.match(r'^[A-Za-z]?\d[\d\-]*$', text):
            return False
        # 必須以常見的梁前綴開頭才算
        # B (beam), G (girder), FB/FG (floor beam/girder), WB (wall beam),
        # FWB (floor wall beam), RB (roof beam), CB, SB, 或 樓層前綴如 B4F, R1F 等
        beam_prefix = re.match(
            r'^(F?W?[BGCS]|FB|FG|RB|CB|SB|[BR]\d+F)\s*',
            text, re.IGNORECASE
        )
        if beam_prefix:
            return True
        return False

    # ================================================================
    # RapidOCR 前置文字提取 (輕量 ONNX Runtime 版)
    # ================================================================
    def _init_ocr(self):
        """延遲初始化 RapidOCR（僅第一次呼叫時載入模型）"""
        if self._ocr_available is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._ocr_available = True
            print("[OCR] RapidOCR 初始化成功")
        except ImportError:
            self._ocr_available = False
            print("[OCR] RapidOCR 未安裝，跳過 OCR 前置處理 (pip install rapidocr_onnxruntime)")
        except Exception as e:
            self._ocr_available = False
            print(f"[OCR] RapidOCR 初始化失敗: {e}")

    def _run_ocr(self, img: Image.Image) -> tuple[str, list]:
        """對 PIL Image 執行 RapidOCR，回傳 (格式化文字提示, 原始資料列表)"""
        self._init_ocr()
        if not self._ocr_available or not self._ocr:
            return "", []
        try:
            import numpy as np
            img_array = np.array(img.convert("RGB"))
            h, w = img_array.shape[:2]
            result, _ = self._ocr(img_array)
            if not result:
                return "", []
            
            raw_items = []
            texts = []
            for bbox, text, conf in result:
                if conf < 0.3 or not text.strip():
                    continue
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cx = sum(xs) / 4.0
                cy = sum(ys) / 4.0
                
                # 計算相對位置
                rx = cx / w
                ry = cy / h
                
                # 九宮格判定
                if rx < 0.33: pos_x = "左"
                elif rx > 0.67: pos_x = "右"
                else: pos_x = "中"
                
                if ry < 0.33: pos_y = "上"
                elif ry > 0.67: pos_y = "下"
                else: pos_y = "中"

                pos_label = ""
                if pos_x == "中" and pos_y == "中": pos_label = "正中央"
                elif pos_x == "中": pos_label = f"正{pos_y}方"
                elif pos_y == "中": pos_label = f"正{pos_x}方"
                else: pos_label = f"{pos_x}{pos_y}方"
                
                clean_text = text.strip()
                # 專門防護 OCR 將長引出線誤判為「減號」的情況 (例如 "-16-#4@15" 變成 "16-#4@15")
                # 這也順便清除了周遭的雜點符號。只清頭尾，不影響單字裡面的 "-" (如 B1-6)。
                clean_text = clean_text.strip('-_.= —–~')
                
                # 自動修正 OCR 的常見識別錯誤：把 "=" 或 "=#" 誤認為 "-#"
                import re
                clean_text = re.sub(r'(\d+)=#?(\d+)', r'\1-#\2', clean_text)
                
                if not clean_text:
                    continue
                
                # 防止 OCR 抽風產生重複辨識的字 (例如兩張重疊的框)
                text_key = f"{clean_text}_{pos_label}"
                if text_key in [item.get("_dup_key") for item in raw_items]:
                    continue
                
                raw_items.append({
                    "text": clean_text, "conf": conf,
                    "cx": cx, "cy": cy, "rx": rx, "ry": ry,
                    "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
                    "pos_label": pos_label, "is_extreme": False,
                    "_dup_key": text_key
                })
                texts.append(f'  "{clean_text}" @ {pos_label} (信心:{conf:.0%})')
                
            return "\n".join(texts) if texts else "", raw_items
        except Exception as e:
            print(f"[OCR] 發生錯誤: {e}")
            return "", []

    def _refresh_ocr_hint(self, ctx) -> str:
        """物理裁切後，根據實體梁框與柱線，重新計算精確的物理九宮格方位提示"""
        w, h = ctx.img.width, ctx.img.height
        if w == 0 or h == 0: return ""
        
        # 缺柱線時：以水平梁邊緣(h_beam_edge)的端點作為替代的左右邊界
        left_col = ctx.left_col
        right_col = ctx.right_col
        if left_col is None or right_col is None:
            h_edges = [l for l in ctx.lines if l.kind == "h_beam_edge"]
            if h_edges:
                min_h_x = min(l.x for l in h_edges)
                max_h_x = max((l.x + l.w) for l in h_edges)
                if left_col is None: left_col = min_h_x
                if right_col is None: right_col = max_h_x

        left_bound = left_col if left_col is not None else 0
        right_bound = right_col if right_col is not None else w
        top_bound = ctx.beam_top if ctx.beam_top is not None else h / 3.0
        bottom_bound = ctx.beam_bottom if ctx.beam_bottom is not None else h * (2.0/3.0)
        
        beam_len = right_bound - left_bound
        # 梁長度範圍中間 40% (即左右各留 30%)
        mid_x_start = left_bound + beam_len * 0.3
        mid_x_end = right_bound - beam_len * 0.3
        
        ctx.grid_mid_x_start = mid_x_start
        ctx.grid_mid_x_end = mid_x_end
        ctx.grid_top_bound = top_bound - 5
        ctx.grid_bottom_bound = bottom_bound + 5
            
        import re
        
        # 垂直疊加鋼筋的物理合併 (雙排/三排主筋合併)
        def is_rebar(txt):
            return bool(re.match(r'^\d+-#\d+$', txt.strip()))

        merged_items = []
        skip_indices = set()
        for i in range(len(ctx.ocr_items)):
            if i in skip_indices: continue
            
            item1 = ctx.ocr_items[i]
            if not is_rebar(item1["text"]):
                merged_items.append(item1)
                continue
                
            y_region1 = "top" if item1["cy"] < top_bound else ("bottom" if item1["cy"] > bottom_bound else "mid")
            
            group = [item1]
            for j in range(i+1, len(ctx.ocr_items)):
                if j in skip_indices: continue
                item2 = ctx.ocr_items[j]
                if not is_rebar(item2["text"]): continue
                
                y_region2 = "top" if item2["cy"] < top_bound else ("bottom" if item2["cy"] > bottom_bound else "mid")
                
                # 同一側 (皆為上或皆為下) 且 X 方向誤差小於 20px
                if y_region1 == y_region2 and abs(item1["cx"] - item2["cx"]) <= 20:
                    group.append(item2)
                    skip_indices.add(j)
                    
            if len(group) == 1:
                merged_items.append(item1)
            else:
                # 按照 Y 座標由上到下排序，組合字串
                group.sort(key=lambda x: x["cy"])
                merged_text = ", ".join([g["text"].strip() for g in group])
                avg_cx = sum(g["cx"] for g in group) / len(group)
                avg_cy = sum(g["cy"] for g in group) / len(group)
                min_conf = min(g.get("conf", 1.0) for g in group)
                
                merged_items.append({
                    "text": merged_text,
                    "conf": min_conf,
                    "cx": avg_cx,
                    "cy": avg_cy,
                    "min_x": min(g["min_x"] for g in group),
                    "max_x": max(g["max_x"] for g in group),
                    "min_y": min(g["min_y"] for g in group),
                    "max_y": max(g["max_y"] for g in group),
                })
                
        ctx.ocr_items = merged_items
        
        texts = []
        for item in ctx.ocr_items:
            cx, cy = item["cx"], item["cy"]
            
            # X 軸分段
            if cx < mid_x_start: pos_x = "左"
            elif cx > mid_x_end: pos_x = "右"
            else: pos_x = "中"
            
            # Y 軸分段 (加寬容差，主筋有時會緊貼或是壓到邊界線，將判定範圍延伸進入梁內 10px)
            if cy < top_bound + 10: pos_y = "上"
            elif cy > bottom_bound - 10: pos_y = "下"
            else: pos_y = "中"
            
            if pos_x == "中" and pos_y == "中": pos_label = "正中央"
            elif pos_x == "中": pos_label = f"正{pos_y}方"
            elif pos_y == "中": pos_label = f"正{pos_x}方"
            else: pos_label = f"{pos_x}{pos_y}方" # 例如: 左上方
            
            # 將運算後的真實地理位置寫回 item 中，供下游的 OCR-First 規則引擎使用
            item["pos_label"] = pos_label
            
            clean_text = item["text"]
            
            from core.ocr_field_assigner import classify_text, FormatType
            fmt = classify_text(clean_text)
            
            # 預設提示後綴
            hint = ""
            
            # 避免錯誤暗示：若 OCR 辨識出無法歸類的亂碼(如 F11-#11)，與其完全隱藏，
            # 等於沒收了 LLM 的參考線索。我們改為加上強烈警告，讓 LLM 知道這是可能有問題的字串。
            if fmt == FormatType.UNKNOWN:
                hint = " [⚠️格式異常/疑似雜訊]"
                
            is_lap = re.match(r'^(L|La|Ld)?\s*[=≈]?\s*\d{2,4}\s*(cm|mm)?$', clean_text.strip(), re.IGNORECASE)
            is_pure_num = re.match(r'^\d{2,4}$', clean_text.strip())
            # 放寬條件：只要位置偏上方或下方，且是純數字，就給出暗示
            if item.get("_rebar_proximity"):
                # 此數值疑似鋼筋部分辨識結果，不暗示為搭接長度
                hint = " [⚠️疑似鋼筋號數殘片，請從圖面確認]"
            elif is_lap or (is_pure_num and ("上" in pos_label or "下" in pos_label)):
                hint = " [🌟搭接長度]"
                
            conf = item.get("conf", 1.0)
            texts.append(f'  "{clean_text}" @ {pos_label}{hint} (信心:{conf:.0%})')
            # 儲存位置資訊供報告使用
            item["pos_label"] = pos_label
            
        return "\n".join(texts)
        
    def _run_sub_crops_ocr(self, ctx) -> None:
        """
        將已定位物理邊界的單跨梁，重疊裁切為 9 張小圖。
        對每張小圖進行增強處理 (CLAHE + Sharpen) 並反覆呼叫 OCR。
        合併掃描到的新字元(或高信心字元)回到 ctx.ocr_items。
        """
        img = ctx.img
        w, h = img.width, img.height
        
        left_col = ctx.left_col
        right_col = ctx.right_col
        if left_col is None or right_col is None:
            h_edges = [l for l in ctx.lines if l.kind == "h_beam_edge"]
            if h_edges:
                min_h_x = min(l.x for l in h_edges)
                max_h_x = max((l.x + l.w) for l in h_edges)
                if left_col is None: left_col = min_h_x
                if right_col is None: right_col = max_h_x

        left_bound = left_col if left_col is not None else 0
        right_bound = right_col if right_col is not None else w
        top_bound = ctx.beam_top if ctx.beam_top is not None else h / 3.0
        bottom_bound = ctx.beam_bottom if ctx.beam_bottom is not None else h * (2.0/3.0)
        
        beam_len = right_bound - left_bound
        mid_x_start = left_bound + beam_len * 0.3
        mid_x_end = right_bound - beam_len * 0.3
        
        y_splits = [0, top_bound, bottom_bound, h]
        x_splits = [0, mid_x_start, mid_x_end, w]
        
        pad = 30 # 重疊 Padding (Text Slicing 防護)
        
        multi_items = []
        import cv2
        import numpy as np
        
        for i in range(3): # Y
            for j in range(3): # X
                x1 = max(0, int(x_splits[j]) - pad)
                y1 = max(0, int(y_splits[i]) - pad)
                x2 = min(w, int(x_splits[j+1]) + pad)
                y2 = min(h, int(y_splits[i+1]) + pad)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                sub_img = img.crop((x1, y1, x2, y2))
                
                # ==== 影像增強: CLAHE + 銳化 ====
                cv_sub = cv2.cvtColor(np.array(sub_img), cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced_sub = clahe.apply(cv_sub)
                
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced_sub = cv2.filter2D(enhanced_sub, -1, kernel)
                
                pil_enhanced = Image.fromarray(cv2.cvtColor(enhanced_sub, cv2.COLOR_GRAY2RGB))
                
                # ==== OCR 掃描 ====
                _, raw_sub_items = self._run_ocr(pil_enhanced)
                
                # 還原絕對座標
                for item in raw_sub_items:
                    item["cx"] += x1
                    item["cy"] += y1
                    item["min_x"] += x1
                    item["max_x"] += x1
                    item["min_y"] += y1
                    item["max_y"] += y1
                    multi_items.append(item)
                    
        # 重疊合併字串 (去重與碎片過濾)
        def compute_overlap(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0: return 0.0, 0.0
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            iom = interArea / float(min(boxAArea, boxBArea)) # Intersection over Minimum
            return iou, iom
            
        final_items = list(ctx.ocr_items)
        
        from core.ocr_field_assigner import classify_text, FormatType
        for m_item in multi_items:
            m_box = (m_item["min_x"], m_item["min_y"], m_item["max_x"], m_item["max_y"])
            m_cx, m_cy = m_item["cx"], m_item["cy"]
            m_text = m_item["text"].strip().replace(" ", "")
            
            matched = False
            for f_idx, f_item in enumerate(final_items):
                f_box = (f_item["min_x"], f_item["min_y"], f_item["max_x"], f_item["max_y"])
                f_text = f_item["text"].strip().replace(" ", "").replace(",", "")
                
                # ============================================================
                # 核心碎片判定：sub-crop 文字的中心點是否落在原始 OCR 文字的 bbox 內
                # 改用 2px 寬容，嚴格審查，避免將上下兩排緊靠的新文字 (例如 11-#11, 14-#11) 誤殺
                # ============================================================
                pad_check = 2
                center_inside = (
                    f_box[0] - pad_check <= m_cx <= f_box[2] + pad_check and
                    f_box[1] - pad_check <= m_cy <= f_box[3] + pad_check
                )
                
                # 備用：IoU / IoM 傳統重疊判定
                iou, iom = compute_overlap(m_box, f_box)
                
                # 純物理空間重疊判定：文字內容是否為子字串等語義條件已被移除
                # 只要高度重疊，就進入留優汰劣機制
                is_physical_overlap = iou > 0.3 or iom > 0.5
                
                if is_physical_overlap:
                    matched = True
                    
                    # 判斷：是碎片還是更好的辨識結果？
                    # 碎片 = 文字比原始短，或格式更差 → 直接丟棄
                    # 更正 = 文字合規且原始不合規，或同格式但信心大勝 → 取代
                    fmt_f = classify_text(f_item["text"])
                    fmt_m = classify_text(m_item["text"])
                    
                    should_replace = False
                    if fmt_m != FormatType.UNKNOWN and fmt_f == FormatType.UNKNOWN:
                        should_replace = True
                    elif fmt_m == fmt_f and m_item["conf"] > f_item["conf"] + 0.1:
                        should_replace = True
                    
                    if should_replace:
                        pass # print(f"  [Sub-Crop] 取代雜訊: '{f_item['text']}'({f_item['conf']:.2f}) -> '{m_item['text']}'({m_item['conf']:.2f})")
                        final_items[f_idx] = m_item
                    else:
                        reason = f"IoU={iou:.2f}/IoM={iom:.2f}"
                        pass # print(f"  [Sub-Crop] 碎片剔除({reason}): '{m_item['text']}' ∈ '{f_item['text']}'")
                    break
            
            if not matched:
                pass # print(f"  [Sub-Crop] 發現未見文字: '{m_item['text']}' ({m_item['conf']:.2f})")
                final_items.append(m_item)
        
        # === 同源去重 (Intra-pass Dedup) ===
        # 同一輪 OCR 可能同時檢測到 "14-#11" 和 "11"，它們的 bbox 高度重疊 (IoM ≈ 1.0)
        # 但 cross-pass NMS 不會處理同源項目，所以需要額外的清理
        dedup_remove = set()
        for i in range(len(final_items)):
            if i in dedup_remove:
                continue
            for j in range(i + 1, len(final_items)):
                if j in dedup_remove:
                    continue
                box_i = (final_items[i]["min_x"], final_items[i]["min_y"], final_items[i]["max_x"], final_items[i]["max_y"])
                box_j = (final_items[j]["min_x"], final_items[j]["min_y"], final_items[j]["max_x"], final_items[j]["max_y"])
                iou, iom = compute_overlap(box_i, box_j)
                if iom > 0.5:
                    # 高度重疊：保留格式更好的、或文字更長的
                    fmt_i = classify_text(final_items[i]["text"])
                    fmt_j = classify_text(final_items[j]["text"])
                    # 優先保留有明確格式的 (REBAR > STIRRUP > FACE_BAR > LAP_LENGTH > UNKNOWN)
                    FORMAT_RANK = {FormatType.REBAR: 5, FormatType.STIRRUP: 4, FormatType.FACE_BAR: 3, 
                                   FormatType.BEAM_ID: 3, FormatType.DIMENSION: 2, FormatType.LAP_LENGTH: 1, FormatType.UNKNOWN: 0}
                    rank_i = FORMAT_RANK.get(fmt_i, 0)
                    rank_j = FORMAT_RANK.get(fmt_j, 0)
                    if rank_i > rank_j:
                        dedup_remove.add(j)
                        pass # print(f"  [Intra-Dedup] 剔除碎片 '{final_items[j]['text']}' (被 '{final_items[i]['text']}' 包含, IoM={iom:.2f})")
                    elif rank_j > rank_i:
                        dedup_remove.add(i)
                        pass # print(f"  [Intra-Dedup] 剔除碎片 '{final_items[i]['text']}' (被 '{final_items[j]['text']}' 包含, IoM={iom:.2f})")
                    elif len(final_items[i]["text"]) >= len(final_items[j]["text"]):
                        dedup_remove.add(j)
                        pass # print(f"  [Intra-Dedup] 剔除碎片 '{final_items[j]['text']}' (被 '{final_items[i]['text']}' 包含, IoM={iom:.2f})")
                    else:
                        dedup_remove.add(i)
                        pass # print(f"  [Intra-Dedup] 剔除碎片 '{final_items[i]['text']}' (被 '{final_items[j]['text']}' 包含, IoM={iom:.2f})")
        
        if dedup_remove:
            final_items = [item for idx, item in enumerate(final_items) if idx not in dedup_remove]
        
        # DEBUG: 針對存活的短純數字，印出它與每個 REBAR 的 bbox 和 IoM
        for item in final_items:
            txt = item["text"].strip()
            if re.match(r'^\d{1,3}$', txt) and classify_text(txt) in (FormatType.LAP_LENGTH, FormatType.UNKNOWN):
                item_box = (item["min_x"], item["min_y"], item["max_x"], item["max_y"])
                pass # print(f"  [DEBUG-11] 存活純數字 '{txt}' bbox={item_box}")
                for other in final_items:
                    if other is item: continue
                    o_fmt = classify_text(other["text"].strip())
                    if o_fmt == FormatType.REBAR:
                        o_box = (other["min_x"], other["min_y"], other["max_x"], other["max_y"])
                        d_iou, d_iom = compute_overlap(item_box, o_box)
                        print(f"    vs REBAR '{other['text']}' bbox={o_box} → IoU={d_iou:.3f}, IoM={d_iom:.3f}")
        
        # === 近距離疑似殘片降信心 (Proximity Confidence Demotion) ===
        # 如果一個短數字出現在 REBAR 密集區域且字面上是鄰近 REBAR 的子字串，
        # 它極可能是 OCR 對某排鋼筋的部分辨識結果 (如 "14-#11" 只讀到 "11")。
        # 不剔除 (因為它代表「這裡還有一排鋼筋」的珍貴線索)，
        # 而是降低信心值，避免它被自動歸類為搭接長度，讓 LLM 從圖面做最終判定。
        for i, item in enumerate(final_items):
            txt = item["text"].strip()
            fmt = classify_text(txt)
            if fmt in (FormatType.LAP_LENGTH, FormatType.UNKNOWN) and re.match(r'^\d{1,3}$', txt):
                item_cx = item["cx"]
                item_cy = item["cy"]
                item_h = item["max_y"] - item["min_y"]
                
                for j, other in enumerate(final_items):
                    if i == j: continue
                    other_fmt = classify_text(other["text"].strip())
                    if other_fmt != FormatType.REBAR: continue
                    x_overlap = other["min_x"] <= item_cx <= other["max_x"]
                    other_h = other["max_y"] - other["min_y"]
                    y_dist = abs(item_cy - (other["min_y"] + other["max_y"]) / 2)
                    y_close = y_dist < max(item_h, other_h) * 2
                    
                    if x_overlap and y_close:
                        other_raw = other["text"].strip().replace(" ", "")
                        if txt in other_raw:
                            item["conf"] = 0.3  # 強制降低信心，送 LLM 覆核
                            item["_rebar_proximity"] = True  # 標記：疑似鋼筋殘片
                            print(f"  [Proximity-Demote] '{txt}' 疑似鋼筋部分辨識 (靠近 '{other['text']}'), 信心降至 30%")
                            break
                
        ctx.ocr_items = final_items

    # ================================================================
    # 物理柱線邊界定位 (OpenCV)
    # ================================================================
    def _detect_column_bounds(self, ctx) -> None:
        """給定 CropContext，進行物理形態學特徵萃取。
        結果直接寫入 ctx.beam_top, ctx.beam_bottom, ctx.left_col, ctx.right_col, ctx.lines 等。
        """
        img_enhanced = ctx.img
        ocr_items = ctx.ocr_items
        
        cv_img = cv2.cvtColor(np.array(img_enhanced), cv2.COLOR_RGB2GRAY)
        
        # 💡 【白化文字防干擾】: 將 OCR 偵測到的文字區域塗白
        for item in ocr_items:
            x1 = max(0, int(item["min_x"]) - 2)
            y1 = max(0, int(item["min_y"]) - 2)
            x2 = min(cv_img.shape[1], int(item["max_x"]) + 2)
            y2 = min(cv_img.shape[0], int(item["max_y"]) + 2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), 255, -1)

        _, binary = cv2.threshold(cv_img, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 1. 水平主梁結構
        # 縮小 kernel 以偵測極短的懸臂梁 (從 100 降至 40)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 將同 Y 座標 (±3px) 的碎片合併，取最長單一塊
        h_segments = []
        for cnt in h_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            h_segments.append((y, w, h, x))
        # 先以 Y 排序，再以 X 排序 (確保由上而下、由左而右)
        h_segments.sort(key=lambda s: (s[0], s[3]))
        
        # 分組：Y 座標相差趨近 且 X 軸斷層不大於 30px (被標註線穿過) 的視為同一條線並反覆接合
        y_groups = [(seg_y, seg_w, seg_h, 1, seg_x) for seg_y, seg_w, seg_h, seg_x in h_segments]
        
        changed = True
        while changed:
            changed = False
            new_groups = []
            for y_item in y_groups:
                seg_y, seg_w, seg_h, seg_c, seg_x = y_item
                merged = False
                for gi, (gy, gw, gh, gc, gx) in enumerate(new_groups):
                    if abs(seg_y - gy) <= 4:
                        # 檢查 X 軸是否足夠靠近
                        gap = max(0, max(gx, seg_x) - min(gx + gw, seg_x + seg_w))
                        if gap < 30:
                            new_x = min(gx, seg_x)
                            new_end = max(gx + gw, seg_x + seg_w)
                            new_w = new_end - new_x
                            new_groups[gi] = (gy, new_w, gh + seg_h, gc + seg_c, new_x)
                            merged = True
                            changed = True
                            break
                if not merged:
                    new_groups.append(y_item)
            y_groups = new_groups
        
        # 尋找梁編號的最大合法 Y 座標 (梁下緣線不能在梁編號的下面)
        max_valid_bottom_y = float('inf')
        beam_id_texts = []
        for item in ocr_items:
            if TableExtractor._is_beam_id(item["text"]):
                beam_id_texts.append(item)
        if beam_id_texts:
            lowest_beam_id = max(beam_id_texts, key=lambda x: x["cy"])
            max_valid_bottom_y = lowest_beam_id["min_y"] - 5
            # 將篩選出的梁名稱紀錄至物件管理器中 (這裡以出現次數最多或綜合串接為主，我們先單純塞入全部不重複的梁名)
            unique_names = list(dict.fromkeys(item["text"] for item in beam_id_texts))
            ctx.beam_name = " / ".join(unique_names)

        # 預先收集「純數字」的 OCR 文字 Y 座標 (標註尺寸數字，如 350, 4500)
        dim_number_ys = []
        for item in ocr_items:
            text = item["text"].strip().replace(',', '').replace('.', '')
            if re.match(r'^\d{2,}$', text):
                dim_number_ys.append(item["cy"])

        candidates = []
        for gy, total_w, total_h, count, gx in y_groups:
            # 放寬長度標準：大於畫面 15% 或是至少 40px，避免短懸臂梁被捨棄
            if total_w > max(40, cv_img.shape[1] * 0.15) and gy < max_valid_bottom_y:
                # 標註線過濾：如果這條水平線的 Y 座標 ±20px 內有純數字，判定為標註線
                is_dimension_line = any(abs(gy - dy) <= 20 for dy in dim_number_ys)
                if is_dimension_line:
                    ctx.lines.append(DetectedLine(
                        kind="h_dimension", status=ObjectStatus.REJECTED,
                        reject_reason="標註線", x=gx, y=gy, w=total_w, h=max(1, total_h // count),
                        abs_x=ctx.to_pdf_x(gx), abs_y=ctx.to_pdf_y(gy),
                        abs_w=ctx.to_pdf_w(total_w), abs_h=ctx.to_pdf_w(max(1, total_h // count))
                    ))
                    continue
                candidates.append((gy, total_w, max(1, total_h // count), gx))
                
        # 重置結論座標
        ctx.beam_top = None
        ctx.beam_bottom = None
        ctx.left_col = None
        ctx.right_col = None
        ctx.all_cols_sorted = []
        
        # 尋找最佳梁上下緣 (Best Pair)
        # 避免直接拿 min/max 導致取到外圍的「柱子截斷線」
        best_pair = None
        best_score = -float('inf')
        candidate_max_scores = [-float('inf')] * len(candidates)
        
        # 確認每條邊緣是否為局部(鄰近40px內)的最高或最低點
        is_topmost = []
        is_botmost = []
        for i, (cy, _, _, _) in enumerate(candidates):
            neighbors = [y for idx, (y, _, _, _) in enumerate(candidates) if abs(y - cy) <= 40]
            if len(neighbors) > 1:
                is_topmost.append(cy == min(neighbors))
                is_botmost.append(cy == max(neighbors))
            else:
                is_topmost.append(False)
                is_botmost.append(False)
        
        if len(candidates) >= 2:
            img_h = cv_img.shape[0]
            img_w = cv_img.shape[1]
            beam_id_ys = [item["cy"] for item in beam_id_texts]
            beam_id_xs = [item["cx"] for item in beam_id_texts]
            
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    y1, w1, th1, x1 = candidates[i]
                    y2, w2, th2, x2 = candidates[j]
                    
                    depth = abs(y1 - y2)
                    # 梁深合規：不能太小。最大深度放寬至 95%，避免截斷超深地梁
                    if depth < 10 or depth > img_h * 0.95:
                        continue
                        
                    # --- 指標正規化 (0.0 ~ 1.0) ---
                    # 1. 基礎長度比例 (佔畫面寬度，大於 50% 即視為滿分)
                    raw_w_ratio = min(w1, w2) / img_w
                    norm_min_w = min(1.0, raw_w_ratio / 0.5)
                    
                    # 2. X軸重疊率 (極度關鍵：上下兩條線必須對齊)
                    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    norm_overlap = x_overlap / max(w1, w2)
                    
                    # 3. 線寬一致性 (上下段粗細不能差太多)
                    norm_th_diff = abs(th1 - th2) / max(th1, th2)
                    
                    # 4. 長度差異比例 (差距佔較長線段的比例)
                    norm_w_diff = abs(w1 - w2) / max(w1, w2)
                    
                    # 5. 中心偏離比例 (偏離程度佔半個畫面高度的比例)
                    pair_center_y = (y1 + y2) / 2
                    norm_dist_center = abs((img_h / 2) - pair_center_y) / (img_h / 2)
                    
                    # 6. 梁名距離加分 (X 軸距離越近分數越高)
                    bonus_beam_id = 0
                    if beam_id_xs:
                        pair_center_x = (x1 + w1/2 + x2 + w2/2) / 2
                        min_dist_to_id = min(abs(pair_center_x - id_x) for id_x in beam_id_xs)
                        norm_dist_id = min_dist_to_id / img_w
                        bonus_beam_id = max(0, 5.0 * (1 - (norm_dist_id / 0.2)))
                        
                    # 7. 移除危險的實體包夾加分 (因梁名有時會在下方當作圖例標題)
                    
                    # 7. 柱位線粗細比對加分 (如果有給上一輪的柱位線資料，誤差 ≤1.0 即給 1.0 加分)
                    bonus_col_stroke = 0.0
                    if ctx.ref_col_stroke > 0:
                        avg_th = (th1 + th2) / 2.0
                        if abs(avg_th - ctx.ref_col_stroke) <= 1.0:
                            bonus_col_stroke = 1.0
                    
                    # 8. 邊界截斷線懲罰
                    edge_penalty = 0
                    if min(y1, y2) < 5 or max(y1, y2) > img_h - 5:
                        edge_penalty = 10.0
                        
                    # 9. 梁深 (Y軸排名位距) 加分
                    # candidates 已照 Y 座標由上到下排序，索引 (j - i) 就是排名差
                    bonus_depth = float(abs(j - i))
                    
                    # 10. 防呆：梁邊緣不能位在梁名稱的下方
                    penalty_below_id = 0.0
                    if beam_id_ys:
                        for id_y in beam_id_ys:
                            if y1 > id_y or y2 > id_y:
                                penalty_below_id = 100.0
                                break
                                
                    # 11. 局部邊緣加分：若為鄰近40px內的最高/最低候選者則 +2.0
                    bonus_outer_top = 2.0 if is_topmost[i] else 0.0
                    bonus_outer_bot = 2.0 if is_botmost[j] else 0.0
                    
                    # --- 最終權重組合 ---
                    # 分配：長度+4, X軸重疊+4.0, 歷史線寬+1.0, 梁深排位差, 梁名靠近+5, 局部最外緣+2
                    # 扣分：長短腳-2.0, 粗細不均-1.0, 偏心-1.0, 截斷-10.0, 越界-100.0
                    score = (norm_min_w * 4.0) + (norm_overlap * 4.0) + bonus_col_stroke + bonus_depth + bonus_beam_id \
                            + bonus_outer_top + bonus_outer_bot \
                            - (norm_w_diff * 2.0) - (norm_th_diff * 1.0) - (norm_dist_center * 1.0) - edge_penalty - penalty_below_id
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        
                    if score > candidate_max_scores[i]: candidate_max_scores[i] = score
                    if score > candidate_max_scores[j]: candidate_max_scores[j] = score
                        
        if best_pair:
            # 寫入正確的梁邊界
            idx1, idx2 = best_pair
            for i, (gy, w, th, gx) in enumerate(candidates):
                max_score_val = candidate_max_scores[i]
                score_str = f"Score:{max_score_val:.1f}" if max_score_val != -float('inf') else "Score:N/A"
                if i in best_pair:
                    ctx.lines.append(DetectedLine(
                        kind="h_beam_edge", status=ObjectStatus.CONFIRMED,
                        x=gx, y=gy, w=w, h=th,
                        abs_x=ctx.to_pdf_x(gx), abs_y=ctx.to_pdf_y(gy),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(th),
                        score_text=score_str
                    ))
                else:
                    ctx.lines.append(DetectedLine(
                        kind="h_rejected", status=ObjectStatus.REJECTED,
                        reject_reason="非最佳梁邊界(可能是柱截斷線)", x=gx, y=gy, w=w, h=th,
                        abs_x=ctx.to_pdf_x(gx), abs_y=ctx.to_pdf_y(gy),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(th),
                        score_text=score_str
                    ))
                    
            y1, _, th1, _ = candidates[idx1]
            y2, _, th2, _ = candidates[idx2]
            ctx.beam_top = min(y1, y2)
            ctx.beam_bottom = max(y1, y2)
            ctx.beam_stroke = max(1, (th1 + th2) / 2)
            
            beam_top = ctx.beam_top
            beam_bottom = ctx.beam_bottom
            beam_stroke = ctx.beam_stroke
            
            # 2. 垂直線
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
            v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_x_cols = []
            for cnt in v_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                tolerance_w = max(1.5, beam_stroke * 0.25)
                if abs(w - beam_stroke) > tolerance_w:
                    ctx.lines.append(DetectedLine(
                        kind="v_rejected", status=ObjectStatus.REJECTED,
                        reject_reason="線寬不符",
                        x=x, y=y, w=w, h=h, lx=x, rx=x, free_y=0,
                        abs_x=ctx.to_pdf_x(x), abs_y=ctx.to_pdf_y(y),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(h),
                        abs_lx=ctx.to_pdf_x(x), abs_rx=ctx.to_pdf_x(x), abs_free_y=ctx.to_pdf_y(0)
                    ))
                    continue
                    
                touches_top = abs((y + h) - beam_top) <= 3 and y < beam_top - 20
                touches_bot = abs(y - beam_bottom) <= 3 and (y + h) > beam_bottom + 20
                crosses_top = (y < beam_top - 10) and ((y + h) > beam_top + 15)
                crosses_bot = ((y + h) > beam_bottom + 10) and (y < beam_bottom - 15)
                
                passed = touches_top or touches_bot or crosses_top or crosses_bot
                
                reject_reason = "幾何不符" if not passed else ""
                lx, rx = x, x
                is_leader = False
                free_y = 0
                
                # === OCR 鄰接濾波器 (Adjacency Filter) ===
                if passed:
                    is_full_cross = crosses_top and crosses_bot
                    if not is_full_cross:
                        free_y = y if (touches_top or crosses_top) else (y + h)
                        
                        # 追蹤 L 型轉折的水平延伸段
                        def get_horizontal_extents(bin_img, base_x, base_y, max_search=200):
                            H, W = bin_img.shape
                            lx_val = base_x
                            while lx_val > 0 and base_x - lx_val < max_search:
                                y_min, y_max = max(0, base_y-4), min(H, base_y+5)
                                if np.any(bin_img[y_min:y_max, lx_val-1]):
                                    lx_val -= 1
                                else:
                                    break
                            rx_val = base_x
                            while rx_val < W - 1 and rx_val - base_x < max_search:
                                y_min, y_max = max(0, base_y-4), min(H, base_y+5)
                                if np.any(bin_img[y_min:y_max, rx_val+1]):
                                    rx_val += 1
                                else:
                                    break
                            return lx_val, rx_val
                            
                        lx, rx = get_horizontal_extents(binary, x, free_y, 250)
                        
                        # 排除 T 形轉角 (十字交叉)
                        is_t_shape = (x - lx) > 8 and (rx - x) > 8
                        
                        if not is_t_shape:
                            for item in ocr_items:
                                x1, x2 = item["min_x"], item["max_x"]
                                y1, y2 = item["min_y"], item["max_y"]
                                
                                near_tip = (x1 - 35 <= x <= x2 + 35) and (y1 - 35 <= free_y <= y2 + 35)
                                near_left = (x1 - 35 <= lx <= x2 + 35) and (y1 - 35 <= free_y <= y2 + 35)
                                near_right = (x1 - 35 <= rx <= x2 + 35) and (y1 - 35 <= free_y <= y2 + 35)
                                
                                if near_tip or near_left or near_right:
                                    is_leader = True
                                    break
                        if is_leader:
                            passed = False
                            reject_reason = "引線判定"
                
                # 建立 DetectedLine 物件
                if passed:
                    ctx.lines.append(DetectedLine(
                        kind="v_column", status=ObjectStatus.CONFIRMED,
                        x=x, y=y, w=w, h=h, lx=lx, rx=rx, free_y=free_y,
                        abs_x=ctx.to_pdf_x(x), abs_y=ctx.to_pdf_y(y),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(h),
                        abs_lx=ctx.to_pdf_x(lx), abs_rx=ctx.to_pdf_x(rx), abs_free_y=ctx.to_pdf_y(free_y)
                    ))
                    detected_x_cols.append(x)
                elif is_leader:
                    ctx.lines.append(DetectedLine(
                        kind="v_leader", status=ObjectStatus.REJECTED,
                        reject_reason=reject_reason,
                        x=x, y=y, w=w, h=h, lx=lx, rx=rx, free_y=free_y,
                        abs_x=ctx.to_pdf_x(x), abs_y=ctx.to_pdf_y(y),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(h),
                        abs_lx=ctx.to_pdf_x(lx), abs_rx=ctx.to_pdf_x(rx), abs_free_y=ctx.to_pdf_y(free_y)
                    ))
                else:
                    ctx.lines.append(DetectedLine(
                        kind="v_rejected", status=ObjectStatus.REJECTED,
                        reject_reason=reject_reason,
                        x=x, y=y, w=w, h=h, lx=lx, rx=rx, free_y=free_y,
                        abs_x=ctx.to_pdf_x(x), abs_y=ctx.to_pdf_y(y),
                        abs_w=ctx.to_pdf_w(w), abs_h=ctx.to_pdf_w(h),
                        abs_lx=ctx.to_pdf_x(lx), abs_rx=ctx.to_pdf_x(rx), abs_free_y=ctx.to_pdf_y(free_y)
                    ))
            
            # 去重 + 排序 (相距 < 10px 的視為同一根柱)
            detected_x_cols.sort()
            deduped = []
            for cx in detected_x_cols:
                if not deduped or abs(cx - deduped[-1]) > 10:
                    deduped.append(cx)
            ctx.all_cols_sorted = deduped
                    
            if ctx.all_cols_sorted:
                center_x = img_enhanced.width / 2
                left_cols = [cx for cx in ctx.all_cols_sorted if cx < center_x]
                right_cols = [cx for cx in ctx.all_cols_sorted if cx >= center_x]
                if left_cols:
                    ctx.left_col = max(left_cols)
                if right_cols:
                    ctx.right_col = min(right_cols)
                    
            # 缺柱線完美降級 (Fallback): 若無法確認左或右柱緣，提取 h_beam_edge 兩端當作柱線
            if ctx.left_col is None or ctx.right_col is None:
                h_edges = [l for l in ctx.lines if l.kind == "h_beam_edge"]
                if h_edges:
                    min_h_x = min(l.x for l in h_edges)
                    max_h_x = max((l.x + l.w) for l in h_edges)
                    fallback_y = min(l.y for l in h_edges)
                    fallback_bot_y = max((l.y + l.h) for l in h_edges)
                    fallback_h = fallback_bot_y - fallback_y
                    
                    if ctx.left_col is None:
                        ctx.left_col = min_h_x
                        ctx.all_cols_sorted.append(min_h_x)
                        ctx.lines.append(DetectedLine(
                            kind="v_column", status=ObjectStatus.CONFIRMED, reject_reason="水平邊緣降級",
                            x=min_h_x, y=fallback_y, w=3, h=fallback_h
                        ))
                    if ctx.right_col is None:
                        ctx.right_col = max_h_x
                        ctx.all_cols_sorted.append(max_h_x)
                        ctx.lines.append(DetectedLine(
                            kind="v_column", status=ObjectStatus.CONFIRMED, reject_reason="水平邊緣降級",
                            x=max_h_x, y=fallback_y, w=3, h=fallback_h
                        ))
                    ctx.all_cols_sorted.sort()

    # ================================================================
    # Debug 七彩診斷繪圖 (從 CropContext 讀取所有偵測物件)
    # ================================================================
    @staticmethod
    def _draw_debug(ctx, index: int):
        """在 ctx.img 上繪製完整的七彩診斷光譜，並存檔到 crops/debug_col/。
        同時回傳一份只畫紅線的 Gemini 用乾淨圖片。
        """
        from PIL import ImageDraw
        
        offset_margin = 20
        img_debug = ctx.img.copy()
        img_gemini = ctx.img.copy()
        draw = ImageDraw.Draw(img_debug)
        draw_gemini = ImageDraw.Draw(img_gemini)
        drawn_lines = False
        
        # 紅線：最終裁切邊界
        if ctx.left_col is not None:
            final_x = max(0, ctx.left_col - offset_margin)
            draw.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            drawn_lines = True
        if ctx.right_col is not None:
            final_x = min(ctx.img.width, ctx.right_col + offset_margin)
            draw.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            drawn_lines = True
        
        # 粉紅線：梁實體上下緣
        if ctx.beam_top is not None:
            draw.line([(0, ctx.beam_top), (ctx.img.width, ctx.beam_top)], fill=(255, 105, 180), width=4)
        if ctx.beam_bottom is not None:
            draw.line([(0, ctx.beam_bottom), (ctx.img.width, ctx.beam_bottom)], fill=(255, 105, 180), width=4)
            
        # 青色線 (Cyan)：九宮格 OCR 定位界線 (畫在 debug 和 gemini 圖上)
        if getattr(ctx, "grid_mid_x_start", None) is not None:
            draw.line([(ctx.grid_mid_x_start, 0), (ctx.grid_mid_x_start, ctx.img.height)], fill=(0, 255, 255), width=3)
            draw_gemini.line([(ctx.grid_mid_x_start, 0), (ctx.grid_mid_x_start, ctx.img.height)], fill=(0, 255, 255), width=3)
        if getattr(ctx, "grid_mid_x_end", None) is not None:
            draw.line([(ctx.grid_mid_x_end, 0), (ctx.grid_mid_x_end, ctx.img.height)], fill=(0, 255, 255), width=3)
            draw_gemini.line([(ctx.grid_mid_x_end, 0), (ctx.grid_mid_x_end, ctx.img.height)], fill=(0, 255, 255), width=3)
            
        if getattr(ctx, "grid_top_bound", None) is not None:
            draw.line([(0, ctx.grid_top_bound), (ctx.img.width, ctx.grid_top_bound)], fill=(0, 255, 255), width=3)
            draw_gemini.line([(0, ctx.grid_top_bound), (ctx.img.width, ctx.grid_top_bound)], fill=(0, 255, 255), width=3)
        if getattr(ctx, "grid_bottom_bound", None) is not None:
            draw.line([(0, ctx.grid_bottom_bound), (ctx.img.width, ctx.grid_bottom_bound)], fill=(0, 255, 255), width=3)
            draw_gemini.line([(0, ctx.grid_bottom_bound), (ctx.img.width, ctx.grid_bottom_bound)], fill=(0, 255, 255), width=3)
        
        # 垂直線：根據 kind 上色
        for d in ctx.lines:
            if d.kind.startswith("h_"):
                continue  # 水平線不畫垂直標記
            
            if d.reject_reason == "幾何不符":
                color = (128, 128, 128)  # 灰色
            elif d.reject_reason == "線寬不符":
                color = (128, 0, 128)    # 紫色
            elif d.reject_reason == "引線判定":
                color = (0, 165, 255)    # 橘色
            elif d.kind == "v_column":
                color = (0, 255, 0)      # 綠色
            else:
                color = (128, 128, 128)  # 灰色 fallback
            
            draw.line([(d.x, d.y), (d.x, d.y + d.h)], fill=color, width=4)
            
            # 畫引線的水平狗腿延伸
            if d.lx < d.x or d.rx > d.x:
                draw.line([(d.lx, d.free_y), (d.rx, d.free_y)], fill=(255, 255, 0), width=3)
        
        # 畫圖例
        legend = [
            ("Beam Top/Bottom", (255, 105, 180)),
            ("Valid Inner Col", (0, 255, 0)),
            ("Offset Final Edge", (255, 0, 0)),
            ("Reject: Geom", (128, 128, 128)),
            ("Reject: Width", (128, 0, 128)),
            ("Reject: Leader", (0, 165, 255)),
            ("Leader Dog-leg", (255, 255, 0))
        ]
        draw.rectangle([(0, 0), (160, 15 + len(legend) * 15)], fill=(0, 0, 0))
        for idx, (txt, color) in enumerate(legend):
            y = 15 + idx * 15
            draw.line([(10, y - 4), (30, y - 4)], fill=color, width=3)
            draw.text((35, y - 10), txt, fill=(255, 255, 255))
        
        # 存檔
        if debug_mode:
            os.makedirs(os.path.join(output_dir, "debug_col"), exist_ok=True)
        if debug_mode:
            img_debug.save(os.path.join(output_dir, "debug_col", f"crop_{index}_debug.png"))
        
        red_line_hint = ""
        if drawn_lines:
            red_line_hint = ""
        
        return img_gemini, drawn_lines, red_line_hint

    # ================================================================
    # 影像增強
    # ================================================================
    @staticmethod
    def _enhance_image(img: Image.Image) -> Image.Image:
        """銳化 + 對比度增強 + 小圖放大，提升 Gemini 視覺辨識率"""
        # 銳化：讓工程圖的細線條文字更清晰
        img = img.filter(ImageFilter.SHARPEN)
        # 對比度拉高 1.4 倍
        img = ImageEnhance.Contrast(img).enhance(1.4)
        # 小圖強制放大至至少 1200px 寬
        if img.width < 1200:
            scale = 1200 / img.width
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS
            )
        return img

    # ================================================================
    # Self-Consistency 欄位級多數決合併
    # ================================================================
    @staticmethod
    def _majority_vote_value(values):
        """從多輪結果中取出最常見的非空值"""
        non_empty = [v for v in values if v and v != "" and v != []]
        if not non_empty:
            return values[0] if values else ""
        # 轉字串做比較 (因為 list 不可 hash)
        str_counter = Counter(str(v) for v in non_empty)
        most_common_str = str_counter.most_common(1)[0][0]
        # 回傳原始型別
        for v in non_empty:
            if str(v) == most_common_str:
                return v
        return non_empty[0]

    @staticmethod
    def _merge_voting_rounds(all_rounds_beams: list) -> list:
        """
        將多輪推論結果做欄位級多數決合併。
        all_rounds_beams: [ [beams_from_round1], [beams_from_round2], ... ]
        """
        if len(all_rounds_beams) == 1:
            return all_rounds_beams[0]

        # 取最多梁的那輪作為骨架 (skeleton)
        skeleton = max(all_rounds_beams, key=len)
        merged = []

        for s_idx, base_beam in enumerate(skeleton):
            base_id = base_beam.get("beam_id", "")
            # 從其他輪次找同名 beam
            candidates = [base_beam]
            for other_round in all_rounds_beams:
                if other_round is skeleton:
                    continue
                for ob in other_round:
                    ob_id = ob.get("beam_id", "")
                    # 同名匹配 或 同位置匹配 (index 相同)
                    if ob_id and ob_id == base_id:
                        candidates.append(ob)
                        break
                else:
                    # fallback: 用索引位置匹配
                    if s_idx < len(other_round):
                        candidates.append(other_round[s_idx])

            if len(candidates) == 1:
                merged.append(candidates[0])
                continue

            # 欄位級多數決
            merged_beam = {}
            all_keys = set()
            for c in candidates:
                all_keys.update(c.keys())

            for key in all_keys:
                vals = [c.get(key) for c in candidates if key in c]
                if key == "self_confidence":
                    # 信心取最高
                    merged_beam[key] = max(vals) if vals else 0
                elif key in ("crop_index",):
                    # 保留骨架的值
                    merged_beam[key] = base_beam.get(key)
                else:
                    merged_beam[key] = TableExtractor._majority_vote_value(vals)

            merged.append(merged_beam)

        return merged

    # ================================================================
    # 核心提取方法
    # ================================================================
    async def extract_tables(self, pdf_bytes: bytes, page_num: int = 0,
                             cv_bboxes: list = None, progress_cb=None,
                             cv_metrics: dict = None, voting_rounds: int = 1,
                             output_dir: str = "crops", debug_mode: bool = False) -> str:
        """
        強制大腦輸出純 JSON 字串，完全杜絕任何問候語或前文後理。
        Phase 5: 單圖單發 (Single-Focus Inference) 解決多模態眼花問題。
        新增：PaddleOCR 前置文字提取 + 影像增強 + Self-Consistency 投票。
        """
        import json
        import asyncio
        if not self.api_key or self.model is None:
            return '{"beams": []}'

        voting_rounds = max(1, min(3, int(voting_rounds)))

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]
            
            if cv_bboxes and len(cv_bboxes) > 0:
                max_dim = max(page.rect.width, page.rect.height)
                scale_factor = 3.0
                if max_dim * scale_factor > 8000.0:
                    scale_factor = max(1.5, 8000.0 / max_dim)
                mat = fitz.Matrix(scale_factor, scale_factor)
                # 已升級為付費帳號：4,000 RPM 上限
                sem = asyncio.Semaphore(50)
                final_beams = []
                
                completed_parents = 0
                completed_children = 0
                total_beams_found = 0
                total_crops = len(cv_bboxes)
                
                parent_total = cv_metrics.get("parent_count", total_crops) if cv_metrics else total_crops
                child_total = cv_metrics.get("child_count", 0) if cv_metrics else 0

                # === 前置初始化：PaddleOCR 模型載入 ===
                if progress_cb:
                    progress_cb(f"[Phase 2] 正在初始化 PaddleOCR 引擎與影像增強模組...")
                self._init_ocr()
                ocr_status = "✅ 已啟用" if self._ocr_available else "⚠️ 未安裝，純視覺模式"
                if progress_cb:
                    progress_cb(f"[Phase 2] PaddleOCR: {ocr_status}。影像增強: ✅ 已啟用。執行 Phase 3.9 預掃描自癒...")

                # === Phase 3.8: 預掃描與結構自癒 (Pre-Scan & Self-Healing) ===
                import io
                from PIL import Image
                import cv2
                import numpy as np
                import os
                import shutil
                
                pass1_dir = os.path.join(output_dir, "rough_cut_pass1")
                pass2_dir = os.path.join(output_dir, "precise_cut_pass2")
                os.makedirs(pass1_dir, exist_ok=True)
                os.makedirs(pass2_dir, exist_ok=True)
                
                with open(os.path.join(pass2_dir, "healing_log.txt"), "w", encoding="utf-8") as _f_heal:
                    _f_heal.write("=== 二次精切 (Phase 3.9) 邊界自癒與紀錄 ===\n\n")
                
                scan_results = []
                cv_bboxes = [list(box) for box in cv_bboxes]  # 確保為可修改的 list
                
                for i, bbox in enumerate(cv_bboxes):
                    rect = fitz.Rect(bbox)
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_enhanced = self._enhance_image(img)
                    
                    img_enhanced.save(os.path.join(pass1_dir, f"crop_{i}.png"))  # 留存初切結果
                    
                    _, ocr_items = self._run_ocr(img_enhanced)
                    actual_scale = img_enhanced.width / (bbox[2] - bbox[0])
                    ctx_scan = CropContext(
                        img=img_enhanced, ocr_items=ocr_items,
                        pdf_bbox=bbox, pdf_scale=actual_scale
                    )
                    self._detect_column_bounds(ctx_scan)
                    
                    # 將畫布上的像素座標，轉換為 PDF 的絕對座標
                    pdf_left = ctx_scan.to_pdf_x(ctx_scan.left_col) if ctx_scan.left_col is not None else None
                    pdf_right = ctx_scan.to_pdf_x(ctx_scan.right_col) if ctx_scan.right_col is not None else None
                    
                    min_obj_x = bbox[2]
                    max_obj_x = bbox[0]
                    for item in ctx_scan.ocr_items:
                        item_pdf_min = ctx_scan.to_pdf_x(item["min_x"])
                        item_pdf_max = ctx_scan.to_pdf_x(item["max_x"])
                        if item_pdf_min < min_obj_x: min_obj_x = item_pdf_min
                        if item_pdf_max > max_obj_x: max_obj_x = item_pdf_max
                    for line in ctx_scan.lines:
                        line_pdf_min = line.abs_x
                        line_pdf_max = line.abs_x + max(line.abs_w, 1.0)
                        if line_pdf_min < min_obj_x: min_obj_x = line_pdf_min
                        if line_pdf_max > max_obj_x: max_obj_x = line_pdf_max
                    
                    scan_results.append({
                        "b_min_x": bbox[0],
                        "b_max_x": bbox[2],
                        "pdf_left": pdf_left,
                        "pdf_right": pdf_right,
                        "min_obj_x": min_obj_x,
                        "max_obj_x": max_obj_x
                    })
                    
                # 執行修復邏輯
                for i in range(len(scan_results) - 1):
                    resA = scan_results[i]
                    resB = scan_results[i+1]
                    
                    # 防呆機制：避免「跨母塊」竊取！如果兩者 Y 座標不同，代表來自不同樓層/不同列的母塊，絕對不能互相推擠邊界
                    if abs(cv_bboxes[i][1] - cv_bboxes[i+1][1]) > 5.0:
                        continue
                        
                    # 若圖 A 右側找到柱子，且距離右邊界大於 33.3 (相當於實體 100px)
                    if resA["pdf_right"] is not None:
                        distA = resA["b_max_x"] - resA["pdf_right"]
                        # 圖 B 沒有左柱，或是左柱離圖 B 左邊緣很遠 (>33.3)
                        b_left_missing = (resB["pdf_left"] is None) or ((resB["pdf_left"] - resB["b_min_x"]) > 33.3)
                        
                        if distA > 33.3 and b_left_missing:
                            old_A_right = cv_bboxes[i][2]
                            old_B_left = cv_bboxes[i+1][0]
                            # A 偷了 B 的邊緣！將 A 的切線往右推 10 單位 (約實體 30px)，讓 A 完整包含柱子甚至一點點 B 的跨度
                            # 同時 B 從柱線邊緣原點起算 (不推進)，創造 30px 的安全重疊區 (Overlap)
                            cv_bboxes[i][2] = resA["pdf_right"] + 10.0
                            cv_bboxes[i+1][0] = resA["pdf_right"]
                            
                            move_A = cv_bboxes[i][2] - old_A_right
                            move_B = cv_bboxes[i+1][0] - old_B_left
                            discard = cv_bboxes[i+1][0] - cv_bboxes[i][2]
                            
                            msg = f"[邊界竊取修復] 圖塊 {i} 偷了 {i+1} 的左邊柱線！\n" \
                                  f"  > 圖塊 {i} 右側框線移動: {move_A*3:.1f}px (從實體像素 {old_A_right*3:.1f} 推至 {cv_bboxes[i][2]*3:.1f})\n" \
                                  f"  > 圖塊 {i+1} 左側框線移動: {move_B*3:.1f}px (從實體像素 {old_B_left*3:.1f} 推至 {cv_bboxes[i+1][0]*3:.1f})\n" \
                                  f"  > 造成重疊(負值)/丟棄(正值)區間寬度: {discard*3:.1f}px\n"
                            print(msg.strip())
                            with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                                _f_heal.write(msg + "\n")

                # 極端邊緣修剪
                if len(scan_results) > 0:
                    first = scan_results[0]
                    if first["pdf_left"] is not None:
                        # 收縮極限：不能切掉任何實體構件 (保護最左側的文字與線條)
                        safe_left = min(first["pdf_left"] - 13.3, first["min_obj_x"] - 5.0)
                        if (safe_left - first["b_min_x"]) > 33.3:
                            old_left = cv_bboxes[0][0]
                            cv_bboxes[0][0] = safe_left
                            move = cv_bboxes[0][0] - old_left
                            msg = f"[最左邊緣修剪] 圖塊 0 的左側有過多空白，向內收縮。\n  > 左邊框線移動: {move*3:.1f}px (在最左構件邊緣 {first['min_obj_x']*3:.1f}px 處停刀)\n"
                            print(msg.strip())
                            with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                                _f_heal.write(msg + "\n")
                    
                    last = scan_results[-1]
                    if last["pdf_right"] is not None:
                        # 收縮極限：不能切掉任何實體構件 (保護最右側的文字與線條)
                        safe_right = max(last["pdf_right"] + 13.3, last["max_obj_x"] + 5.0)
                        if (last["b_max_x"] - safe_right) > 33.3:
                            old_right = cv_bboxes[-1][2]
                            cv_bboxes[-1][2] = safe_right
                            move = old_right - cv_bboxes[-1][2]
                            msg = f"[最右邊緣修剪] 圖塊 {len(scan_results)-1} 的右側有過多空白，向內收縮。\n  > 右邊框線移動: {move*3:.1f}px (在最右構件邊緣 {last['max_obj_x']*3:.1f}px 處停刀)\n"
                            print(msg.strip())
                            with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                                _f_heal.write(msg + "\n")

                # Healing 後邊界合法性檢查：移除 min_x >= max_x 的無效 bbox
                valid_bboxes = []
                for idx, b in enumerate(cv_bboxes):
                    if b[2] - b[0] > 5:
                        valid_bboxes.append(b)
                    else:
                        msg = f"[過短丟棄] 圖塊 {idx} 修剪後剩餘寬度 {(b[2]-b[0])*3:.1f}px 過小，被完全丟棄！"
                        print(msg)
                        with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                            _f_heal.write(msg + "\n\n")
                cv_bboxes = valid_bboxes

                # === 二次分裂檢查 (Re-Split Check) ===
                # 用 OCR 語意判斷：如果一張圖裡面偵測到 ≥2 組不同的梁編號，才觸發分裂
                # 柱線位置僅作為「切在哪裡」的輔助參考
                new_bboxes = []
                for i, bbox in enumerate(cv_bboxes):
                    # 安全檢查：bbox 寬度必須 > 5 PDF units
                    if (bbox[2] - bbox[0]) < 5 or (bbox[3] - bbox[1]) < 5:
                        msg = f"[自癒機制] crop_{i} 實體尺寸過小 (寬 {(bbox[2]-bbox[0])*3:.1f}px, 高 {(bbox[3]-bbox[1])*3:.1f}px)，跳過二次分裂"
                        print(msg)
                        with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                            _f_heal.write(msg + "\n\n")
                        continue
                    rect = fitz.Rect(bbox)
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_enh = self._enhance_image(img)
                    _, ocr_items_rescan = self._run_ocr(img_enh)
                    
                    # 找出所有梁編號及其 X 中心座標
                    beam_ids = []
                    for item in ocr_items_rescan:
                        if TableExtractor._is_beam_id(item["text"]):
                            beam_ids.append({"text": item["text"], "cx": item["cx"]})
                    
                    # 去重：同名梁編號只留一個 (可能 OCR 重複偵測)
                    seen_names = set()
                    unique_beams = []
                    for b in sorted(beam_ids, key=lambda x: x["cx"]):
                        name = re.sub(r'[\s\-_]', '', b["text"])
                        if name not in seen_names:
                            seen_names.add(name)
                            unique_beams.append(b)
                    
                    if len(unique_beams) >= 2:
                        # ≥2 組不同梁編號 → 需要切割！
                        # 切割點數量 = 梁編號數 - 1
                        actual_scale = img_enh.width / (rect.x1 - rect.x0)
                        ctx_resplit = CropContext(
                            img=img_enh, ocr_items=ocr_items_rescan,
                            pdf_bbox=(rect.x0, rect.y0, rect.x1, rect.y1), pdf_scale=actual_scale
                        )
                        self._detect_column_bounds(ctx_resplit)
                        all_cols = ctx_resplit.all_cols_sorted
                        internal_cols = all_cols[1:-1] if len(all_cols) >= 3 else []
                        
                        split_points_pdf = []
                        for j in range(len(unique_beams) - 1):
                            # 計算相鄰兩組梁編號的 X 中點
                            mid_px = (unique_beams[j]["cx"] + unique_beams[j+1]["cx"]) / 2.0
                            
                            # 在內部柱線中找最接近此中點的那一根
                            best_col = None
                            if internal_cols:
                                best_col = min(internal_cols, key=lambda c: abs(c - mid_px))
                                # 柱線必須在兩組梁編號之間才有意義
                                if not (unique_beams[j]["cx"] < best_col < unique_beams[j+1]["cx"]):
                                    best_col = None
                            
                            if best_col is not None:
                                split_points_pdf.append(ctx_resplit.to_pdf_x(best_col))
                            else:
                                split_points_pdf.append(ctx_resplit.to_pdf_x(mid_px))
                        
                        edges = [bbox[0]] + split_points_pdf + [bbox[2]]
                        
                        # 安全檢查：每個子圖塊的寬度必須 > 10 PDF units (約 30px)
                        valid = all((edges[j+1] - edges[j]) > 10 for j in range(len(edges) - 1))
                        if not valid:
                            new_bboxes.append(bbox)
                            msg = f"[二次分裂失敗] 圖塊 {i} 偵測到 {len(unique_beams)} 組名為 {[b['text'] for b in unique_beams]} 的梁。\n  > 但是！強制分裂將會把某些片段切得小於 30px。演算法決定不切！ (保留原圖)"
                            print(msg)
                            with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                                _f_heal.write(msg + "\n\n")
                            continue
                        
                        beam_names = [b["text"] for b in unique_beams]
                        method = "柱線" if internal_cols else "梁編號中點"
                        msg = f"[二次分裂成功] 圖塊 {i} 偵測到 {len(unique_beams)} 組梁編號 {beam_names}！\n  > 因此以「{method}」為依據，將其從中斬斷成 {len(edges)-1} 片子圖塊。"
                        print(msg)
                        with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                            _f_heal.write(msg + "\n\n")
                            
                        for j in range(len(edges) - 1):
                            sub = list(bbox)
                            sub[0] = edges[j]
                            sub[2] = edges[j+1] + 10.0 if j < len(edges) - 2 else edges[j+1]
                            new_bboxes.append(sub)
                    else:
                        new_bboxes.append(bbox)
                
                cv_bboxes = new_bboxes
                        
                # 最終過濾：移除任何尺寸過小的 bbox
                valid_again = []
                for idx, b in enumerate(cv_bboxes):
                    if (b[2] - b[0]) >= 5 and (b[3] - b[1]) >= 5:
                        valid_again.append(b)
                    else:
                        msg = f"[終端過濾丟棄] 圖塊再次被修剪後，尺寸不足{(b[2]-b[0])*3:.1f}x{(b[3]-b[1])*3:.1f}，整塊直接剃除死亡！"
                        print(msg)
                        with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                            _f_heal.write(msg + "\n\n")
                cv_bboxes = valid_again
                
                # 儲存精準二切結果 (方便除錯觀看)
                for i, bbox in enumerate(cv_bboxes):
                    rect = fitz.Rect(bbox)
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_enhanced = self._enhance_image(img)
                    img_enhanced.save(os.path.join(pass2_dir, f"crop_{i}.png"))
                
                total_crops = len(cv_bboxes)  # 更新總數 (可能因分裂而增加)
                print(f"[自癒機制] 修復完成，最終 {total_crops} 個圖塊已配置。")
                # === END Phase 3.8 ===

                # 前置處理計數器
                ocr_completed = 0
                enhance_completed = 0
                
                # LLM 呼叫統計
                llm_call_count = 0
                prompt_tokens = 0
                candidates_tokens = 0

                # -----------------------------------------
                # 單次推論核心 (可被 voting wrapper 重複呼叫)
                # -----------------------------------------
                async def _single_inference(img_enhanced, prompt_text, config, index, retries=3):
                    """執行一次 Gemini 推論，回傳 beams list 或 None"""
                    nonlocal llm_call_count, prompt_tokens, candidates_tokens
                    async with sem:
                        for attempt in range(retries):
                            try:
                                resp = await self.model.generate_content_async(
                                    contents=[prompt_text, img_enhanced],
                                    generation_config=config,
                                    request_options={"timeout": 60}
                                )
                                llm_call_count += 1
                                if getattr(resp, "usage_metadata", None):
                                    prompt_tokens += resp.usage_metadata.prompt_token_count
                                    candidates_tokens += resp.usage_metadata.candidates_token_count
                                    
                                result_data = json.loads(resp.text)
                                return result_data.get("beams", [])
                            except Exception as e:
                                err_str = str(e).lower()
                                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                                    wait_time = 2 ** attempt
                                    print(f"[警告] 片段 {index} 遭遇短暫限流 (429)，等待 {wait_time} 秒後重試...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    print(f"[錯誤] 片段 {index} 推論例外: {e}")
                                    return None
                        return None

                # -----------------------------------------
                # 帶有 OCR + 增強 + Voting 的完整 crop 處理器
                # -----------------------------------------
                async def process_crop(bbox, index):
                    nonlocal completed_parents, completed_children, total_beams_found, ocr_completed, enhance_completed
                    current_bbox = list(bbox)
                    
                    try:
                        for retry_count in range(2):
                            rect = fitz.Rect(current_bbox)
                            pix = page.get_pixmap(matrix=mat, clip=rect)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            
                            file_suffix = f"_retry{retry_count}" if retry_count > 0 else ""
                            if debug_mode:
                                os.makedirs(output_dir, exist_ok=True)
                            img.save(f"crops/crop_{index}{file_suffix}.png")
                            
                            # === 影像增強 (放入背景執行緒避免卡住 asyncio) ===
                            img_enhanced = await asyncio.to_thread(self._enhance_image, img)
                            if retry_count == 0:
                                enhance_completed += 1
                            
                            # === PaddleOCR 前置文字提取 ===
                            # 改為將「增強後、放大後」的圖片餵給 OCR，確保 OCR 也能看清小字！
                            # 放進背景執行緒，確保它不會阻塞其他已發出 LLM 請求的 await 等待
                            ocr_hint, ocr_items = await asyncio.to_thread(self._run_ocr, img_enhanced)
                            if retry_count == 0:
                                ocr_completed += 1
                            
                            if progress_cb and retry_count == 0 and (ocr_completed % 5 == 0 or ocr_completed == total_crops):
                                progress_cb(f"[Phase 2] 前置處理進度：影像增強 {enhance_completed}/{total_crops}，OCR 文字提取 {ocr_completed}/{total_crops}")
                            
                            ocr_section = ""
                            if ocr_hint:
                                ocr_section = (
                                    "【OCR 預掃結果】以下是 OCR 引擎從此圖塊偵測到的原始文字與「相對位置」（僅供參考，請結合視覺判斷修正）：\n"
                                    f"{ocr_hint}\n\n"
                                )
                            
                            # === Pass 1: 初步柱位偵測 ===
                            actual_scale = img_enhanced.width / (current_bbox[2] - current_bbox[0])
                            ctx = CropContext(
                                img=img_enhanced, ocr_items=ocr_items, ocr_hint=ocr_hint,
                                pdf_bbox=current_bbox, pdf_scale=actual_scale
                            )
                            self._detect_column_bounds(ctx)
                            
                            # 儲存第一輪(Pass 1)算出的平均柱線寬度，傳遞給第二次擷取使用 (若為重試也持續繼承)
                            v_strokes = [l.w for l in ctx.lines if l.kind == "v_column"]
                            if v_strokes:
                                ctx.ref_col_stroke = sum(v_strokes) / len(v_strokes)
                            
                            offset_margin = 20
                            final_left = max(0, ctx.left_col - offset_margin) if ctx.left_col is not None else 0
                            final_right = min(ctx.img.width, ctx.right_col + offset_margin) if ctx.right_col is not None else ctx.img.width
                            
                            # === 真實物理裁切 (Physical Crop Pass 2) ===
                            if final_left > 0 or final_right < ctx.img.width:
                                ctx.img = ctx.img.crop((final_left, 0, final_right, ctx.img.height))
                                ctx.shift_after_crop(final_left)
                                
                                # 對物理裁切後的乾淨小圖重跑一次 OCR，捨棄舊座標，避免邊緣雜訊干擾
                                ocr_hint_pass2, new_ocr_items = self._run_ocr(ctx.img)
                                ctx.ocr_hint = ocr_hint_pass2
                                ctx.ocr_items = new_ocr_items
                            
                            # === Pass 2: 純淨小圖精準偵測 ===
                            ctx.clear_lines()
                            self._detect_column_bounds(ctx)
                            
                            # === 多域切割與多重曝光 OCR (Sub-cropping & Multi-Exposure) ===
                            self._run_sub_crops_ocr(ctx)
                            
                            # === 重新校準 OCR 提示 (因物理裁切導致畫布座標改變) ===
                            ocr_hint_refreshed = self._refresh_ocr_hint(ctx)
                            ctx.ocr_hint = ocr_hint_refreshed
                            
                            # 儲存最後一次 OCR 的結果到 txt
                            if debug_mode:
                                os.makedirs(os.path.join(output_dir, "debug_col"), exist_ok=True)
                                with open(os.path.join(output_dir, "debug_col", f"crop_{index}_ocr.txt"), "w", encoding="utf-8") as f:
                                    f.write(ocr_hint_refreshed)
                                
                            # === 繪圖 + 存檔 ===
                            img_gemini, drawn_lines, red_line_hint = self._draw_debug(ctx, index)
                            img_gemini.save(os.path.join(output_dir, f"crop_{index}{file_suffix}.png"))
                            
                            # === OCR-First: 規則引擎直接分配欄位 ===
                            from core.ocr_field_assigner import assign_fields, classify_text, FormatType
                            rule_beam, low_conf_items = assign_fields(ctx.ocr_items, ctx)
                            
                            # === Fallback: 如果 beam_id 為空，回去未裁切的 OCR 掃描結果找 ===
                            if not rule_beam.get("beam_id") and ocr_items:
                                from core.normalizer import normalize_text as _norm
                                for item in ocr_items:
                                    if classify_text(item["text"]) == FormatType.BEAM_ID:
                                        rule_beam["beam_id"] = _norm(item["text"])
                                        pass # print(f"  [Fallback] 從原始 OCR 找到 beam_id: '{rule_beam['beam_id']}'")
                                        break
                            
                            # === Fallback: dimensions 也可能被裁切過濾掉 ===
                            if not rule_beam.get("dimensions") and ocr_items:
                                import re as _re
                                for item in ocr_items:
                                    clean_txt = item["text"].strip().strip('-_.= —–~()（）[]【】')
                                    if _re.match(r'^\d+\s*[xX×*]\s*\d+$', clean_txt):
                                        from core.normalizer import normalize_text as _norm2
                                        rule_beam["dimensions"] = _norm2(item["text"])
                                        pass # print(f"  [Fallback] 從原始 OCR 找到 dimensions: '{rule_beam['dimensions']}'")
                            # 判斷是否需要 LLM 補位
                            LIST_FIELDS = {"top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
                                           "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right"}
                            needs_llm = bool(low_conf_items) or not rule_beam.get("beam_id")
                            
                            if needs_llm and self.model:
                                # === LLM Fallback: 只處理規則引擎搞不定的部分 ===
                                pass # print(f"[OCR-First] 片段 {index}: 需要 LLM 補位 ({len(low_conf_items)} 筆低信心項目)")
                                
                                # 建構精簡 prompt — 任務分類機制 (Task-Oriented)
                                
                                # 找出已經有值的欄位，作為 OCR 草稿參考
                                already_filled = []
                                for k, v in rule_beam.items():
                                    if k in ("self_confidence", "note", "crop_index", "_ocr_text", "_crop_file"):
                                        continue
                                    if (isinstance(v, list) and v) or (isinstance(v, str) and v):
                                        if isinstance(v, list):
                                            already_filled.append(f"  - {k} = {v}")
                                        else:
                                            already_filled.append(f"  - {k} = '{v}'")
                                
                                task1_section = ""
                                if low_conf_items:
                                    uncertain_lines = []
                                    for item in low_conf_items:
                                        uncertain_lines.append(f'  - "{item["text"]}" @ {item.get("pos_label", "?")} (信心:{item.get("conf", 0):.0%})')
                                    task1_section = "🌟 任務一：校對與歸類 (Review & Assign)\n以下是前置系統偵測到、但把握度不高的字串。請對照圖片填入對應的空缺欄位中。如果您能依稀看出是什麼字串，請盡量推測作答！\n"
                                    task1_section += "若確定該區塊印得太模糊完全無法判讀，請輸出字串「LLM看不出來」。\n"
                                    task1_section += "【絕對禁令】若確定該區塊是干擾雜訊，或根本沒有東西，絕對不可留空字串！字串欄位必須填入「LLM沒有東西」，陣列欄位必須填入 [\"LLM沒有東西\"]！\n\n" + "\n".join(uncertain_lines) + "\n\n"
                                
                                task2_section = "🌟 任務二：全面覆核與補漏 (Full Review & Fill)\n"
                                task2_section += "既然已經啟動了進階掃描，請您「全局掃描」整張圖片，並在 JSON 中輸出**完整的所有配筋欄位**！\n"
                                task2_section += "若本系統下方的 OCR 初步答案正確，請照抄保留；若 OCR 有錯或該對應方位沒有東西，請以您的視覺判斷為準來替換它；若是空缺的請幫忙補上。\n"
                                task2_section += "若確定完全無法判讀，請輸出「LLM看不出來」。若確定該格子對應的方位沒有標示任何東西，請輸出「LLM沒有東西」。\n\n"
                                
                                # 【OCR 草稿資料】
                                filled_section = ""
                                if already_filled:
                                    filled_section = "【OCR 初步掃描結果 (供參考，請覆核並保留對的、替換錯的)】\n" + "\n".join(already_filled) + "\n\n"
                                
                                ocr_section = ""
                                if ocr_hint_refreshed:
                                    ocr_section = (
                                        "【OCR 預掃結果】以下是 OCR 引擎偵測到的所有文字與精確位置（供找尋未決項目時參考）：\n"
                                        f"{ocr_hint_refreshed}\n\n"
                                    )
                                
                                prompt = (
                                    "這是一張經過系統『精準裁剪』的「單跨梁配筋詳圖」，圖片的左右邊界已經貼合了這根梁的端點。請仔細讀取並解析資料。\n\n"
                                    "【任務清單】請嚴格執行以下任務，並將結果輸出至 JSON 格式。請務必針對每一個需要尋找或校對的地方給出答案！絕對禁止擅自留空不答！\n\n"
                                    f"{task1_section}"
                                    f"{task2_section}"
                                    f"{filled_section}"
                                    f"{ocr_section}"
                                    "【解析規則】\n"
                                    "1. 實體幾何定位：九宮格方位等同於欄位的結尾！(極度重要)\n"
                                    "   - 包含「左方」(如 左上方/正左方/左下方) 的文字，必須放入帶有 `_left` 的欄位。\n"
                                    "   - 包含「中央」或純上/下方 (如 正上方/正中央/正下方) 的文字，必須放入帶有 `_mid` 或 `_middle` 的欄位。\n"
                                    "   - 包含「右方」(如 右上方/正右方/右下方) 的文字，必須放入帶有 `_right` 的欄位。\n"
                                    "2. ⚠️嚴禁混淆主筋與箍筋！主筋格式為 `N-#S` (如 3-#8)，絕不含 @。箍筋格式為 `N-#S@D` (如 13-#4@15)，一定含 @。\n"
                                    "3. 搭接長度：圖面上方區域的純數字是上層搭接長度，下方區域的是下層搭接長度。\n"
                                    "4. 梁編號與尺寸：若有要求您尋找，通常位於圖面上端或下端邊緣。\n"
                                    "請直接輸出 JSON 格式的 BeamList 資料。"
                                )
                                
                                # === LLM 推論 ===
                                if voting_rounds <= 1:
                                    single_config = genai.GenerationConfig(
                                        response_mime_type="application/json",
                                        response_schema=BeamList
                                    )
                                    llm_beams = await _single_inference(img_gemini, prompt, single_config, index)
                                else:
                                    temps = [0.2, 0.5, 0.8][:voting_rounds]
                                    round_results = []
                                    for t_idx, temp in enumerate(temps):
                                        vote_config = genai.GenerationConfig(
                                            response_mime_type="application/json",
                                            response_schema=BeamList,
                                            temperature=temp
                                        )
                                        r = await _single_inference(img_gemini, prompt, vote_config, index)
                                        if r is not None:
                                            round_results.append(r)
                                    if round_results:
                                        llm_beams = self._merge_voting_rounds(round_results)
                                    else:
                                        llm_beams = None
                                
                                if llm_beams and len(llm_beams) > 0:
                                    # === 合併策略：規則引擎的高信心值優先，LLM 填補空缺 ===
                                    llm_beam = llm_beams[0]  # 通常單跨只有一筆
                                    
                                    # 儲存原始 LLM 回答供 Benchmark 顯示
                                    rule_beam["_raw_llm"] = json.dumps(llm_beam, ensure_ascii=False)
                                    
                                    def _apply_llm_result(llm_source, is_retry=False):
                                        for k, v in llm_source.items():
                                            if k in ("self_confidence", "note", "crop_index"):
                                                continue
                                            rule_val = rule_beam.get(k)
                                            
                                            # === 陣列欄位特殊策略 ===
                                            if k in LIST_FIELDS:
                                                if isinstance(v, list) and v:
                                                    clean = [x for x in v if x != "LLM沒有東西"]
                                                    if not clean:
                                                        if rule_val:
                                                            pass # print(f"  [LLM清空/否決] {k}: {rule_val} -> (空)")
                                                        rule_beam[k] = []
                                                        continue  # LLM 認為完全沒東西，清空並保留空狀態
                                                        
                                                    # 檢查 LLM 的結果是否全部合規格
                                                    from core.ocr_field_assigner import classify_text, FormatType
                                                    all_valid = all(classify_text(x) == FormatType.REBAR for x in clean)
                                                    if all_valid:
                                                        # LLM 合規格 → 以 LLM 為主，取代規則引擎的值
                                                        rule_beam[k] = clean
                                                        pass # print(f"  [LLM補位(取代)] {k} = {clean}")
                                                    elif not (isinstance(rule_val, list) and rule_val):
                                                        # LLM 不合規格但原本是空的 → 姑且用 LLM 的
                                                        rule_beam[k] = clean
                                                        pass # print(f"  [LLM補位] {k} = {clean}")
                                                    else:
                                                        # LLM 不合規格且原本有值 → 保留原值
                                                        if not is_retry:
                                                            pass # print(f"  [LLM略過] {k}: LLM回答不合規格 {v}，保留原值 {rule_val}")
                                                continue
                                            
                                            # === 字串欄位：以 LLM 覆核結果為準 ===
                                            # 但 beam_id 和 dimensions 若 OCR 已高信心填入，LLM 不可覆蓋
                                            # (LLM 常把 -6 看成 -5，OCR 對精確文字辨識更可靠)
                                            if isinstance(v, str) and v and v != "LLM沒有東西":
                                                if k in ("beam_id", "dimensions") and isinstance(rule_val, str) and rule_val:
                                                    if rule_val != v:
                                                        pass # print(f"  [LLM覆蓋被駁回] {k}: OCR='{rule_val}', LLM='{v}' → 保留 OCR")
                                                    continue
                                                if rule_val != v:
                                                    pass # print(f"  [LLM補位/更正] {k}: '{rule_val}' -> '{v}'")
                                                rule_beam[k] = v
                                            elif v == "LLM沒有東西" or v == "LLM看不出來":
                                                if rule_val:
                                                    pass # print(f"  [LLM清空/否決] {k}: '{rule_val}' -> (空)")
                                                rule_beam[k] = ""

                                    _apply_llm_result(llm_beam)
                                    
                                    # === LLM 二次重試 (Iterative Refinement) ===
                                    # 遍歷所有配筋欄位（而非僅限於 LLM 呼叫前就已空的 empty_fields）
                                    ALL_CHECK_FIELDS = [
                                        "beam_id", "dimensions", "face_bars",
                                        "top_main_bars_left", "top_main_bars_mid", "top_main_bars_right",
                                        "bottom_main_bars_left", "bottom_main_bars_mid", "bottom_main_bars_right",
                                        "stirrups_left", "stirrups_middle", "stirrups_right",
                                        "lap_length_top_left", "lap_length_top_right",
                                        "lap_length_bottom_left", "lap_length_bottom_right",
                                    ]
                                    still_empty = []
                                    for k in ALL_CHECK_FIELDS:
                                        # 先檢查 LLM 的第一次回覆，若已明確宣告「LLM沒有東西」，則尊重其判定，不再二次逼問
                                        first_ans = llm_beam.get(k)
                                        explicitly_empty = False
                                        if k in LIST_FIELDS:
                                            if isinstance(first_ans, list) and len(first_ans) > 0 and first_ans[0] in ("LLM沒有東西", "LLM看不出來"):
                                                explicitly_empty = True
                                        else:
                                            if isinstance(first_ans, str) and first_ans in ("LLM沒有東西", "LLM看不出來"):
                                                explicitly_empty = True
                                                
                                        if explicitly_empty:
                                            continue
                                            
                                        # 若未宣告放棄，且最終合併到 rule_beam 的結果依然是空的，則列入重試
                                        val = rule_beam.get(k)
                                        is_empty = False
                                        if k in LIST_FIELDS:
                                            if not isinstance(val, list) or not [x for x in val if x not in ("LLM沒有東西", "LLM看不出來")]:
                                                is_empty = True
                                        else:
                                            if not isinstance(val, str) or not val or val in ("LLM沒有東西", "LLM看不出來"):
                                                is_empty = True
                                        if is_empty:
                                            still_empty.append(k)
                                            
                                    MAX_RETRY_FIELDS = 12
                                    pass # print(f"  [DEBUG-RETRY] still_empty = {still_empty}")
                                    pass # print(f"  [DEBUG-RETRY] ALL_CHECK_FIELDS 檢查結果:")
                                    for _dbg_k in ALL_CHECK_FIELDS:
                                        _dbg_rule = rule_beam.get(_dbg_k)
                                        _dbg_llm = llm_beam.get(_dbg_k)
                                        _dbg_in = _dbg_k in still_empty
                                        print(f"    {_dbg_k}: rule_beam={repr(_dbg_rule)}, llm_first={repr(_dbg_llm)}, 列入重試={_dbg_in}")
                                    if still_empty and len(still_empty) <= MAX_RETRY_FIELDS:
                                        pass # print(f"  [LLM二次重試] 發現仍有缺漏 {still_empty}，發動針對性詢問...")
                                        
                                        clean_rule_beam = {}
                                        for key, value in rule_beam.items():
                                            if key.startswith("_") or key in ("self_confidence", "note", "crop_index"):
                                                continue
                                            if key in still_empty:
                                                clean_rule_beam[key] = ["❗請重新掃描填補❗"] if key in LIST_FIELDS else "❗請重新掃描填補❗"
                                            else:
                                                clean_rule_beam[key] = value
                                                
                                        prompt_retry = (
                                            "這是一次覆核任務。您正在解析單跨梁配筋詳圖。\n"
                                            "在前一次的解析中，您遺漏了以下重要欄位：\n"
                                            f"  - 遺漏欄位: {', '.join(still_empty)}\n\n"
                                            "請重新將注意力高度集中在圖片對應的角落與邊緣部位（例如 `_left` 務必看最左側邊界，`_right` 看最右側邊界）。\n"
                                            "請輸出完整的 JSON。您可以直接把以下「已確定的資料」原封不動保留，但務必將標記為「❗請重新掃描填補❗」的欄位替換成您新找到的鋼筋資訊！\n"
                                            "若您非常確定圖面上對應位置真的完全沒有標示這些資訊，請再次於該欄位填寫「LLM沒有東西」。\n\n"
                                            "【解析規則】\n"
                                            "1. 實體幾何定位：九宮格方位等同於欄位的結尾！(極度重要)\n"
                                            "   - 包含「左方」(如 左上方/正左方/左下方) 的文字，必須放入帶有 `_left` 的欄位。\n"
                                            "   - 包含「中央」或純上/下方 (如 正上方/正中央/正下方) 的文字，必須放入帶有 `_mid` 或 `_middle` 的欄位。\n"
                                            "   - 包含「右方」(如 右上方/正右方/右下方) 的文字，必須放入帶有 `_right` 的欄位。\n"
                                            "2. ⚠️嚴禁混淆主筋與箍筋！主筋格式為 `N-#S` (如 3-#8)，絕不含 @。箍筋格式為 `N-#S@D` (如 13-#4@15)，一定含 @。\n"
                                            "3. 🌟 若主筋有多排（看到兩個以上的鋼筋數量標示），請務必存成陣列 (例如: [\"3-#11\", \"14-#11\"])。\n"
                                            "4. 搭接長度：圖面上方區域的純數字是上層搭接長度，下方區域的是下層搭接長度。\n"
                                            "5. 梁編號與尺寸：若有要求您尋找，通常位於圖面上端或下端邊緣。\n\n"
                                            f"【已確定資料 (請保留，並填補遺失處)】:\n{json.dumps(clean_rule_beam, ensure_ascii=False, indent=2)}"
                                        )
                                        retry_config = genai.GenerationConfig(
                                            response_mime_type="application/json",
                                            response_schema=BeamList
                                        )
                                        llm_beams_retry = await _single_inference(img_gemini, prompt_retry, retry_config, index)
                                        if llm_beams_retry and len(llm_beams_retry) > 0:
                                            _apply_llm_result(llm_beams_retry[0], is_retry=True)
                                            rule_beam["_raw_llm_retry"] = json.dumps(llm_beams_retry[0], ensure_ascii=False)
                                else:
                                    print(f"  [錯誤] LLM 回傳為空或發生例外，強制退回純 OCR 兜底")
                                    rule_beam["_raw_llm"] = '{"error": "LLM 伺服器異常或推論失敗，已強制使用 OCR 原始數據兜底"}'
                                            
                                # === OCR 兜底策略 (Safe Fallback) ===
                                # 如果 OCR 當初預測該項目屬於某個欄位 (只是因為信心不到95%被送到 LLM)
                                # 結果 LLM 卻選擇不回答 (或忘記回答)，為了避免遺失資料，我們決定採用 OCR 的原始預測。
                                if needs_llm:
                                    # 排序：優先處理沒有 fallback_suffix (即確實吻合格式、只是信心略低) 的項目
                                    sorted_fallback = sorted(low_conf_items, key=lambda x: 1 if "fallback_suffix" in x else 0)
                                    for item in sorted_fallback:
                                        pk = item.get("predicted_field")
                                        if pk:
                                            v = rule_beam.get(pk)
                                            
                                            # 不合規格的雜訊：絕對不兜底
                                            if "fallback_suffix" in item and item["fallback_suffix"]:
                                                pass # print(f"  [OCR Fallback] 放棄兜底不符規格的雜訊: '{item.get('text', '')}'")
                                                continue
                                            
                                            # LLM 已明確否決此欄位 (回覆「沒有東西」)：尊重 LLM 判定，不兜底
                                            llm_first_val = llm_beam.get(pk) if llm_beam else None
                                            llm_vetoed = False
                                            if pk in LIST_FIELDS:
                                                if isinstance(llm_first_val, list) and llm_first_val and llm_first_val[0] in ("LLM沒有東西", "LLM看不出來"):
                                                    llm_vetoed = True
                                            else:
                                                if isinstance(llm_first_val, str) and llm_first_val in ("LLM沒有東西", "LLM看不出來"):
                                                    llm_vetoed = True
                                            if llm_vetoed:
                                                pass # print(f"  [OCR Fallback] LLM 已明確否決 {pk}，不兜底: '{item.get('text', '')}'")
                                                continue
                                            
                                            # 陣列欄位：已有值就不追加 (信任 LLM 或規則引擎的判定)
                                            if pk in LIST_FIELDS and isinstance(v, list) and v:
                                                # 但如果裡面全是 LLM 標記，視為空
                                                real_vals = [x for x in v if x not in ("LLM沒有東西", "LLM看不出來")]
                                                if real_vals:
                                                    pass # print(f"  [OCR Fallback] {pk} 已有值 {real_vals}，略過: '{item.get('text', '')}'")
                                                    continue
                                            
                                            # 字串欄位：已有值就不覆蓋
                                            if isinstance(v, str) and v and v not in ("LLM沒有東西", "LLM看不出來"):
                                                pass # print(f"  [OCR Fallback] {pk} 已有值 '{v}'，略過: '{item.get('text', '')}'")
                                                continue
                                            
                                            # 欄位為空或只有 LLM 標記 → 執行兜底
                                            fallback_val = str(item["text"])
                                            if pk in LIST_FIELDS:
                                                parts = [p.strip() for p in fallback_val.split(",") if p.strip()]
                                                if not isinstance(v, list):
                                                    rule_beam[pk] = []
                                                # 清除 LLM 標記後追加
                                                rule_beam[pk] = [x for x in rule_beam[pk] if x not in ("LLM沒有東西", "LLM看不出來")]
                                                for part in parts:
                                                    if part not in rule_beam[pk]:
                                                        rule_beam[pk].append(part)
                                                pass # print(f"  [OCR Fallback] 補回空欄位: {pk} = {rule_beam[pk]}")
                                            else:
                                                rule_beam[pk] = fallback_val
                                                pass # print(f"  [OCR Fallback] 補回空欄位: {pk} = '{fallback_val}'")

                                # 全域清理: 將 "LLM沒有東西" 轉為真正的空位
                                for k_beam, v_beam in list(rule_beam.items()):
                                    if isinstance(v_beam, list):
                                        new_v = [x for x in v_beam if x != "LLM沒有東西"]
                                        if len(new_v) != len(v_beam):
                                            rule_beam[k_beam] = new_v
                                    elif isinstance(v_beam, str):
                                        if v_beam == "LLM沒有東西":
                                            rule_beam[k_beam] = ""
                                            
                                crops_beams = [rule_beam]
                            else:
                                # === 全部高信心，不需要 LLM ===
                                pass # print(f"[OCR-First] 片段 {index}: 全部高信心！免 LLM 呼叫 ✅")
                                crops_beams = [rule_beam]

                            if crops_beams is None:
                                break # 嚴重錯誤直接失敗
                                
                            break
                            
                        # (迴圈外) 更新結果並回傳
                        if crops_beams is None:
                            if index <= parent_total:
                                completed_parents += 1
                            else:
                                completed_children += 1
                            if progress_cb:
                                progress_cb(f"[Phase 2] 微觀圖塊解析中... 已解析完成 {completed_parents}/{parent_total} 張原始圖檔，已解析完成 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                            return None
                            
                        # 為每個跨度物件寫入 crop_index + OCR 結果
                        for b in crops_beams:
                            b["crop_index"] = index
                            b["_ocr_text"] = ctx.ocr_hint if ctx.ocr_hint else "(無 OCR 結果)"
                            # 使用最終重試次數的圖片檔名
                            f_suf = f"_retry{retry_count}" if retry_count > 0 else ""
                            b["_crop_file"] = f"crop_{index}{f_suf}.png"
                            
                            debug_lines_data = []
                            for ln in ctx.lines:
                                debug_lines_data.append({
                                    "kind": ln.kind,
                                    "status": ln.status.value,
                                    "reject_reason": ln.reject_reason,
                                    "abs_x": round(ln.abs_x, 2),
                                    "abs_y": round(ln.abs_y, 2),
                                    "abs_w": round(ln.abs_w, 2),
                                    "abs_h": round(ln.abs_h, 2),
                                    "abs_lx": round(ln.abs_lx, 2),
                                    "abs_rx": round(ln.abs_rx, 2),
                                    "abs_free_y": round(ln.abs_free_y, 2),
                                    "score_text": ln.score_text
                                })
                            b["_debug_lines"] = debug_lines_data
                            
                            b["_final_bbox"] = {
                                "abs_x0": round(ctx.to_pdf_x(0), 2),
                                "abs_y0": round(ctx.to_pdf_y(0), 2),
                                "abs_x1": round(ctx.to_pdf_x(ctx.img.width), 2),
                                "abs_y1": round(ctx.to_pdf_y(ctx.img.height), 2)
                            }
                            
                            print(f"[Gemini Vision] 片段 {index} 產出：{b.get('beam_id', 'Unknown')}")
                        
                        if index <= parent_total:
                            completed_parents += 1
                        else:
                            completed_children += 1
                            
                        total_beams_found += len(crops_beams)
                        if progress_cb:
                            progress_cb(f"[Phase 2] 微觀圖塊解析中... 已解析完成 {completed_parents}/{parent_total} 張原始圖檔，已解析完成 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                            
                        return crops_beams
                        
                    except Exception as e:
                        print(f"[錯誤] 片段 {index} 處理失敗: {e}")
                        if index <= parent_total:
                            completed_parents += 1
                        else:
                            completed_children += 1
                        if progress_cb:
                            progress_cb(f"[Phase 2] 微觀圖塊解析中... 已解析完成 {completed_parents}/{parent_total} 張原始圖檔，已解析完成 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                        return None
                            
                print(f"[Gemini Vision] 啟用微觀視覺單圖平行推論機制，共 {len(cv_bboxes)} 張圖 (Self-Consistency x{voting_rounds})...")
                tasks = [process_crop(bbox, i + 1) for i, bbox in enumerate(cv_bboxes)]
                results = await asyncio.gather(*tasks)
                
                for r_list in results:
                    if r_list is not None and isinstance(r_list, list):
                        final_beams.extend(r_list)
                        
                # === 繪製全域除錯原圖 ===
                try:
                    from PIL import ImageDraw
                    debug_max_dim = max(page.rect.width, page.rect.height)
                    scale_factor = 3.0
                    if debug_max_dim * scale_factor > 8000.0:
                        scale_factor = max(1.5, 8000.0 / debug_max_dim)
                    debug_mat = fitz.Matrix(scale_factor, scale_factor)
                    debug_pix = page.get_pixmap(matrix=debug_mat)
                    debug_img = Image.open(io.BytesIO(debug_pix.tobytes("png")))
                    draw = ImageDraw.Draw(debug_img)
                    
                    for b in final_beams:
                        if "_debug_lines" in b:
                            for ln in b["_debug_lines"]:
                                x = ln["abs_x"] * 3.0
                                y = ln["abs_y"] * 3.0
                                w = ln["abs_w"] * 3.0
                                h = ln["abs_h"] * 3.0
                                
                                status = ln.get("reject_reason", "")
                                kind = ln.get("kind", "")
                                
                                if kind == "h_beam_edge":
                                    draw.line([(x, y), (x + w, y)], fill=(255, 105, 180), width=4)
                                    score_text = ln.get("score_text", "")
                                    if score_text:
                                        draw.text((x + w/10, y - 25), score_text, fill=(255, 105, 180))
                                elif kind == "h_rejected":
                                    draw.line([(x, y), (x + w, y)], fill=(255, 0, 0), width=2)
                                    score_text = ln.get("score_text", "")
                                    if score_text:
                                        draw.text((x + w/10, y - 20), score_text, fill=(255, 0, 0))
                                elif kind == "h_dimension":
                                    draw.line([(x, y), (x + w, y)], fill=(255, 255, 0), width=3)
                                elif kind == "v_column":
                                    draw.line([(x, y), (x, y + h)], fill=(0, 255, 0), width=4)
                                else:
                                    if status == "幾何不符": color = (128, 128, 128)
                                    elif status == "線寬不符": color = (128, 0, 128)
                                    elif status == "引線判定" or kind == "v_leader": color = (0, 165, 255)
                                    else: color = (128, 128, 128)
                                    
                                    draw.line([(x, y), (x, y + h)], fill=color, width=4)
                                    
                                    if "abs_lx" in ln and "abs_rx" in ln and "abs_free_y" in ln:
                                        lx = ln["abs_lx"] * 3.0
                                        rx = ln["abs_rx"] * 3.0
                                        fy = ln["abs_free_y"] * 3.0
                                        if lx < x or rx > x:
                                            draw.line([(lx, fy), (rx, fy)], fill=(255, 255, 0), width=3)
                                            
                        if "_final_bbox" in b:
                            fb = b["_final_bbox"]
                            sx0, sy0 = fb["abs_x0"] * 3.0, fb["abs_y0"] * 3.0
                            sx1, sy1 = fb["abs_x1"] * 3.0, fb["abs_y1"] * 3.0
                            
                            # 繪製虛線邊框 (Dashed line) function
                            def draw_dashed_box(d, x0, y0, x1, y1, fill, width, dash_len=10):
                                lines = [
                                    ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                                    ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))
                                ]
                                for pt1, pt2 in lines:
                                    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                                    dist = (dx**2 + dy**2)**0.5
                                    if dist == 0: continue
                                    vx, vy = dx/dist, dy/dist
                                    curr_len = 0
                                    drawn = True
                                    while curr_len < dist:
                                        step = min(dash_len, dist - curr_len)
                                        if drawn:
                                            nx1, ny1 = pt1[0] + vx*curr_len, pt1[1] + vy*curr_len
                                            nx2, ny2 = nx1 + vx*step, ny1 + vy*step
                                            d.line([(nx1, ny1), (nx2, ny2)], fill=fill, width=width)
                                        curr_len += step
                                        drawn = not drawn
                                        
                            draw_dashed_box(draw, sx0, sy0, sx1, sy1, fill=(255, 165, 0), width=6, dash_len=15)
                            # 在左上角標示 crop 短檔名
                            crop_fname = b.get("_crop_file", "crop")
                            draw.text((sx0 + 5, sy0 + 5), crop_fname, fill=(255, 165, 0))
    
                    if debug_mode:
                                os.makedirs(output_dir, exist_ok=True)
                    debug_img.save("crops/debug_full_pdf.png")
                    print("[Debug] 全域除錯原圖已儲存至 crops/debug_full_pdf.png")
                except Exception as e:
                    print(f"[Debug Error] 繪製全域除錯原圖失敗: {e}")

                return json.dumps({
                    "beams": final_beams,
                    "metrics": {
                        "llm_calls": llm_call_count,
                        "prompt_tokens": prompt_tokens,
                        "candidates_tokens": candidates_tokens
                    }
                }, ensure_ascii=False)
                
            else:
                # 傳統全圖掃描架構
                full_config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BeamList
                )
                
                print("[Gemini Vision] 正在將 PDF 轉為高解析度全域大圖...")
                max_dim = max(page.rect.width, page.rect.height)
                scale_factor = 2.0
                if max_dim * scale_factor > 8000.0:
                    scale_factor = max(1.5, 8000.0 / max_dim)
                mat = fitz.Matrix(scale_factor, scale_factor)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                prompt = "身為一位專精於建築結構工程的資深工程師，請你仔細檢視這張高解析度工程圖紙。請將圖面上的所有配筋詳圖轉換為完整的 JSON 格式。請盡可能保留所有欄位，如果圖面上某個位置沒有鋼筋資訊，請填寫空字串 \"\" 或空陣列 []。"
                
                response = await self.model.generate_content_async(
                    contents=[prompt, img],
                    generation_config=full_config,
                    request_options={"timeout": 120}
                )
                return response.text
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[異常] Gemini 解析失敗: {str(e)}")
            return '{"beams": []}'
        finally:
            try:
                doc.close()
            except Exception:
                pass
