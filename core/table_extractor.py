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
    beam_id: str = Field(description="構件編號或名稱，例如 B1F FWB1。若沒有明確標示請填寫空字串。")
    dimensions: str = Field(description="構件外觀尺寸，例如 100x380。若沒有標示請填入空字串。")
    
    top_main_bars_left: List[str] = Field(description="上層主筋(左端)。若有多排請存入字串陣列，例如 ['5-#8', '3-#8']。圖面上無標示則為空陣列 []。")
    top_main_bars_mid: List[str] = Field(description="上層主筋(中央)。若有多排請存入陣列。無標示則為空陣列 []。")
    top_main_bars_right: List[str] = Field(description="上層主筋(右端)。若有多排請存入陣列。無標示則為空陣列 []。")
    bottom_main_bars_left: List[str] = Field(description="下層主筋(左端)。若有多排請存入陣列。無標示則為空陣列 []。")
    bottom_main_bars_mid: List[str] = Field(description="下層主筋(中央)。若有多排請存入陣列。無標示則為空陣列 []。")
    bottom_main_bars_right: List[str] = Field(description="下層主筋(右端)。若有多排請存入陣列。無標示則為空陣列 []。")
    
    stirrups_left: str = Field(description="箍筋(左側)，例如 1-#4@10，若圖面無標示則留空")
    stirrups_middle: str = Field(description="箍筋(中央)，例如 2-#4@10，若圖面無標示則留空。如果整支梁只有標註一個箍筋，請統一填入此欄位。")
    stirrups_right: str = Field(description="箍筋(右側)，例如 1-#4@10，若圖面無標示則留空")
    face_bars: str = Field(description="腰筋(又稱側邊鋼筋)，例如 12-#5 (E.F)，無標示則留空")
    
    lap_length_top_left: str = Field(description="上層鋼筋搭接長度(左)，通常標示在圖面上端。無則留空。")
    lap_length_top_right: str = Field(description="上層鋼筋搭接長度(右)，通常標示在圖面上端。無則留空。")
    lap_length_bottom_left: str = Field(description="下層鋼筋搭接長度(左)，通常標示在圖面下端。無則留空。")
    lap_length_bottom_right: str = Field(description="下層鋼筋搭接長度(右)，通常標示在圖面下端。無則留空。")
    
    self_confidence: int = Field(description="請給出本次辨識的信心分數(0-100)。畫面清晰完整則為95-100，有雜訊或字跡難辨則降低。")
    note: str = Field(description="若圖片中有任何文字無法歸類到上方標準欄位(例如特殊工法說明文字)，請全文抄錄至此。若無則留空。")

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
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
            self.model = genai.GenerativeModel(model_name)

    @staticmethod
    def _is_beam_id(text):
        """嚴格辨識梁編號。只有明確匹配工程梁命名慣例才算。"""
        text = text.strip()
        if len(text) < 2:
            return False
        if '@' in text or '#' in text:
            return False
        # 尺寸格式 (50x70, 110x250) 不是梁編號      
        if re.search(r'\d+\s*[xX×*]\s*\d+', text):
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
                if conf < 0.5 or not text.strip():
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
                
                # 極端邊緣標記 (例如最邊緣15%)
                edge_tags = []
                if rx < 0.15: edge_tags.append("極左邊緣")
                if rx > 0.85: edge_tags.append("極右邊緣")
                if ry < 0.15: edge_tags.append("極上邊緣")
                if ry > 0.85: edge_tags.append("極下邊緣")
                
                if edge_tags:
                    pos_label += " (" + ", ".join(edge_tags) + ")"

                clean_text = text.strip()
                # 專門防護 OCR 將長引出線誤判為「減號」的情況 (例如 "-16-#4@15" 變成 "16-#4@15")
                # 這也順便清除了周遭的雜點符號。只清頭尾，不影響單字裡面的 "-" (如 B1-6)。
                clean_text = clean_text.strip('-_.= ')
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
                    "pos_label": pos_label, "is_extreme": bool(edge_tags),
                    "_dup_key": text_key
                })
                texts.append(f'  "{clean_text}" @ {pos_label} (信心:{conf:.0%})')
                
            return "\n".join(texts) if texts else "", raw_items
        except Exception as e:
            print(f"[OCR] 發生錯誤: {e}")
            return "", []

    def _refresh_ocr_hint(self, ctx) -> str:
        """物理裁切後，根據新圖布大小與位移量，重算 OCR 的「上下左右」提示字串"""
        w, h = ctx.img.width, ctx.img.height
        if w == 0 or h == 0: return ""
        
        texts = []
        for item in ctx.ocr_items:
            cx, cy = item["cx"], item["cy"]
            rx, ry = cx / w, cy / h
            
            if rx < 0.33: pos_x = "左"
            elif rx > 0.67: pos_x = "右"
            else: pos_x = "中"
            
            if ry < 0.33: pos_y = "上"
            elif ry > 0.67: pos_y = "下"
            else: pos_y = "中"

            if pos_x == "中" and pos_y == "中": pos_label = "正中央"
            elif pos_x == "中": pos_label = f"正{pos_y}方"
            elif pos_y == "中": pos_label = f"正{pos_x}方"
            else: pos_label = f"{pos_x}{pos_y}方"
            
            edge_tags = []
            if rx < 0.15: edge_tags.append("極左邊緣")
            if rx > 0.85: edge_tags.append("極右邊緣")
            if ry < 0.15: edge_tags.append("極上邊緣")
            if ry > 0.85: edge_tags.append("極下邊緣")
            
            if edge_tags:
                pos_label += " (" + ", ".join(edge_tags) + ")"
                
            import re
            clean_text = item["text"]
            
            # OCR 預讀作弊: 針對純數字或是帶有 L= 的尺寸格式，強制塞入強烈暗示
            hint = ""
            is_lap = re.match(r'^(L|La|Ld)?\s*[=≈]?\s*\d{2,4}\s*(cm|mm)?$', clean_text.strip(), re.IGNORECASE)
            is_pure_num = re.match(r'^\d{2,4}$', clean_text.strip())
            # 放寬條件：只要位置偏上方或下方，且是純數字，就給出暗示
            if is_lap or (is_pure_num and ("上" in pos_label or "下" in pos_label)):
                hint = " [🌟系統暗示: 極可能是搭接長度]"
                
            conf = item["conf"]
            texts.append(f'  "{clean_text}" @ {pos_label}{hint} (信心:{conf:.0%})')
            
        return "\n".join(texts)

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
        
        # 分組：Y 座標相差 ≤3px 且 X 軸斷層不大於 50px (被標註線穿過) 的視為同一條線並接合
        y_groups = []  # [(representative_y, merged_width, total_h_for_avg, count, left_x)]
        for seg_y, seg_w, seg_h, seg_x in h_segments:
            merged = False
            for gi, (gy, gw, gh, gc, gx) in enumerate(y_groups):
                if abs(seg_y - gy) <= 3:
                    # 檢查 X 軸是否足夠靠近
                    gap = max(0, max(gx, seg_x) - min(gx + gw, seg_x + seg_w))
                    if gap < 50:
                        new_x = min(gx, seg_x)
                        new_end = max(gx + gw, seg_x + seg_w)
                        new_w = new_end - new_x
                        y_groups[gi] = (gy, new_w, gh + seg_h, gc + 1, new_x)
                        merged = True
                        break
            if not merged:
                y_groups.append((seg_y, seg_w, seg_h, 1, seg_x))
        
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
                    
                    # 6. 梁名距離加分 (距離越近分數越高)
                    bonus_beam_id = 0
                    if beam_id_ys:
                        min_dist_to_id = min(abs(pair_center_y - id_y) for id_y in beam_id_ys)
                        norm_dist_id = min_dist_to_id / img_h
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
        
        # 紅線：最終裁切邊界 (同時畫在 debug 和 gemini 圖上)
        if ctx.left_col is not None:
            final_x = max(0, ctx.left_col - offset_margin)
            draw.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            draw_gemini.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            drawn_lines = True
        if ctx.right_col is not None:
            final_x = min(ctx.img.width, ctx.right_col + offset_margin)
            draw.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            draw_gemini.line([(final_x, 0), (final_x, ctx.img.height)], fill="red", width=8)
            drawn_lines = True
        
        # 粉紅線：梁上下緣
        if ctx.beam_top is not None:
            draw.line([(0, ctx.beam_top), (ctx.img.width, ctx.beam_top)], fill=(255, 105, 180), width=4)
        if ctx.beam_bottom is not None:
            draw.line([(0, ctx.beam_bottom), (ctx.img.width, ctx.beam_bottom)], fill=(255, 105, 180), width=4)
        
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
        os.makedirs("crops/debug_col", exist_ok=True)
        img_debug.save(f"crops/debug_col/crop_{index}_debug.png")
        
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
                             cv_metrics: dict = None, voting_rounds: int = 1) -> str:
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
                mat = fitz.Matrix(3.0, 3.0)
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
                
                pass1_dir = "crops/rough_cut_pass1"
                pass2_dir = "crops/precise_cut_pass2"
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
                    
                    scan_results.append({
                        "b_min_x": bbox[0],
                        "b_max_x": bbox[2],
                        "pdf_left": pdf_left,
                        "pdf_right": pdf_right
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
                    if first["pdf_left"] is not None and (first["pdf_left"] - first["b_min_x"]) > 33.3:
                        old_left = cv_bboxes[0][0]
                        cv_bboxes[0][0] = first["pdf_left"] - 13.3
                        move = cv_bboxes[0][0] - old_left
                        msg = f"[最左邊緣修剪] 圖塊 0 的左側有過多空白，向內收縮。\n  > 左邊框線移動: {move*3:.1f}px (捨棄了外部無效畫面)\n"
                        print(msg.strip())
                        with open(os.path.join(pass2_dir, "healing_log.txt"), "a", encoding="utf-8") as _f_heal:
                            _f_heal.write(msg + "\n")
                    
                    last = scan_results[-1]
                    if last["pdf_right"] is not None and (last["b_max_x"] - last["pdf_right"]) > 33.3:
                        old_right = cv_bboxes[-1][2]
                        cv_bboxes[-1][2] = last["pdf_right"] + 13.3
                        move = cv_bboxes[-1][2] - old_right
                        msg = f"[最右邊緣修剪] 圖塊 {len(scan_results)-1} 的右側有過多空白，向內收縮。\n  > 右邊框線移動: {move*3:.1f}px (捨棄了外部無效畫面)\n"
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
                        name = re.sub(r'[\s\-_]', '', b["text"]).upper()
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

                # -----------------------------------------
                # 單次推論核心 (可被 voting wrapper 重複呼叫)
                # -----------------------------------------
                async def _single_inference(img_enhanced, prompt_text, config, index, retries=3):
                    """執行一次 Gemini 推論，回傳 beams list 或 None"""
                    async with sem:
                        for attempt in range(retries):
                            try:
                                resp = await self.model.generate_content_async(
                                    contents=[prompt_text, img_enhanced],
                                    generation_config=config,
                                    request_options={"timeout": 60}
                                )
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
                            os.makedirs("crops", exist_ok=True)
                            img.save(f"crops/crop_{index}{file_suffix}.png")
                            
                            # === 影像增強 ===
                            img_enhanced = self._enhance_image(img)
                            if retry_count == 0:
                                enhance_completed += 1
                            
                            # === PaddleOCR 前置文字提取 ===
                            # 改為將「增強後、放大後」的圖片餵給 OCR，確保 OCR 也能看清小字！
                            ocr_hint, ocr_items = self._run_ocr(img_enhanced)
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
                                # 過濾裁切外的 OCR 文字框
                                new_ocr_items = []
                                for item in ctx.ocr_items:
                                    cx_val = (item["min_x"] + item["max_x"]) / 2
                                    if final_left <= cx_val <= final_right:
                                        new_ocr_items.append(item)
                                ctx.ocr_items = new_ocr_items
                                ctx.shift_after_crop(final_left)
                            
                            # === Pass 2: 純淨小圖精準偵測 ===
                            ctx.clear_lines()
                            self._detect_column_bounds(ctx)
                            
                            # === 重新校準 OCR 提示 (因物理裁切導致畫布座標改變) ===
                            ocr_hint_refreshed = self._refresh_ocr_hint(ctx)
                            ocr_section = ""
                            if ocr_hint_refreshed:
                                ocr_section = (
                                    "【OCR 預掃結果】以下是 OCR 引擎從『裁切後』的這張小圖偵測到的文字與「精確相對位置」（請參考此精確座標判定左中右）：\n"
                                    f"{ocr_hint_refreshed}\n\n"
                                )
                            
                            # === 繪圖 + 存檔 ===
                            img_gemini, drawn_lines, red_line_hint = self._draw_debug(ctx, index)
                            img_gemini.save(f"crops/crop_{index}{file_suffix}.png")

                            prompt = (
                                f"{ocr_section}"
                                "這是一張經過系統『精準裁剪』的「單跨梁配筋詳圖」，圖片的左右邊界已經完美的貼合了這根梁的實際端點。請仔細讀取並解析配筋數據。\n\n"
                                "【解析規則】\n"
                                "1. 水平位置對應：這張圖切分為 左端(前30%)、中央(中40%)、右端(後30%)。若 OCR 預掃清單告訴你某個文字在「正上方/正下方/正中央」，這『絕對』是中央位置(_mid)的主筋或箍筋！絕對不可以因為右邊剛好沒畫面，就把原本在中央的東西『錯誤擠到右邊去』！請嚴格依照它的絕對位置填表。\n"
                                "2. 只讀取實體墨水：你是一名嚴格的數據萃取員。畫面上有繪製文字的地方你才寫，沒有文字的地方請直接留空 (字串填 ''、陣列填 [])。嚴禁依靠工程常理去『猜測』或『複製』別格的數字來填補空白！\n"
                                "2.5 OCR 容錯：OCR 預掃結果偶爾會出錯。如果 OCR 文字看起來不像任何工程標註格式，請無視該筆 OCR的內容，直接用你的視覺能力從圖片上讀取正確數值。\n"
                                "3. 主鋼筋陣列：主鋼筋分為上層(梁圖例之上)與下層(梁圖例之下)。同一個位置若有數排數字，請依序放入陣列 `['5-#8', '3-#8']`。如果只有一排，就是 `['5-#8']`。\n"
                                "4. 搭接長度：圖面最上方或最下方偶爾會出現獨立的二位或三位數數字(例如80, 150, 420)，若其偏左請填入搭接長度(左)，偏右則填入搭接長度(右)。搭接長度通常是純數字，沒有鋼筋符號，若在上下邊緣看到孤立數字，極高機率是它，千萬別漏掉！\n"
                                "5. 梁編號與尺寸：通常位於圖面上端或下端。請準確找尋並帶入 (例如 beam_id: 'G1', dimensions: '40x80')。\n"
                                "6. 圖紙邊緣文字：如果在極度邊緣或裁切邊有殘留的破裂文字，可能是掃描到的其他梁圖紙渣滓，請忽略它。\n"
                                "7. 非配筋無關圖片：如果這張圖只是一張表格的純標題 (例如寫著『梁配筋標準圖』) 而沒有任何鋼筋繪製，請直接回傳空陣列 `[]`。\n"
                                "請直接輸出 JSON 格式的 BeamList 資料，不要有其他廢話。"
                            )
                            
                            # === 推論/多數決 ===
                            if voting_rounds <= 1:
                                single_config = genai.GenerationConfig(
                                    response_mime_type="application/json",
                                    response_schema=BeamList
                                )
                                crops_beams = await _single_inference(img_gemini, prompt, single_config, index)
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
                                        print(f"[Voting] 片段 {index} (Retry:{retry_count}) 第 {t_idx+1}/{voting_rounds} 輪完成, 偵測到 {len(r)} beams")
                                
                                if round_results:
                                    crops_beams = self._merge_voting_rounds(round_results)
                                else:
                                    crops_beams = None

                            if crops_beams is None:
                                break # 嚴重錯誤直接失敗
                                
                            # === 適應性裁切 (Adaptive Re-cropping) ===
                            if retry_count == 0 and len(crops_beams) > 0:
                                # 提取預測的所有字串（忽略大小寫、空白）
                                pred_texts = []
                                for b in crops_beams:
                                    for v in b.values():
                                        if isinstance(v, list):
                                            for item in v: pred_texts.append(str(item).replace(" ", "").lower())
                                        else:
                                            pred_texts.append(str(v).replace(" ", "").lower())
                                
                                trim_left = 0
                                trim_right = 0
                                img_w = ctx.img.width
                                has_trim = False
                                
                                for item in ctx.ocr_items:
                                    if item["is_extreme"]:
                                        raw_t = item["text"].replace(" ", "").lower()
                                        used = False
                                        # 檢查是否被使用
                                        for pt in pred_texts:
                                            if raw_t in pt or pt in raw_t:
                                                used = True
                                                break
                                        
                                        if not used:
                                            if "極左" in item["pos_label"]:
                                                # 使用由 CropContext 提供的絕對座標系
                                                raw_pdf_x = ctx.to_pdf_x(item["max_x"])
                                                trim_dist = raw_pdf_x - current_bbox[0]
                                                trim_left = max(trim_left, trim_dist + 2)
                                                has_trim = True
                                            elif "極右" in item["pos_label"]:
                                                raw_pdf_min = ctx.to_pdf_x(item["min_x"])
                                                trim_dist = current_bbox[2] - raw_pdf_min
                                                trim_right = max(trim_right, trim_dist + 2)
                                                has_trim = True
                                
                                if has_trim:
                                    new_left = current_bbox[0] + trim_left
                                    new_right = current_bbox[2] - trim_right
                                    # 確保新的裁切框夠寬 (至少 30 pts)
                                    if new_right - new_left > 30:
                                        current_bbox[0] = new_left
                                        current_bbox[2] = new_right
                                        print(f"[Adaptive Cropping] 偵測到邊際孤兒字，片段 {index} 啟動重新裁切重試...")
                                        continue
                                
                            # 若無須裁切或已完成第二次重試，跳出迴圈
                            break
                            
                        # (迴圈外) 更新結果並回傳
                        if crops_beams is None:
                            if index <= parent_total:
                                completed_parents += 1
                            else:
                                completed_children += 1
                            if progress_cb:
                                progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
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
                            
                            print(f"[Gemini Vision] 片段 {index} 產出：{b.get('beam_id', 'Unknown')}")
                        
                        if index <= parent_total:
                            completed_parents += 1
                        else:
                            completed_children += 1
                            
                        total_beams_found += len(crops_beams)
                        if progress_cb:
                            progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                            
                        return crops_beams
                        
                    except Exception as e:
                        print(f"[錯誤] 片段 {index} 處理失敗: {e}")
                        if index <= parent_total:
                            completed_parents += 1
                        else:
                            completed_children += 1
                        if progress_cb:
                            progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
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
                    debug_mat = fitz.Matrix(3.0, 3.0)
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
    
                    os.makedirs("crops", exist_ok=True)
                    debug_img.save("crops/debug_full_pdf.png")
                    print("[Debug] 全域除錯原圖已儲存至 crops/debug_full_pdf.png")
                except Exception as e:
                    print(f"[Debug Error] 繪製全域除錯原圖失敗: {e}")

                return json.dumps({"beams": final_beams})
                
            else:
                # 傳統全圖掃描架構
                full_config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BeamList
                )
                
                print("[Gemini Vision] 正在將 PDF 轉為高解析度全域大圖...")
                mat = fitz.Matrix(2.0, 2.0)
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
