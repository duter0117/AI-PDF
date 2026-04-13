import os
import io
import json
import fitz
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

class VectorExtractor:
    def __init__(self, pdf_bytes: bytes):
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __del__(self):
        if self.doc:
            self.doc.close()

    def extract_page_data(self, page_num: int = 0) -> Dict[str, Any]:
        """
        提取 PDF 指定頁面的所有向量線段與純文字塊
        避開圖片失真，直接抓取 CAD 轉出時的底層幾何座標。
        """
        if page_num >= len(self.doc):
            return {"error": "Page number out of range"}
            
        page = self.doc[page_num]
        
        # 1. 提取所有向量幾何 (線段、矩形 等 CAD Path)
        drawings = page.get_drawings()
        vectors = []
        for d in drawings:
            # 簡化記錄每個繪圖物件的邊界框與屬性
            vectors.append({
                "type": "path",
                "rect": [round(x, 2) for x in d["rect"]], # [x0, y0, x1, y1]
                "color": d.get("color"),
                "width": d.get("width")
            })
            
        # 2. 提取所有文字塊
        text_blocks = page.get_text("blocks")
        texts = []
        for b in text_blocks:
            # b format: (x0, y0, x1, y1, "text", block_no, block_type)
            if b[6] == 0: # type 0 is text block
                text_content = b[4].strip()
                if text_content:
                    texts.append({
                        "rect": [round(x, 2) for x in b[:4]],
                        "text": text_content
                    })
                
        return {
            "page_num": page_num,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "vector_count": len(vectors),
            "text_count": len(texts),
            "texts_data": texts,
            "vectors_sample": vectors[:50]
        }

    def find_beam_bboxes_heuristic(self, page_num: int = 0) -> list:
        """
        Phase 3 核心演算法：自適應幾何尋邊 (Micro-Vision Crop Box)
        利用文字 Regex 定位梁標題，並往上搜索幾何邊界，使用聯集算出精確 Bounding Box。
        """
        import re
        page = self.doc[page_num]
        blocks = page.get_text("blocks")
        
        # 1. 抓出所有潛在的梁標題 (特徵：結尾伴隨著尺寸，如 B1F FWB1 (100x500))
        titles = []
        for b in blocks:
            if b[6] == 0:
                txt = b[4].strip()
                # 過濾含有 "(數字x數字)" 的工程標題常態字串
                if re.search(r'[a-zA-Z0-9_-]+', txt) and re.search(r'\([0-9]+[xX*][0-9]+\)', txt):
                    titles.append({"text": txt, "rect": b[:4]})
                    
        # 2. 為每個標題動態尋找涵蓋其配筋繪圖的 Bounding Box
        drawings = page.get_drawings()
        vectors = [d["rect"] for d in drawings]
        
        results = []
        for t in titles:
            tx0, ty0, tx1, ty1 = t["rect"]
            title_width = tx1 - tx0
            
            # 定義搜尋範圍閾值 (防呆機制：最多往上找 6 倍寬度，左右寬容 2.5 倍寬度)
            max_search_height = max(title_width * 6.0, 300) # 給個合理極限下限
            search_area = fitz.Rect(
                tx0 - (title_width * 2.5), 
                ty0 - max_search_height, 
                tx1 + (title_width * 2.5), 
                ty1 + 20 # 稍微往下包一點避免切到下方文字
            )
            
            # 蒐集落在此預估勢力範圍內的所有向量幾何
            contained_rects = []
            for v_rect in vectors:
                vr = fitz.Rect(v_rect)
                if vr.intersects(search_area):
                    contained_rects.append(vr)
                    
            # 幾何連通集計算：計算這些落網線條的聯集 (Rectangle Union)
            if contained_rects:
                final_box = contained_rects[0]
                for vr in contained_rects[1:]:
                    final_box = final_box | vr
                
                # 防呆閥值機制：萬一連通集爆掉 (幾何範圍失控交疊到隔壁)，強制裁斷其越境長度
                if final_box.height > max_search_height * 1.5:
                    final_box.y0 = ty0 - max_search_height
                    
                # 確保標題本身一定在框內
                final_box = final_box | fitz.Rect(tx0, ty0, tx1, ty1)
            else:
                # 萬一這隻梁只有文字沒有畫圖? 就退回安全搜尋框
                final_box = search_area
                
            # 將最終完美的畫框擴大 15 pixel 作為邊界緩衝
            final_box.x0 -= 15
            final_box.y0 -= 15
            final_box.x1 += 15
            final_box.y1 += 15
            
            results.append({
                "beam_id": t["text"],
                "anchor_rect": t["rect"],
                "adaptive_bbox": [round(final_box.x0, 2), round(final_box.y0, 2), round(final_box.x1, 2), round(final_box.y1, 2)]
            })
            
        return results

    def _nms_bboxes(self, bboxes: list, iou_thresh: float = 0.5) -> tuple[list, list]:
        """
        Non-Maximum Suppression：過濾高度重疊的 bbox，避免同一張圖被傳給 Gemini 多次。
        輸入按面積由大到小排序，優先保留較大的框。
        """
        if not bboxes:
            return [], []
        # 按面積降序，保留大框優先
        sorted_boxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep = []
        drops = []
        for bbox in sorted_boxes:
            suppressed = False
            for k in keep:
                inter_x0 = max(bbox[0], k[0])
                inter_y0 = max(bbox[1], k[1])
                inter_x1 = min(bbox[2], k[2])
                inter_y1 = min(bbox[3], k[3])
                inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
                area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_b = (k[2] - k[0]) * (k[3] - k[1])
                iou = inter / (area_a + area_b - inter + 1e-6)
                if iou > iou_thresh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(bbox)
            else:
                drops.append(bbox)
        return keep, drops

    def extract_opencv_bboxes(self, page_num: int = 0, cv_params: dict = None) -> tuple[list, dict]:
        """
        Phase 3: OpenCV 形態學尋邊 (Morphological Bounding)
        接收動態 cv_params (dilation_iterations, min_area, padding_bottom) 取代寫死的魔法數字。
        最後執行 NMS 去除重疊的父子輪廓，避免重複送圖給 Gemini。
        """
        if cv_params is None:
            cv_params = {}
            
        dilation_iterations = int(cv_params.get('dilation_iterations', 2))
        min_area = int(cv_params.get('min_area', 100000))
        padding_bottom = int(cv_params.get('padding_bottom', 160))
        import cv2
        import numpy as np
        
        page = self.doc[page_num]
        
        # 放大渲染做二值化處理
        mat = fitz.Matrix(4.0, 4.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        
        # 黑白轉換與膨脹
        _, thresh = cv2.threshold(img_data, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- Hough Transform 邊框清除器 ---
        hough_threshold_pct = int(cv_params.get('hough_threshold', 95)) / 100.0
        # 允許的斷線間隙放大一點，因為有時候圖框線會被跨過的字截斷
        gap_limit = 100 
        
        h_len = int(pix.width * hough_threshold_pct)
        v_len = int(pix.height * hough_threshold_pct)
        
        # 找尋影像中所有的極長直線
        min_search_length = min(h_len, v_len)
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=min_search_length//2, minLineLength=min_search_length, maxLineGap=gap_limit)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                # 嚴格判斷方向與佔比：
                # 1. 如果是水平線 (X跨度極大，Y沒什麼變)，就用寬度 (h_len) 來當標準
                is_horizontal = (dx >= h_len) and (dy < 50)
                # 2. 如果是垂直線 (Y跨度極大，X沒什麼變)，就用高度 (v_len) 來當標準
                is_vertical = (dy >= v_len) and (dx < 50)
                
                if is_horizontal or is_vertical:
                    cv2.line(thresh, (x1, y1), (x2, y2), 0, thickness=20)
        # -----------------------------------
        
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
        
        # 儲存膨脹後的海島圖供使用者視覺除錯
        os.makedirs("crops", exist_ok=True)
        img_island = Image.fromarray(dilated)
        img_island.save(f"crops/debug_islands_page_{page_num}.png")
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        total_contours = len(contours)
        noise_dropped = 0
        pre_nms_results = []
        
        # 記錄要存檔除錯的跌落圖塊
        dropped_for_save = []
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # 過濾掉雜訊、單獨文字體，以及超級大的頁框
            if area > min_area and area < (pix.width * pix.height * 0.25):
                # 退回給 PDF 的原始單位坐標 (除以 4.0)
                orig_x0 = max(0, (x - 40) / 4.0)
                orig_y0 = max(0, (y - 40) / 4.0)
                orig_x1 = min(page.rect.width, (x + w + 40) / 4.0)
                orig_y1 = min(page.rect.height, (y + h + padding_bottom) / 4.0)
                pre_nms_results.append([orig_x0, orig_y0, orig_x1, orig_y1])
            else:
                noise_dropped += 1
                # 若面積過大 (> 25%)，很可能是誤判為整張大圖框而被丟棄者，必須存檔供檢閱
                if area >= (pix.width * pix.height * 0.25):
                    orig_x0 = max(0, (x - 10) / 4.0)
                    orig_y0 = max(0, (y - 10) / 4.0)
                    orig_x1 = min(page.rect.width, (x + w + 10) / 4.0)
                    orig_y1 = min(page.rect.height, (y + h + 10) / 4.0)
                    dropped_for_save.append(("oversize", [orig_x0, orig_y0, orig_x1, orig_y1]))
                # 對於面積實在太小(例如只有單獨文字、碎點，面積 < 4000)者直接放生不存檔，否則會跑出幾千張圖拖垮系統
                elif area > 4000:
                    orig_x0 = max(0, (x - 10) / 4.0)
                    orig_y0 = max(0, (y - 10) / 4.0)
                    orig_x1 = min(page.rect.width, (x + w + 10) / 4.0)
                    orig_y1 = min(page.rect.height, (y + h + 10) / 4.0)
                    dropped_for_save.append(("noise", [orig_x0, orig_y0, orig_x1, orig_y1]))
        
        pre_nms_len = len(pre_nms_results)
        
        # NMS 去重後，再按 y, x 排序符合人類閱讀習慣
        results, nms_drops = self._nms_bboxes(pre_nms_results, iou_thresh=0.5)
        post_nms_len = len(results)
        nms_dropped = pre_nms_len - post_nms_len
        
        # 儲存 NMS 被刷掉的圖塊 (可能非常有價值，因為是被判斷為重複區域而刷掉)
        for nd in nms_drops:
            dropped_for_save.append(("nms", nd))
            
        results.sort(key=lambda b: (b[1], b[0]))
        
        # === Phase 3.5: OpenCV 形態學垂直微聚類 (視覺斬波) ===
        # 對抗過度 Padding 造成的雜訊，掃描標題與下根梁之間的「無文真空帶」切斷
        refined_results = []
        for bbox in results:
            orig_x0, orig_y0, orig_x1, orig_y1 = bbox
            px0, py0 = int(orig_x0 * 4.0), int(orig_y0 * 4.0)
            px1, py1 = int(orig_x1 * 4.0), int(orig_y1 * 4.0)
            
            sub_thresh = thresh[py0:py1, px0:px1]
            if sub_thresh.shape[0] < 20 or sub_thresh.shape[1] < 20:
                refined_results.append(bbox)
                continue
                
            # 專門針對文字特性，進行「水平強烈、垂直極輕微(以免吃掉跨梁間隙)」的膨脹
            text_kernel = np.ones((4, 40), np.uint8)
            sub_dilated = cv2.dilate(sub_thresh, text_kernel, iterations=1)
            
            sub_contours, _ = cv2.findContours(sub_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not sub_contours:
                refined_results.append(bbox)
                continue
                
            sub_boxes = [cv2.boundingRect(c) for c in sub_contours]
            
            # 1. 找出佔地最大的「主梁本體」
            main_box = max(sub_boxes, key=lambda b: b[2] * b[3])
            main_bottom_y = main_box[1] + main_box[3]
            
            # 2. 收集位於主梁下方所有的獨立文字島嶼
            blocks_below = []
            for b_idx, b in enumerate(sub_boxes):
                if b[1] > main_bottom_y - 15: # 允許些微重疊
                    blocks_below.append(b)
                    
            # 依據 Y 軸高度由上往下排列
            blocks_below.sort(key=lambda b: b[1])
            
            # 預設切除線定在框底
            cutoff_py1 = py1 - py0 
            
            # === 利用 1D 水平像素投影 (Horizontal Projection) 強效尋找真空斬波帶 ===
            # 因為 OpenCV 的膨脹容易把雜訊或格線連在一起，我們直接結算每行的黑色像素數量
            if main_bottom_y < cutoff_py1:
                below_img = sub_thresh[main_bottom_y:, :]
                row_sums = np.sum(below_img > 0, axis=1)
                
                # 定義真空帶：單列像素 < 10 (容許一點點垂直線或雜訊穿過)
                vacuum_mask = row_sums < 10
                vacuums = []
                current_vac = 0
                for i, is_vac in enumerate(vacuum_mask):
                    if is_vac:
                        current_vac += 1
                    else:
                        if current_vac > 0:
                            vacuums.append((i - current_vac, current_vac)) # (相對起點, 長度)
                            current_vac = 0
                if current_vac > 0:
                    vacuums.append((len(vacuum_mask) - current_vac, current_vac))
                    
                # 尋找最佳的一刀切斷點：
                best_cut_offset = -1
                for start, length in vacuums:
                    # 1. 如果遇到超級大斷層 (> 25px)，豪不猶豫馬上切，這一定是梁間縫隙
                    if length > 25:
                        best_cut_offset = start + (length // 2)
                        break
                    # 2. 中等斷層 (> 10px)，且距離主梁超過 40px (代表已經越過大部分文字) -> 切！
                    elif length > 10 and start > 40:
                        best_cut_offset = start + (length // 2)
                        break
                    # 3. 即使斷層很小 (>= 3px)，但距離主梁高達 70px 以上 (代表圖形極度擁擠但已經到了下一階段) -> 照切！
                    elif length >= 3 and start > 70:
                        best_cut_offset = start + (length // 2)
                        break
                        
                if best_cut_offset != -1:
                    # 成功找到斬波點
                    cutoff_py1 = main_bottom_y + best_cut_offset
                else:
                    # 如果什麼真空帶都找不到 (全部黏死)，退回安全底線：最後一個文字的底部 + 15
                    if blocks_below:
                        cutoff_py1 = min(cutoff_py1, blocks_below[-1][1] + blocks_below[-1][3] + 15)
                    else:
                        cutoff_py1 = min(cutoff_py1, main_bottom_y + 15)
            # =======================================================================
            
            # 反算回原始座標
            new_orig_y1 = orig_y0 + (cutoff_py1 / 4.0)
            bbox[3] = min(orig_y1, new_orig_y1)
            refined_results.append(bbox)
            
        results = list(refined_results)  # 必須淺拷貝！避免 extend 污染 refined_results 的 len()
        # ==============================================================
        
        # === Phase 3.6: Title Reclaim (標題歸屬回收) ===
        # 策略：先檢查每個 bbox 是否已包含梁編號標題。
        #       若已包含 → 不動作。
        #       若缺少 → 往下延伸搜尋，直到找到至少一組梁編號為止。
        import re
        
        def is_title_candidate(text):
            """
            用結構化濾除法辨識梁編號標題。
            不是強制規定格式，而是排除不可能是標題的東西。
            """
            text = text.strip()
            if len(text) < 2: 
                return False
            # 絕對不是標題的：鋼筋號數(#) 與 間距(@)
            if '@' in text or '#' in text: 
                return False
            # 尺寸格式 (如 50x70, 50*70) 是標題附屬物 -> 是標題 
            if re.search(r'\d+\s*[xX×*]\s*\d+', text): 
                return True
            # 如果含有英文字母 (如 G5-1, RF, WB-3) -> 大機率是標題
            if re.search(r'[A-Za-z]', text): 
                return True
            return False
            
        title_reclaim_count = 0
        for i, bbox in enumerate(results):
            orig_x0, orig_y0, orig_x1, orig_y1 = bbox
            
            # Step 1: 檢查目前 bbox 底層是否已包含梁編號
            # 標題一定在梁本體最下方，設定檢查箱為底部 30 pt，避免誤認到梁內部的普通註記
            inner_rect = fitz.Rect(orig_x0, max(orig_y0, orig_y1 - 30), orig_x1, orig_y1)
            inner_words = page.get_text("words", clip=inner_rect)
            
            has_title_inside = any(is_title_candidate(w[4]) for w in inner_words)
            
            if has_title_inside:
                continue  # bbox 本身已經含有標題結構，不往下延伸 (通過)
                
            # Step 2: bbox 底部沒有標題 → 往下延伸搜尋
            # 安全搜尋下限：最多往下搜 150 PDF pts，並排除 X 軸沒交集的障礙物
            max_search_y = min(orig_y1 + 150, page.rect.height)
            for j, other_bbox in enumerate(results):
                if j != i and other_bbox[1] > orig_y1:
                    o_x0, o_x1 = other_bbox[0], other_bbox[2]
                    overlap_x = max(0, min(orig_x1, o_x1) - max(orig_x0, o_x0))
                    if overlap_x > (orig_x1 - orig_x0) * 0.1:  # X軸有重疊超過 10%
                        max_search_y = min(max_search_y, other_bbox[1] + 5) # 容許些微壓線重疊，避免裁切文字
            
            if max_search_y <= orig_y1 + 3:
                continue  # 沒有搜尋空間
                
            # --- 策略 1: PDF 向量文字層 ---
            # 梁標題可能會偏離梁鋼筋圖的 X 軸範圍，將 X 軸向外擴張搜尋
            search_rect = fitz.Rect(orig_x0 - 60, orig_y1, orig_x1 + 60, max_search_y)
            below_words = page.get_text("words", clip=search_rect)
            
            title_y_max = orig_y1
            found_title = False
            title_line_y = None  # 記錄找到標題的那一行的 Y 坐標
            
            for w in below_words:
                w_text, w_y0, w_y1 = w[4], w[1], w[3]
                if is_title_candidate(w_text):
                    found_title = True
                    title_line_y = w_y0
                    title_y_max = max(title_y_max, w_y1 + 5)
                    bbox[0] = min(bbox[0], w[0] - 10)
                    bbox[2] = max(bbox[2], w[2] + 10)
            
            # 找到主標題行後，順便把在同一行或稍微低一點的尺寸相關文字也包進來
            if found_title and title_line_y is not None:
                for w in below_words:
                    w_text, w_y0, w_y1 = w[4], w[1], w[3]
                    if w_y0 - title_line_y < 12 and w_y0 >= title_line_y - 5:  # 同一行或略低
                        title_y_max = max(title_y_max, w_y1 + 5)
                        bbox[0] = min(bbox[0], w[0] - 10)
                        bbox[2] = max(bbox[2], w[2] + 10)
            
            # --- 策略 2: OCR Fallback ---
            if not found_title:
                try:
                    if not hasattr(self, '_title_ocr'):
                        from rapidocr_onnxruntime import RapidOCR
                        self._title_ocr = RapidOCR()
                    
                    if self._title_ocr:
                        search_pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=search_rect)
                        if search_pix.height > 5 and search_pix.width > 5:
                            channels = search_pix.n
                            search_img = np.frombuffer(search_pix.samples, dtype=np.uint8).reshape(
                                search_pix.height, search_pix.width, channels
                            )
                            if channels == 1:
                                search_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
                            elif channels == 4:
                                search_img = cv2.cvtColor(search_img, cv2.COLOR_BGRA2RGB)
                            
                            ocr_result, _ = self._title_ocr(search_img)
                            if ocr_result:
                                for ocr_bbox, ocr_text, ocr_conf in ocr_result:
                                    if ocr_conf > 0.5 and is_title_candidate(ocr_text):
                                        found_title = True
                                        ocr_y_bottom = max(pt[1] for pt in ocr_bbox) / 4.0
                                        title_y_max = max(title_y_max, search_rect.y0 + ocr_y_bottom + 5)
                                        ocr_x_left = min(pt[0] for pt in ocr_bbox) / 4.0
                                        ocr_x_right = max(pt[0] for pt in ocr_bbox) / 4.0
                                        bbox[0] = min(bbox[0], search_rect.x0 + ocr_x_left - 10)
                                        bbox[2] = max(bbox[2], search_rect.x0 + ocr_x_right + 10)
                except Exception:
                    pass
            
            if found_title and title_y_max > orig_y1:
                bbox[3] = title_y_max
                title_reclaim_count += 1
        
        if title_reclaim_count > 0:
            print(f"[Phase 3.6] 標題歸屬回收: {title_reclaim_count} 個 bbox 已擴展以包含梁編號")
        # ==============================================================
        
        
        # === Phase 3.8: 連續跨水平分解 (Continuous Beam Decomposition) ===
        # 對於超長連續梁，沿著各個跨度的標題之間進行一維聚類與斬波，切散成單跨獨立影像
        final_single_spans = []
        original_parents = []
        child_to_parent_map = {}
        enable_decomp = cv_params.get('enable_decomp', True)
        
        if enable_decomp:
            os.makedirs("crops/rough_cut_pass1", exist_ok=True)
            with open("crops/rough_cut_pass1/titles_log.txt", "w", encoding="utf-8") as _f:
                _f.write("=== 初切標題與字體大小紀錄 ===\n\n")

            # --- Two-Pass Architecture: Pass 1 (收集) ---
            all_potential_titles = []
            global_title_id = 0
            intermediate_bboxes_data = []
                
            for p_idx, bbox in enumerate(results):
                orig_x0, orig_y0, orig_x1, orig_y1 = bbox
                px0, py0 = int(orig_x0 * 4.0), int(orig_y0 * 4.0)
                px1, py1 = int(orig_x1 * 4.0), int(orig_y1 * 4.0)
                
                search_rect = fitz.Rect(orig_x0, orig_y0, orig_x1, orig_y1)
                
                potential_titles = []
                raw_titles = []
                if search_rect.width > 10 and search_rect.height > 10:
                    if not hasattr(self, '_title_ocr'):
                        from rapidocr_onnxruntime import RapidOCR
                        self._title_ocr = RapidOCR()
                    
                    search_pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=search_rect)
                    channels = search_pix.n
                    search_img = np.frombuffer(search_pix.samples, dtype=np.uint8).reshape(
                        search_pix.height, search_pix.width, channels
                    )
                    if channels == 1:
                        search_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
                    elif channels == 4:
                        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGRA2RGB)
                        
                    ocr_result, _ = self._title_ocr(search_img)
                    
                    if ocr_result:
                        for idx, (ocr_bbox, ocr_text, ocr_conf) in enumerate(ocr_result):
                            if ocr_conf > 0.5 and is_title_candidate(ocr_text):
                                ys = [pt[1] for pt in ocr_bbox]
                                xs = [pt[0] for pt in ocr_bbox]
                                raw_titles.append({
                                    "text": ocr_text,
                                    "cx": sum(xs)/len(xs),
                                    "cy": sum(ys)/len(ys),
                                    "bottom_y": max(ys),
                                    "h": max(ys) - min(ys),
                                    "w": max(xs) - min(xs)
                                })
                                
                    # 標題合併修正 (Spatial Merging)
                    for rt in raw_titles:
                        merged = False
                        for pt in potential_titles:
                            if abs(rt["cx"] - pt["cx"]) < 300 and abs(rt["cy"] - pt["cy"]) < 60:
                                pt["text"] += " " + rt["text"]
                                pt["cx"] = (pt["cx"] + rt["cx"]) / 2
                                pt["cy"] = (pt["cy"] + rt["cy"]) / 2
                                pt["bottom_y"] = max(pt["bottom_y"], rt["bottom_y"])
                                pt["w"] = max(pt["w"], rt["w"])
                                pt["h"] = max(pt["h"], rt["h"])
                                merged = True
                                break
                        if not merged:
                            potential_titles.append(rt)
                            
                    # 為本母塊所有標題賦予 global id 並收集
                    for pt in potential_titles:
                        pt["id"] = global_title_id
                        all_potential_titles.append({"id": global_title_id, "text": pt["text"]})
                        global_title_id += 1
                        
                intermediate_bboxes_data.append({
                    "p_idx": p_idx,
                    "bbox": bbox,
                    "px0": px0, "py0": py0, "px1": px1, "py1": py1,
                    "search_rect": search_rect,
                    "raw_titles": raw_titles,
                    "potential_titles": potential_titles
                })

            # --- Two-Pass Architecture: LLM 過濾 ---
            valid_ids_set = set()
            if all_potential_titles:
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    try:
                        import google.generativeai as l_genai
                        l_genai.configure(api_key=api_key)
                        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
                        model = l_genai.GenerativeModel(model_name)
                        
                        prompt = """你是一位專業的結構工程師。我正在進行結構配筋圖（RC梁）的自動切圖作業。
以下是一批透過 OCR 從圖紙上擷取出來的候選文字。這些字串裡面：
- 有些是真實的「RC 梁標題」（通常帶有樓層、B/G/CB/SB 等梁代號，且可能附加上如 50x70 等尺寸）。
- 有些是不相干的「雜訊」（例如：圖號像是13f415、剖面索引、網格線標記如 2-5(EF)、無意義碎字、鋼筋配置提醒）。

請你幫我嚴格過濾這份名單，剔除所有雜訊，只保留真實的「RC 梁標題」。
請「嚴格且僅以」 JSON 陣列 (List of Integers) 格式輸出合法標題的 id。
例如：如果 id 1, 5, 8 是合法梁標題，請輸出 [1, 5, 8]。不要回傳任何其他字串內容，也不要包含解釋或 Markdown 語法（如果有 Markdown 請確保只有 ```json 包著）。

【輸入字串列表】：
""" + json.dumps(all_potential_titles, ensure_ascii=False)

                        response = model.generate_content(prompt)
                        text = response.text
                        s_idx = text.find('[')
                        e_idx = text.rfind(']')
                        if s_idx != -1 and e_idx != -1:
                            valid_ids_list = json.loads(text[s_idx:e_idx+1])
                            valid_ids_set = set(valid_ids_list)
                            print(f"[Phase 3.8] LLM 高效雜訊過濾完成: 保留 {len(valid_ids_set)} / {len(all_potential_titles)} 個標題")
                        else:
                            raise ValueError("回傳格式不含 JSON 陣列")
                    except Exception as e:
                        print(f"[Phase 3.8] LLM 雜訊過濾失敗 ({e})，退回寬鬆免過濾模式。")
                        valid_ids_set = set(t["id"] for t in all_potential_titles)
                else:
                    valid_ids_set = set(t["id"] for t in all_potential_titles)
            
            # --- Two-Pass Architecture: Pass 2 (落刀裁切) ---
            trimmed_parent_logs = []
            phase37_deleted = 0
            phase37_split = 0
            for item in intermediate_bboxes_data:
                p_idx = item["p_idx"]
                orig_x0, orig_y0, orig_x1, orig_y1 = item["bbox"]
                px0, px1 = item["px0"], item["px1"]
                py0 = item["py0"]
                raw_titles = item["raw_titles"]
                
                # LLM 篩選合法標題
                filtered_titles = [t for t in item["potential_titles"] if t["id"] in valid_ids_set]
                was_split = False
                
                # === Phase 3.7 自檢 規則 1: 無合法標題 → 刪除 ===
                if len(filtered_titles) == 0:
                    phase37_deleted += 1
                    print(f"[Phase 3.7] 刪除無標題母塊 {p_idx} (y0={orig_y0:.1f})")
                    original_parents.append([orig_x0, orig_y0, orig_x1, orig_y1])
                    trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": []})
                    continue
                
                # === Phase 3.7 自檢 規則 2: Y 軸多排偵測 ===
                # cy 是 4x 像素座標，120px at 4x ≈ 30pt
                sorted_by_y = sorted(filtered_titles, key=lambda t: t["cy"])
                y_groups = [[sorted_by_y[0]]]
                for t in sorted_by_y[1:]:
                    if t["cy"] - y_groups[-1][-1]["cy"] > 120:  # 120px at 4x = 30pt
                        y_groups.append([t])
                    else:
                        y_groups[-1].append(t)
                
                if len(y_groups) >= 2:
                    # 垂直分割：從上排最低標題的 bottom_y + 20px(≈5pt) 切一刀
                    phase37_split += 1
                    print(f"[Phase 3.7] 母塊 {p_idx} 垂直分割: {len(y_groups)} 排, 標題: {[t['text'] for t in filtered_titles]}")
                    
                    prev_pdf_y = orig_y0
                    for g_idx, group in enumerate(y_groups):
                        lowest_bottom = max(t["bottom_y"] for t in group)
                        
                        # 截斷尾巴：每個子塊的底邊緊貼其所屬那排的最低標題
                        sub_y1 = (py0 + lowest_bottom + 5) / 4.0
                        sub_y1 = min(orig_y1, sub_y1)
                        
                        if g_idx < len(y_groups) - 1:
                            cut_pdf_y = (py0 + lowest_bottom + 20) / 4.0  # 5pt below
                            # 上半部
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, sub_y1]
                            original_parents.append(sub_bbox)
                            trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": group})
                            
                            # 這個子塊內如果有 >= 2 個 X 方向的標題，也要做水平分割
                            x_sorted = sorted(group, key=lambda t: t["cx"])
                            if len(x_sorted) >= 2:
                                split_points_x = []
                                for i in range(len(x_sorted) - 1):
                                    mid_x = (x_sorted[i]["cx"] + x_sorted[i+1]["cx"]) / 2.0
                                    split_points_x.append(px0 + mid_x)
                                span_edges = [px0] + split_points_x + [px1]
                                overlap = 200
                                for i in range(len(span_edges) - 1):
                                    child_px0 = max(px0, span_edges[i] - (overlap if i > 0 else 0))
                                    child_px1 = min(px1, span_edges[i+1] + (overlap if i < len(span_edges) - 2 else 0))
                                    child_orig_x0 = orig_x0 + ((child_px0 - px0) / 4.0)
                                    child_orig_x1 = orig_x0 + ((child_px1 - px0) / 4.0)
                                    final_single_spans.append([child_orig_x0, prev_pdf_y, child_orig_x1, sub_y1])
                                    child_to_parent_map[len(final_single_spans) - 1] = p_idx
                            else:
                                final_single_spans.append(sub_bbox)
                            
                            prev_pdf_y = cut_pdf_y
                        else:
                            # 最後一排：底邊用原始的 orig_y1
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, orig_y1]
                            original_parents.append(sub_bbox)
                            trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": group})
                            
                            x_sorted = sorted(group, key=lambda t: t["cx"])
                            if len(x_sorted) >= 2:
                                split_points_x = []
                                for i in range(len(x_sorted) - 1):
                                    mid_x = (x_sorted[i]["cx"] + x_sorted[i+1]["cx"]) / 2.0
                                    split_points_x.append(px0 + mid_x)
                                span_edges = [px0] + split_points_x + [px1]
                                overlap = 200
                                for i in range(len(span_edges) - 1):
                                    child_px0 = max(px0, span_edges[i] - (overlap if i > 0 else 0))
                                    child_px1 = min(px1, span_edges[i+1] + (overlap if i < len(span_edges) - 2 else 0))
                                    child_orig_x0 = orig_x0 + ((child_px0 - px0) / 4.0)
                                    child_orig_x1 = orig_x0 + ((child_px1 - px0) / 4.0)
                                    final_single_spans.append([child_orig_x0, prev_pdf_y, child_orig_x1, orig_y1])
                                    child_to_parent_map[len(final_single_spans) - 1] = p_idx
                            else:
                                final_single_spans.append(sub_bbox)
                    
                    continue  # 已處理完，跳過下面的原有邏輯
                
                # === 原有邏輯：單排母塊 ===
                if len(filtered_titles) >= 1:
                    # [動態截斷尾巴]
                    lowest_y_px = max(t["bottom_y"] for t in filtered_titles)
                    # lowest_y_px 位於 search_img 內，該圖片最上緣對應的 PDF 坐標是 py0 / 4.0
                    new_orig_y1 = (py0 + lowest_y_px + 5) / 4.0
                    orig_y1 = min(orig_y1, new_orig_y1)
                    
                original_parents.append([orig_x0, orig_y0, orig_x1, orig_y1])
                trimmed_parent_logs.append({
                    "idx": len(original_parents) - 1,
                    "titles": filtered_titles
                })
                            
                if len(filtered_titles) >= 2:
                    was_split = True
                    dominant_group = filtered_titles
                    dominant_group.sort(key=lambda t: t["cx"])
                    print(f"[Phase 3.8] 母塊 {p_idx} 共保留 {len(dominant_group)} 組真實梁標題: {[t['text'] for t in dominant_group]}")
                    
                    with open("crops/rough_cut_pass1/titles_log.txt", "a", encoding="utf-8") as _f:
                        _f.write(f"▼ 母塊 {p_idx} (包圍盒: [x0={orig_x0:.1f}, y0={orig_y0:.1f}, x1={orig_x1:.1f}, y1={orig_y1:.1f}])\n")
                        _f.write(f"  原始 OCR 單詞數量: {len(raw_titles)}\n")
                        for idx, rt in enumerate(raw_titles):
                            _f.write(f"    [Raw {idx}] {rt['text']} (cx={rt['cx']:.1f}, cy={rt['cy']:.1f}, w={rt['w']:.1f}, h={rt['h']:.1f})\n")
                        _f.write(f"  總計偵測並採納 {len(dominant_group)} 個標題:\n")
                        for pt in item["potential_titles"]:
                            status = "✅採用" if pt["id"] in valid_ids_set else "❌LLM雜訊剔除"
                            if status == "✅採用":
                                _f.write(f"    [{status}] 文字: {pt['text']:<20} | cx={pt['cx']:.1f}, cy={pt['cy']:.1f} | 寬度: {pt['w']:>4.1f}px | 高度: {pt['h']:>4.1f}px\n")
                            else:
                                _f.write(f"    [{status}] 文字: {pt['text']:<20}\n")
                        _f.write("\n")
                    
                    split_points_x = []
                    for i in range(len(dominant_group) - 1):
                        mid_x = (dominant_group[i]["cx"] + dominant_group[i+1]["cx"]) / 2.0
                        split_points_x.append(px0 + mid_x)
                        
                    span_edges = [px0] + split_points_x + [px1]
                    overlap = 200
                    
                    for i in range(len(span_edges) - 1):
                        child_px0 = max(px0, span_edges[i] - (overlap if i > 0 else 0))
                        child_px1 = min(px1, span_edges[i+1] + (overlap if i < len(span_edges) - 2 else 0))
                        
                        child_orig_x0 = orig_x0 + ((child_px0 - px0) / 4.0)
                        child_orig_x1 = orig_x0 + ((child_px1 - px0) / 4.0)
                        
                        final_single_spans.append([child_orig_x0, orig_y0, child_orig_x1, orig_y1])
                        child_to_parent_map[len(final_single_spans) - 1] = p_idx
                        
                # If we did not split it, we simply retain the entire parent as our final single span
                if not was_split:
                    final_single_spans.append([orig_x0, orig_y0, orig_x1, orig_y1])
                            
            if phase37_deleted > 0 or phase37_split > 0:
                print(f"[Phase 3.7] Y軸自檢: 刪除 {phase37_deleted} 無標題塊, 垂直分割 {phase37_split} 多排塊")
            print(f"[Phase 3.8] 原有 {len(original_parents)} 母塊。套用單跨裁切後，共準備送出 {len(final_single_spans)} 個最終獨立測資。")
            results = final_single_spans
        # ==============================================================
        # 執行除錯裁切並儲存
        if dropped_for_save or final_single_spans or original_parents:
            
            drop_dir = "crops/drop"
            trimmed_dir = "crops/trimmed_parents"
            os.makedirs(drop_dir, exist_ok=True)
            os.makedirs(trimmed_dir, exist_ok=True)
            mat_save = fitz.Matrix(2.0, 2.0)
            
            for idx, (reason, rect_coords) in enumerate(dropped_for_save):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    img.save(os.path.join(drop_dir, f"drop_{reason}_{idx}.png"))
                except Exception as e:
                    pass
                    
            for idx, rect_coords in enumerate(original_parents):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    # 這是被 Phase 3.5 垂直微聚類過濾後的原始母圖
                    img.save(os.path.join(trimmed_dir, f"trimmed_parent_{idx}.png"))
                except Exception as e:
                    pass
                    
            try:
                log_path = os.path.join(trimmed_dir, "titles_summary.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    for log in trimmed_parent_logs:
                        f.write(f"▼ 母塊 trimmed_parent_{log['idx']}.png\n")
                        f.write(f"  包含 {len(log['titles'])} 個過濾後的真梁編號:\n")
                        for t in log["titles"]:
                            f.write(f"    - {t['text']} (X={t['cx']:.1f}, Y={t['cy']:.1f})\n")
                        f.write("\n")
            except Exception as e:
                print("Failed to write titles_summary.txt", e)
        
        metrics = {
            "total_contours": total_contours,
            "noise_dropped": noise_dropped,
            "noise_drop_rate": round((noise_dropped / total_contours) * 100, 1) if total_contours > 0 else 0,
            "nms_dropped": int(nms_dropped),
            "nms_drop_rate": round((nms_dropped / pre_nms_len) * 100, 1) if pre_nms_len > 0 else 0,
            "parent_count": len(original_parents),
            "child_count": len(final_single_spans),
            "child_to_parent_map": child_to_parent_map,
            "original_parents": original_parents
        }
        
        return results, metrics
