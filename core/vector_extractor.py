import fitz
from typing import Dict, Any

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
        import os
        from PIL import Image
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
        
        # === Phase 3.8: 連續跨水平分解 (Continuous Beam Decomposition) ===
        # 對於超長連續梁，沿著各個跨度的標題之間進行一維聚類與斬波，切散成單跨獨立影像
        decomposed_bboxes = []
        child_to_parent = {}
        enable_decomp = cv_params.get('enable_decomp', True)
        
        if enable_decomp:
            for p_idx, bbox in enumerate(results):
                orig_x0, orig_y0, orig_x1, orig_y1 = bbox
                px0, py0 = int(orig_x0 * 4.0), int(orig_y0 * 4.0)
                px1, py1 = int(orig_x1 * 4.0), int(orig_y1 * 4.0)
                
                sub_thresh = thresh[py0:py1, px0:px1]
                if sub_thresh.shape[0] < 20 or sub_thresh.shape[1] < 20: continue
                
                text_kernel = np.ones((8, 40), np.uint8)
                sub_dilated = cv2.dilate(sub_thresh, text_kernel, iterations=1)
                sub_contours, _ = cv2.findContours(sub_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sub_boxes = [cv2.boundingRect(c) for c in sub_contours]
                if not sub_boxes: continue
                    
                main_box = max(sub_boxes, key=lambda b: b[2] * b[3])
                main_bottom_y = main_box[1] + main_box[3]
                
                # 定位潛在的多跨度標題
                potential_titles = []
                for (sx, sy, sw, sh) in sub_boxes:
                    if sy > main_bottom_y - 20 and sw > 60:
                        potential_titles.append((sx, sy, sw, sh))
                        
                if len(potential_titles) >= 2:
                    # 以 25px 虛擬容忍度做 Y 軸基線對齊聚類
                    y_groups = []
                    for pt in potential_titles:
                        added = False
                        for group in y_groups:
                            group_avg_y = sum(g[1] for g in group) / len(group)
                            if abs(pt[1] - group_avg_y) < 25:
                                group.append(pt)
                                added = True
                                break
                        if not added:
                            y_groups.append([pt])
                            
                    dominant_group = max(y_groups, key=len)
                    # 只有找出的同一條水平線上大於等於兩個文字孤島，才視為連續梁
                    if len(dominant_group) >= 2:
                        dominant_group.sort(key=lambda t: t[0])
                        
                        split_points_x = []
                        for i in range(len(dominant_group) - 1):
                            left_r = dominant_group[i][0] + dominant_group[i][2]
                            right_l = dominant_group[i+1][0]
                            mid_x = (left_r + right_l) / 2
                            split_points_x.append(px0 + mid_x)
                            
                        span_edges = [px0] + split_points_x + [px1]
                        # 給予 200 像素 (約 50 pt) 的超大重疊緩衝區 (Overlap)
                        overlap = 200
                        
                        for i in range(len(span_edges) - 1):
                            child_px0 = max(px0, span_edges[i] - (overlap if i > 0 else 0))
                            child_px1 = min(px1, span_edges[i+1] + (overlap if i < len(span_edges) - 2 else 0))
                            
                            child_orig_x0 = orig_x0 + ((child_px0 - px0) / 4.0)
                            child_orig_x1 = orig_x0 + ((child_px1 - px0) / 4.0)
                            
                            # 加入子圖 (保留原母圖在 results 中)
                            decomposed_bboxes.append([child_orig_x0, orig_y0, child_orig_x1, orig_y1])
                            child_to_parent[len(decomposed_bboxes) - 1] = p_idx
                            
            print(f"[Phase 3.8] 原有 {len(results)} 母塊，藉由水平分解獲得 {len(decomposed_bboxes)} 個額外單跨附屬塊。")
            results.extend(decomposed_bboxes)
        # ==============================================================
        
        # 執行除錯裁切並儲存
        if dropped_for_save or decomposed_bboxes or refined_results:
            import os
            import io
            from PIL import Image
            
            drop_dir = "crops/drop"
            split_dir = "crops/split"
            trimmed_dir = "crops/trimmed_parents"
            os.makedirs(drop_dir, exist_ok=True)
            os.makedirs(split_dir, exist_ok=True)
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
                    
            for idx, rect_coords in enumerate(refined_results):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    # 這是被 Phase 3.5 垂直微聚類過濾後的最終母圖
                    img.save(os.path.join(trimmed_dir, f"trimmed_parent_{idx}.png"))
                except Exception as e:
                    pass
                    
            for idx, rect_coords in enumerate(decomposed_bboxes):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    img.save(os.path.join(split_dir, f"split_child_{idx}.png"))
                except Exception as e:
                    pass
        
        metrics = {
            "total_contours": total_contours,
            "noise_dropped": noise_dropped,
            "noise_drop_rate": round((noise_dropped / total_contours) * 100, 1) if total_contours > 0 else 0,
            "nms_dropped": nms_dropped,
            "nms_drop_rate": round((nms_dropped / pre_nms_len) * 100, 1) if pre_nms_len > 0 else 0,
            "parent_count": len(refined_results),
            "child_count": len(decomposed_bboxes),
            "child_to_parent_map": child_to_parent
        }
        
        return results, metrics
