import os
import io
import fitz
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List

# 定義嚴格的 JSON 輸出規格
class BeamDetail(BaseModel):
    beam_id: str = Field(description="構件編號或名稱，例如 B1F FWB1")
    dimensions: str = Field(description="構件外觀尺寸，例如 100x380")
    
    # 上下端部分為 左、中、右 共 6 個獨立區間的主筋配置 (支援多排)
    top_main_bars_left: List[str] = Field(description="上層主筋(左端)。若有多排請分開記錄 (例如 ['第一排: 5-#8', '第二排: 3-#8'])。如果該位置圖面上沒有標示請給空陣列 []。")
    top_main_bars_mid: List[str] = Field(description="上層主筋(中央)。結構同上。")
    top_main_bars_right: List[str] = Field(description="上層主筋(右端)。結構同上。")
    bottom_main_bars_left: List[str] = Field(description="下層主筋(左端)。結構同上。")
    bottom_main_bars_mid: List[str] = Field(description="下層主筋(中央)。結構同上。")
    bottom_main_bars_right: List[str] = Field(description="下層主筋(右端)。結構同上。")
    
    stirrups_left: str = Field(description="箍筋左區，例如 1-#4@10，如果沒有則填空字串")
    stirrups_middle: str = Field(description="箍筋中區，例如 2-#4@10，如果沒有則填空字串")
    stirrups_right: str = Field(description="箍筋右區，例如 1-#4@10，如果沒有則填空字串")
    face_bars: str = Field(description="腰筋或側邊鋼筋，例如 12-#5 (E.F)，如果沒有則填空字串")
    
    lap_length_top_left: str = Field(description="上層筋搭接距離(左端)，如果圖面上沒有標示請填空字串")
    lap_length_top_right: str = Field(description="上層筋搭接距離(右端)，如果圖面上沒有標示請填空字串")
    lap_length_bottom_left: str = Field(description="下層筋搭接距離(左端)，如果圖面上沒有標示請填空字串")
    lap_length_bottom_right: str = Field(description="下層筋搭接距離(右端)，如果圖面上沒有標示請填空字串")
    
    self_confidence: int = Field(description="滿分為100。如果解析清晰且完整請填入 95~100 的數字。如果圖面模糊、有無法填入既有欄位的神秘文字，請給 90 以下的數字。")
    note: str = Field(description="【極度重要】如果有任何文字或數字出現在圖上，但無法分類到上方標準欄位(例如特殊工法說明等)，『務必』全文抄錄在此欄位。若完美解析無遺漏請留空字串。")

class BeamList(BaseModel):
    beams: List[BeamDetail] = Field(description="圖面上解析出的所有配筋詳圖物件清單")

class TableExtractor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
            self.model = genai.GenerativeModel(model_name)

    async def extract_tables(self, pdf_bytes: bytes, page_num: int = 0, cv_bboxes: list = None, progress_cb=None, cv_metrics: dict = None) -> str:
        """
        強制大腦輸出純 JSON 字串，完全杜絕任何問候語或前文後理。
        Phase 5: 單圖單發 (Single-Focus Inference) 解決多模態眼花問題。
        """
        import json
        import asyncio
        if not self.api_key or self.model is None:
            return '{"beams": []}'

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]
            
            if cv_bboxes and len(cv_bboxes) > 0:
                print(f"[Gemini Vision] 啟用微觀視覺單圖平行推論機制，共 {len(cv_bboxes)} 張圖...")
                
                # 改為強迫 AI 每次吐出可能包含多體的 BeamList (支援一圖多跨)，但注意力維持在單圖上
                single_config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BeamList
                )
                
                mat = fitz.Matrix(3.0, 3.0)
                # 已升級為付費帳號：4,000 RPM 上限，將 Semaphore 上限調高以大幅縮短處理時間
                sem = asyncio.Semaphore(50)
                final_beams = []
                
                completed_parents = 0
                completed_children = 0
                total_beams_found = 0
                total_crops = len(cv_bboxes)
                
                parent_total = cv_metrics.get("parent_count", total_crops) if cv_metrics else total_crops
                child_total = cv_metrics.get("child_count", 0) if cv_metrics else 0
                
                async def process_crop(bbox, index, retries=3):
                    nonlocal completed_parents, completed_children, total_beams_found
                    async with sem:
                        for attempt in range(retries):
                            try:
                                rect = fitz.Rect(bbox)
                                pix = page.get_pixmap(matrix=mat, clip=rect)
                                img = Image.open(io.BytesIO(pix.tobytes("png")))
                                
                                # 存檔供你視覺除錯 (Visual Debugging)
                                os.makedirs("crops", exist_ok=True)
                                img.save(f"crops/crop_{index}.png")
                                
                                prompt = (
                                    "這是一張獨立裁切的「單跨梁或多跨單隻梁配筋詳圖」。請仔細解讀並填入 JSON 結構。\n\n"
                                    "【極度重要警告：工程圖的上下層邏輯】\n"
                                    "0. 若這張圖是一張從連續梁被切下來的「局部單跨」，你會看到主鋼筋的線條在左右兩側被直接切斷，這是正常的！這些貫穿畫面的鋼筋就是你要提取的「主體配筋」！\n"
                                    "   【絕對注意】：所謂的「邊緣殘骸」僅限於隔壁跨度外掛的散落「文字或不相關的標註」。圖中主要的鋼筋線條與標註，即使靠近邊緣或被切斷，也絕對不能忽略！\n"
                                    "   【梁編號】：即便這是單跨子圖，圖中宣告梁名稱的文字(Beam ID)也必定存在（通常在畫面中間），請務必抓出該梁編號填入 `beam_id`！\n"
                                    "1. 配筋通常分為上下配置：位於梁圖例上方的是『上層主筋(Top bars)』；位於梁下方，且「嚴格限定在該梁專屬編號的正上方或緊鄰區域」，才是『下層主筋(Bottom bars)』。絕對不可越界往下讀取到其他梁的資料！\n"
                                    "2. 鋼筋的橫向排列為「左、中、右」三個位置，請嚴格根據「當前這隻梁的實體長度邊界」來判定與擷取。絕對不可將視線延伸並讀取到隔壁左右兩側相鄰之其他梁的配筋資料！各位置可能有 1~3 排鋼筋，請依序嚴謹填寫。\n"
                                    "3. 上下排共6個位置，裡面可能有空白。空白就不填。上排至少有一個位置有數字，下排至少有一個位置有數字。\n"
                                    "4. 『絕對不可以』只輸出上層或只輸出下層，你務必要把上下兩部分的數字都找出來並分別填寫！\n"
                                    "5. 若位置有兩組以上鋼筋數量，請以陣列形式依序裝好，上排筋由上而下，下排筋由下而上。"
                                    "6. 若上層筋在往上有數字，那很有可能是lap_length，如果偏左請將數字填在lap_length_top_left，如果偏右請將數字填在lap_length_top_right。\n"
                                    "7. 若下層筋在往下有數字，那很有可能是lap_length，如果偏左請將數字填在lap_length_bottom_left，如果偏右請將數字填在lap_length_bottom_right。\n"
                                    "8. stirrups可能會分3區標示，例如：1-#4@10(左), 2-#4@10(中), 1-#4@10(右)，請將數量-#4@10的格式依序填入stirrups_left, stirrups_middle, stirrups_right。若只有中間一組數字，則只填入stirrups_middle。\n"
                                    "9. face_bars標示為E.F.，可能為 數量-#5 (E.F.)的形式\n"
                                    "10. 【自我評估審查】請在 self_confidence 填寫 0~100 的整數 (例如 98 代表幾乎完美)。一旦畫面上出現你無法歸類到標準欄位的文字/標註/神秘數字，請將 self_confidence 扣分到 80 以下，並且『強制』將這些無法歸類的文字一字不漏抄寫在 note 欄位中！"
                                )
                                
                                resp = await self.model.generate_content_async(
                                    contents=[prompt, img],
                                    generation_config=single_config,
                                    request_options={"timeout": 60}
                                )
                                
                                result_data = json.loads(resp.text)
                                crops_beams = result_data.get("beams", [])
                                
                                # 在 Python 端保證為每個跨度物件寫入程式化的 crop_index
                                for b in crops_beams:
                                    b["crop_index"] = index
                                    print(f"[Gemini Vision] 片段 {index} 成功產出：{b.get('beam_id', 'Unknown')}")
                                
                                if index <= parent_total:
                                    completed_parents += 1
                                else:
                                    completed_children += 1
                                    
                                total_beams_found += len(crops_beams)
                                if progress_cb:
                                    progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                                    
                                return crops_beams
                                
                            except Exception as e:
                                err_str = str(e).lower()
                                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                                    # 付費版若遇到短暫限流 (429)，採用較短的退避時間
                                    wait_time = 2 ** attempt
                                    print(f"[警告] 片段 {index} 遭遇短暫限流 (429)，等待 {wait_time} 秒後重試...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    print(f"[錯誤] 片段 {index} 解析拋出例外: {e}")
                                    if index <= parent_total:
                                        completed_parents += 1
                                    else:
                                        completed_children += 1
                                    if progress_cb:
                                        progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                                    return None
                                    
                        print(f"[徹底失敗] 片段 {index} 經過 {retries} 次重試仍無法解析。")
                        if index <= parent_total:
                            completed_parents += 1
                        else:
                            completed_children += 1
                        if progress_cb:
                            progress_cb(f"[Phase 2] 微觀圖塊解析中... 已發送 {completed_parents}/{parent_total} 張原始圖檔，已發送 {completed_children}/{child_total} 張分割圖檔，累積辨識出 {total_beams_found} 個梁物件。")
                        return None
                            
                tasks = [process_crop(bbox, i + 1) for i, bbox in enumerate(cv_bboxes)]
                results = await asyncio.gather(*tasks)
                
                for r_list in results:
                    if r_list is not None and isinstance(r_list, list):
                        final_beams.extend(r_list)
                        
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
