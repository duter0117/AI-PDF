from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import asyncio
from fastapi.responses import FileResponse
import os

# 自動讀取專案底下的 .env 檔案
load_dotenv()

import sys

# 將專案根目錄加入路徑，解決 python api/main.py 直接執行時找不到 workflow / core 模組的問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.drawing_agent import build_graph
from core.task_manager import tasks_db, create_task, complete_task, fail_task

app = FastAPI(title="Structural Drawing AI Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有網域請求
    allow_credentials=False, # 當 origin 為 * 時，credentials 必須是 False (標準 CORS 規範)
    allow_methods=["*"],  # 允許所有方法 (GET, POST 等)
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    # 返回我們準備好的前端 UI 頁面
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)

# 初始化 LangGraph
graph = build_graph()

# ---------------------------------------------------------
# 新版 Phase 4：非同步背景輪詢 API (不會被 30 秒斷線)
# ---------------------------------------------------------

async def run_extraction_workflow(task_id: str, pdf_bytes: bytes, page_num: int, cv_params: dict = None):
    import os
    import shutil
    # 建立以 task_id 為名的專屬隔離資料夾，避免併發 (Race Condition) 互相覆蓋圖檔
    base_crops_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crops")
    output_dir = os.path.join(base_crops_dir, task_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    if cv_params is None:
        cv_params = {}
        
    cv_params["output_dir"] = output_dir
    cv_params["debug_mode"] = os.getenv("DEBUG_MODE", "false").lower() == "true"
    try:
        import time
        start_time = time.time()
        
        result = await graph.ainvoke({
            "pdf_bytes": pdf_bytes,
            "page_num": page_num,
            "task_id": task_id,
            "cv_params": cv_params
        })
        final_output = result.get("final_output", {})
        
        # 產生全域圖片給前端
        global_image = ""
        try:
            import fitz
            import base64
            doc = fitz.Document(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]
            for beam in final_output.get("aligned_beams", []):
                rect = beam.get("spatial_anchor_rect_x_y")
                if rect and len(rect) == 4:
                    r = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                    page.draw_rect(r, color=(1, 0, 0), width=2)
                    beam_id = beam.get("beam_id", "")
                    if beam_id:
                        page.insert_text((r.x0, r.y0 - 5), str(beam_id), fontsize=10, color=(0, 0, 1))

            # 將 DPI 降至 72 以避免伺服器記憶體不足 (OOM) 或過久等待
            pix = page.get_pixmap(dpi=72)
            img_data = pix.tobytes("jpeg") # 改用 jpeg 壓縮
            global_image_b64 = base64.b64encode(img_data).decode("utf-8")
            global_image = f"data:image/jpeg;base64,{global_image_b64}"
        except BaseException as img_err:
            print(f"Failed to generate global image for background task: {img_err}")

        # 計算指標
        elapsed = round(time.time() - start_time, 2)
        api_metrics = final_output.get("api_metrics", {})
        total_tokens = api_metrics.get("prompt_tokens", 0) + api_metrics.get("candidates_tokens", 0)
        
        # 整理為與同步 API 相同的前端格式
        front_end_result = {
            "global_image": global_image,
            "execution_time_seconds": elapsed,
            "llm_calls": api_metrics.get("llm_calls", 0),
            "tokens_used": total_tokens,
            "analysis_result": final_output.get("aligned_beams", []),
            "raw_json_string": final_output.get("raw_json_string", "")
        }

        complete_task(task_id, front_end_result)
    except Exception as e:
        fail_task(task_id, str(e))

@app.post("/api/v1/task/extract")
async def create_extract_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    page_num: int = Form(0),
    dilation_iterations: int = Form(2),
    min_area: int = Form(100000),
    padding_bottom: int = Form(160),
    hough_threshold: int = Form(95),
    auto_tune: str = Form("false"),
    enable_decomp: str = Form("true"),
    voting_rounds: int = Form(1)
):
    """
    [推薦] 非同步提交圖紙任務。立即回傳 task_id 給前端，不怕 Timeout 斷線。
    前端後續需不斷輪詢 GET /api/v1/task/{task_id} 獲取即時進度。
    """
    pdf_bytes = await file.read()
    from core.task_manager import create_task
    task_id = create_task()
    
    cv_params = {
        "dilation_iterations": dilation_iterations,
        "min_area": min_area,
        "padding_bottom": padding_bottom,
        "hough_threshold": hough_threshold,
        "auto_tune": str(auto_tune).lower() == "true",
        "enable_decomp": str(enable_decomp).lower() == "true",
        "voting_rounds": max(1, min(3, int(voting_rounds)))
    }
    
    # 拋入背景任務執行
    background_tasks.add_task(run_extraction_workflow, task_id, pdf_bytes, page_num, cv_params)
    
    return {
        "status": "pending",
        "task_id": task_id,
        "message": "Task created successfully. Please poll the status endpoint."
    }

@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    """
    獲取目前 AI 解析進度。返回包含 progress (%) 與 message。
    """
    task = tasks_db.get(task_id)
    if not task:
        return {"error": "Task not found"}
    return task

# ---------------------------------------------------------
# 舊版相容：同步等待 API (若超時易斷線)
# ---------------------------------------------------------
@app.post("/api/v1/extract-drawings")
async def extract_drawings(
    file: UploadFile = File(...),
    page_num: int = Form(0),
    dilation_iterations: int = Form(2),
    min_area: int = Form(100000),
    padding_bottom: int = Form(160),
    hough_threshold: int = Form(95),
    auto_tune: str = Form("false"),
    enable_decomp: str = Form("true"),
    voting_rounds: int = Form(1)
):
    """
    [Demo預備] 上傳 PDF 工程圖說與設定參數，同步等待結果。
    回傳格式符合展示需求：全域圖預留、分析時間、LLM次數、消耗TOKEN、梁辨識結果
    """
    import time
    start_time = time.time()

    pdf_bytes = await file.read()
    cv_params = {
        "dilation_iterations": dilation_iterations,
        "min_area": min_area,
        "padding_bottom": padding_bottom,
        "hough_threshold": hough_threshold,
        "auto_tune": str(auto_tune).lower() == "true",
        "enable_decomp": str(enable_decomp).lower() == "true",
        "voting_rounds": max(1, min(3, int(voting_rounds)))
    }

    result = await graph.ainvoke({
        "pdf_bytes": pdf_bytes,
        "page_num": page_num,
        "task_id": None,
        "cv_params": cv_params
    })
    
    final = result.get("final_output", {})
    elapsed = round(time.time() - start_time, 2)
    api_metrics = final.get("api_metrics", {})
    
    # 計算 Token 與呼叫次數
    llm_calls = api_metrics.get("llm_calls", 0)
    prompt_tokens = api_metrics.get("prompt_tokens", 0)
    candidates_tokens = api_metrics.get("candidates_tokens", 0)
    total_tokens = prompt_tokens + candidates_tokens

    # 最終全域圖欄位 (可供前端放置 base64 或是圖檔 url)
    global_image = ""
    try:
        import fitz
        import base64
        doc = fitz.Document(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        
        # 繪製所有偵測到的梁框
        for beam in final.get("aligned_beams", []):
            rect = beam.get("spatial_anchor_rect_x_y")
            if rect and len(rect) == 4:
                # rect 為 [x0, y0, x1, y1]
                r = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                # 畫紅色半透明框
                page.draw_rect(r, color=(1, 0, 0), width=2)
                
                beam_id = beam.get("beam_id", "")
                if beam_id:
                    # 在框的左上角標上藍色字體的梁編號
                    page.insert_text((r.x0, r.y0 - 5), str(beam_id), fontsize=10, color=(0, 0, 1))

        # 轉譯成圖片 (DPI 150 畫質與效能平衡)
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        global_image_b64 = base64.b64encode(img_data).decode("utf-8")
        global_image = f"data:image/png;base64,{global_image_b64}"
    except Exception as e:
        print(f"Failed to generate global image: {e}")

    return {
        "global_image": global_image,
        "execution_time_seconds": elapsed,
        "llm_calls": llm_calls,
        "tokens_used": total_tokens,
        "analysis_result": final.get("aligned_beams", []),
        "raw_json_string": final.get("raw_json_string", "")
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PDF AI Backend", "env": "Zeabur Ready"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
