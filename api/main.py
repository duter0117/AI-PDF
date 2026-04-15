from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
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
    # 每次任務前清空 crops 資料夾
    crops_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crops")
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir, ignore_errors=True)
    os.makedirs(crops_dir, exist_ok=True)

    if cv_params is None:
        cv_params = {}
    try:
        result = await graph.ainvoke({
            "pdf_bytes": pdf_bytes,
            "page_num": page_num,
            "task_id": task_id,
            "cv_params": cv_params
        })
        final_output = result.get("final_output", {})
        
        from core.archiver import create_run_dir, archive_item
        run_dir = create_run_dir("API非同步")
        # 前端上傳通常沒有好檔名，我們用個預設加上部分 task_id 作識別
        archive_item(run_dir, f"api_upload_{task_id[:6]}.pdf", pdf_bytes, final_output)

        complete_task(task_id, final_output)
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
    page_num: int = Form(0)
):
    """
    上傳 PDF 工程圖說，同步等待結果 (約 20~40 秒)。
    """
    pdf_bytes = await file.read()
    result = await graph.ainvoke({
        "pdf_bytes": pdf_bytes,
        "page_num": page_num,
        "task_id": None,
        "cv_params": {}
    })
    
    final_output = result.get("final_output", {"error": "Pipeline failed"})
    
    from core.archiver import create_run_dir, archive_item
    run_dir = create_run_dir("API同步")
    original_name = file.filename if file.filename else "api_upload.pdf"
    archive_item(run_dir, original_name, pdf_bytes, final_output)
    
    return final_output

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PDF AI Backend", "env": "Zeabur Ready"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
