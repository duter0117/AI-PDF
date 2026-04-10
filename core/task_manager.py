import time
import uuid
from typing import Dict, Any

# 全域記憶體工作池，供 PoC 階段與輕量線上部屬使用
# 若未來要展開多節點架構，可輕鬆置換成 Redis
tasks_db: Dict[str, Any] = {}

_MAX_TASK_AGE_SECONDS = 1800  # 30 分鐘 TTL

def _cleanup_old_tasks():
    """清除超過 TTL 的舊 task，防止記憶體無限膨脹"""
    now = time.time()
    expired = [k for k, v in tasks_db.items() if now - v.get("created_at", 0) > _MAX_TASK_AGE_SECONDS]
    for k in expired:
        del tasks_db[k]

def create_task() -> str:
    _cleanup_old_tasks()
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任務已建立，等待排程器分配資源...",
        "created_at": time.time(),
        "result": None
    }
    return task_id


def update_task_progress(task_id: str, progress: int, message: str, status: str = "processing"):
    if task_id in tasks_db:
        tasks_db[task_id]["progress"] = progress
        tasks_db[task_id]["message"] = message
        tasks_db[task_id]["status"] = status

def complete_task(task_id: str, result: dict):
    if task_id in tasks_db:
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["message"] = "分析完成"
        tasks_db[task_id]["status"] = "success"
        tasks_db[task_id]["result"] = result

def fail_task(task_id: str, error_msg: str):
    if task_id in tasks_db:
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["message"] = f"分析失敗: {error_msg}"
        tasks_db[task_id]["status"] = "failed"
