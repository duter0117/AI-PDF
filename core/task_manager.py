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

def _elapsed(task):
    """回傳自建立以來的秒數"""
    return round(time.time() - task.get("created_at", time.time()), 1)

def create_task() -> str:
    _cleanup_old_tasks()
    task_id = str(uuid.uuid4())
    now = time.time()
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任務已建立，等待排程器分配資源...",
        "created_at": now,
        "result": None,
        "steps": [{"t": 0, "msg": "任務已建立，等待排程器分配資源..."}]
    }
    return task_id


def update_task_progress(task_id: str, progress: int, message: str, status: str = "processing"):
    if task_id in tasks_db:
        task = tasks_db[task_id]
        task["progress"] = progress
        task["message"] = message
        task["status"] = status
        # 只有訊息變化時才追加步驟，避免重複堆積相同訊息
        if not task["steps"] or task["steps"][-1]["msg"] != message:
            task["steps"].append({"t": _elapsed(task), "msg": message})

def complete_task(task_id: str, result: dict):
    if task_id in tasks_db:
        task = tasks_db[task_id]
        task["progress"] = 100
        task["message"] = "分析完成"
        task["status"] = "success"
        task["result"] = result
        task["steps"].append({"t": _elapsed(task), "msg": "✅ 分析圓滿完成！"})

def fail_task(task_id: str, error_msg: str):
    if task_id in tasks_db:
        task = tasks_db[task_id]
        task["progress"] = 100
        task["message"] = f"分析失敗: {error_msg}"
        task["status"] = "failed"
        task["steps"].append({"t": _elapsed(task), "msg": f"❌ 分析失敗: {error_msg}"})
