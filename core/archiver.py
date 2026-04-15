import os
import shutil
import json
from datetime import datetime

def create_run_dir(prefix=""):
    """建立一個以日期時間命名的資料夾，並可選擇附加前綴識別來源"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finished"))
    os.makedirs(base_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{prefix}" if prefix else ts
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def archive_item(run_dir, pdf_name, pdf_bytes, attr_dict, crops_dir="crops"):
    """將單一 PDF 相關的「原圖」、「屬性(JSON)」及「全域圖」封存"""
    base_name = os.path.splitext(pdf_name)[0]
    
    # 1. 存原圖 (.pdf)
    pdf_path = os.path.join(run_dir, f"{base_name}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    # 2. 存屬性 (.json)
    json_path = os.path.join(run_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(attr_dict, f, ensure_ascii=False, indent=2)
        
    # 3. 存全域圖 (.png) - 從 crops 目錄複製
    # 因為系統每次跑一張圖都會在 crops 覆寫 debug_full_pdf.png，所以要每跑完一張就備份出來
    crops_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", crops_dir))
    debug_png = os.path.join(crops_dir_path, "debug_full_pdf.png")
    if os.path.exists(debug_png):
        dst_png = os.path.join(run_dir, f"{base_name}_全域圖.png")
        shutil.copy2(debug_png, dst_png)

def archive_report(run_dir, html_path):
    """將評測跑出來的 HTML 報告歸檔"""
    if html_path and os.path.exists(html_path):
        dst_html = os.path.join(run_dir, os.path.basename(html_path))
        shutil.copy2(html_path, dst_html)
