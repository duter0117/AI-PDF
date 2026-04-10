import os
import asyncio
from dotenv import load_dotenv
import fitz
import json

from workflow.drawing_agent import build_graph
from core.vector_extractor import VectorExtractor
from core.table_extractor import TableExtractor

load_dotenv()

async def debug_graph():
    pdf_path = r"..\TEST15.pdf"
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("=== Step 1: 測試 Vector Extraction (OpenCV) ===", flush=True)
    v = VectorExtractor(pdf_bytes)
    cv_bboxes = v.extract_opencv_bboxes(0)[:1] # 只測 1 個，避免等待 35 秒
    print(f"提取了 {len(cv_bboxes)} 個 bbox", flush=True)
    
    print("=== Step 2: 測試 Table Extractor (Gemini) ===", flush=True)
    t = TableExtractor()
    try:
        json_str = await t.extract_tables(pdf_bytes, 0, cv_bboxes)
        print("Gemini 回傳字串: ", json_str, flush=True)
    except Exception as e:
        print(f"發生未捕捉錯誤: {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(debug_graph())
