import asyncio
import os
import json
import time

from workflow.drawing_agent import build_graph

async def main():
    graph = build_graph()
    pdf_path = r"C:\Users\USER\Desktop\PDF AI\python-ai-backend\benchmarks\3.pdf"
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    print("Running graph...")
    t0 = time.time()
    try:
        result = await graph.ainvoke({
            "pdf_bytes": pdf_bytes, 
            "page_num": 0,
            "task_id": None, 
            "cv_params": {
                "dilation_iterations": 2,
                "min_area": 100000,
                "padding_bottom": 160,
                "hough_threshold": 95,
                "enable_decomp": True,
                "voting_rounds": 1
            }
        })
        print(f"Elapsed: {time.time() - t0:.1f}s")
        
        final = result.get("final_output", {})
        aligned = final.get("aligned_beams", [])
        print(f"Result count: {len(aligned)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
