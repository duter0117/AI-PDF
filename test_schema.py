import os
import asyncio
import PIL.Image
from dotenv import load_dotenv
import google.generativeai as genai
from core.table_extractor import BeamList

load_dotenv()

async def test():
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=BeamList
    )
    
    img = PIL.Image.new('RGB', (100, 100))
    try:
        res = await model.generate_content_async(["請給我一個空的測試資料", img], generation_config=config)
        print("Success:", res.text)
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")

asyncio.run(test())
