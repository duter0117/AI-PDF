import os
import cv2
import fitz
import numpy as np

def test_opencv_crop():
    pdf_path = r"..\TEST15.pdf"
    output_dir = "crops"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"載入 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[0]

    # 1. 將 PDF 頁面渲染為高解析度灰階圖片
    mat = fitz.Matrix(4.0, 4.0) # 放非常大，確保細節都在
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    
    # 將 Pixmap 轉為純 numpy 陣列供 OpenCV 使用
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    
    # 2. 二值化：將有墨水的地方變白色 (255)，背景變黑色 (0)
    # 通常黑線在 PDF 中數值為 0 或小於 100
    _, thresh = cv2.threshold(img_data, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. 滴墨水：形態學膨脹 (Morphological Dilation)
    # 剛才的 70x70 太大，導致全部的梁跟「外框線」互相交疊黏死成一個畫面。
    # 改用 15x15 的小筆刷，只讓文字跟鋼筋本身彼此融合，避免越界黏到隔壁梁
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 4. 根據融合成一塊塊的墨水區，找出它們的外框
    # 改用 RETR_LIST 避免被最大的 PDF 頁面圖紙外框給「反客為主」吃掉內部所有的梁
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"在圖面上找到了 {len(contours)} 個墨水聚落。開始面積篩選...")
    
    # 重新渲染一張全彩圖準備用來純裁切，不再用灰階
    color_pix = page.get_pixmap(matrix=mat)
    color_img = np.frombuffer(color_pix.samples, dtype=np.uint8).reshape(color_pix.height, color_pix.width, 3)

    count = 1
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # 5. 過濾掉雜訊、文字單體，以及超過整張圖 25% 的超大外框
        if area > 100000 and area < (pix.width * pix.height * 0.25):
            # 加點邊緣緩衝 padding
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(pix.width, x + w + pad)
            y2 = min(pix.height, y + h + pad)
            
            crop_img = color_img[y1:y2, x1:x2]
            
            out_path = os.path.join(output_dir, f"crop_CV2_{count}.png")
            # 注意這裡用 OpenCV 存圖，OpenCV 預設是 BGR，但 PyMuPDF 吐的是 RGB，轉一下
            crop_out = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, crop_out)
            print(f"Success! Saved beam image: {out_path} (Area: {area})")
            count += 1

    print(f"\n全部 OpenCV 影像強拆完成！共成功輸出 {count-1} 張小圖。")

if __name__ == "__main__":
    test_opencv_crop()
