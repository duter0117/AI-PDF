"""
CropContext — 單一圖塊的物件管理器
貫穿 Pass1 → 物理裁切 → Pass2 → Debug 繪圖 → Gemini 推論
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from PIL import Image


class ObjectStatus(Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class DetectedLine:
    """單一偵測到的線段物件"""
    kind: str                    # "h_beam_edge" | "h_dimension" | "v_column" | "v_leader" | "v_rejected"
    status: ObjectStatus
    reject_reason: str = ""      # "標註線" | "引線" | "幾何不符" | "線寬不符" | ""
    x: float = 0
    y: float = 0
    w: float = 0                 # 線寬 (thickness)
    h: float = 0                 # 線高/長
    # 引線追蹤資訊
    lx: float = 0               # 狗腿左端 X
    rx: float = 0               # 狗腿右端 X
    free_y: float = 0           # 引線自由端 Y
    # 絕對 PDF 座標
    abs_x: float = 0
    abs_y: float = 0
    abs_w: float = 0
    abs_h: float = 0
    abs_lx: float = 0
    abs_rx: float = 0
    abs_free_y: float = 0
    score_text: str = ""


@dataclass
class CropContext:
    """單一圖塊的物件管理器"""
    # 圖片與 OCR
    img: Optional[Image.Image] = None
    ocr_items: list = field(default_factory=list)
    ocr_hint: str = ""

    # 偵測到的所有物件
    lines: list = field(default_factory=list)  # List[DetectedLine]

    # 結論性座標 (由 pass 1/2 更新)
    beam_top: Optional[float] = None
    beam_bottom: Optional[float] = None
    beam_stroke: float = 1.0     # 平均梁線寬 (用於柱線線寬比對)
    beam_name: str = ""          # 從 OCR 篩選出的確認梁名
    ref_col_stroke: float = 0.0  # 上一輪偵測的柱位線寬 (用於第二次擷取的比對加分)
    left_col: Optional[float] = None
    right_col: Optional[float] = None
    all_cols_sorted: list = field(default_factory=list)

    # 物理裁切偏移紀錄
    crop_offset_x: float = 0

    # 絕對座標轉換輔助
    pdf_bbox: tuple = (0, 0, 0, 0)  # 母塊在原始 PDF 的 (x0, y0, x1, y1)
    pdf_scale: float = 3.0          # 影像解析度相對於 PDF 單位 (72 DPI) 的放大倍率

    def to_pdf_x(self, x: float) -> float:
        """將當前畫布的 X 座標，還原回原始 PDF 座標"""
        original_x = x + self.crop_offset_x
        return self.pdf_bbox[0] + original_x / self.pdf_scale
        
    def to_pdf_y(self, y: float) -> float:
        """將當前畫布的 Y 座標，還原回原始 PDF 座標"""
        return self.pdf_bbox[1] + y / self.pdf_scale

    def to_pdf_w(self, w: float) -> float:
        """將畫布寬度轉換為 PDF 寬度"""
        return w / self.pdf_scale

    # === 查詢工具 ===
    def get_lines(self, kind: str = None, status: ObjectStatus = None):
        """取得指定類型+狀態的物件"""
        result = self.lines
        if kind is not None:
            result = [line for line in result if line.kind == kind]
        if status is not None:
            result = [line for line in result if line.status == status]
        return result

    def reject(self, line: DetectedLine, reason: str):
        """將一條線判定為淘汰"""
        line.status = ObjectStatus.REJECTED
        line.reject_reason = reason

    def shift_after_crop(self, left_offset: float):
        """物理裁切後，所有物件座標做 X 偏移"""
        self.crop_offset_x += left_offset
        for line in self.lines:
            line.x -= left_offset
            line.lx -= left_offset
            line.rx -= left_offset
        for item in self.ocr_items:
            item["min_x"] -= left_offset
            item["max_x"] -= left_offset
            item["cx"] -= left_offset
        # 更新結論座標
        if self.left_col is not None:
            self.left_col -= left_offset
        if self.right_col is not None:
            self.right_col -= left_offset
        self.all_cols_sorted = [c - left_offset for c in self.all_cols_sorted]

    def clear_lines(self):
        """清除所有偵測線段（用於 Pass 2 重新偵測）"""
        self.lines.clear()
        self.beam_top = None
        self.beam_bottom = None
        self.beam_stroke = 1.0
        self.beam_name = ""
        self.left_col = None
        self.right_col = None
        self.all_cols_sorted = []
