import pandas as pd
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema


# Pydantic schema for the LLM’s structured output
class CommandPandasOutput(BaseModel):
    command_pandas: str = Field(
        ...,
        description="Pandas command string to execute immediately, e.g., "
                    "df[['id', 'ten_nganh', 'nam', 'phuong_thuc', 'loai_xet', 'diem_chuan']].query('nam == 2024')"
    )

# Input schema for our tool
class DiemChuanInput(BaseModel):
    question: str = Field(description="Câu hỏi về điểm chuẩn cần truy vấn")

class DiemChuanTool(BaseTool):
    """Tool để chuyển câu hỏi tuyển sinh thành và thực thi câu lệnh pandas."""
    name: str = "DiemChuanTool"
    description: str = (
        "Nhận câu hỏi về điểm chuẩn đại học, "
        "sinh câu lệnh pandas để truy vấn DataFrame `df`, "
        "và trả về kết quả. Nếu lệnh pandas lỗi, in ra lỗi kèm lệnh."
    )
    args_schema: Optional[ArgsSchema] = DiemChuanInput
    return_direct: bool = False

    # Non-field attributes for LLM and DataFrame
    _llm: Any = PrivateAttr()
    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, llm: Any, df: pd.DataFrame):
        """Khởi tạo tool với một LLM và DataFrame đầu vào."""
        super().__init__()
        # Assign to private attributes
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, '_df', df)

    # Template prompt
    PROMPT_TEMPLATE: str = """
Bạn là trợ lý tư vấn tuyển sinh đại học chuyên phân tích điểm chuẩn. 
Nhiệm vụ của bạn là chuyển câu hỏi người dùng thành câu lệnh pandas để truy vấn dữ liệu.

Bạn chỉ trả về câu lệnh pandas Python, không giải thích, không ghi chú.

DataFrame có tên df với các cột: id, ten_nganh, nam, phuong_thuc, loai_xet, diem_chuan.

Cấu trúc dữ liệu:
- phuong_thuc: 1=Điểm THPT, 2=Học bạ, 3=Đánh giá năng lực
- loai_xet: 'Điểm thi THPT', 'Điểm học bạ THPT', 'Điểm học bạ cả năm lớp 12', 'Điểm học bạ lớp 10, 11 và HK1 lớp 12', 'Đánh giá năng lực' 
- ten_nganh: Chuẩn hóa tên ngành theo danh sách đã định sẵn:
    'Công nghệ thực phẩm', 'Đảm bảo chất lượng & ATTP', 'Công nghệ chế biến thủy sản', 'Quản trị kinh doanh', 'Kế toán',
    'Tài chính ngân hàng', 'Công nghệ kỹ thuật hóa học', 'Công nghệ vật liệu', 'Công nghệ kỹ thuật môi trường',
    'Công nghệ sinh học', 'Công nghệ thông tin', 'Công nghệ dệt, may', 'Công nghệ chế tạo máy', 'Công nghệ kỹ thuật cơ điện tử',
    'Công nghệ kỹ thuật điện - điện tử', 'Khoa học dinh dưỡng và ẩm thực', 'Quản trị dịch vụ du lịch và lữ hành',
    'Quản lý tài nguyên và môi trường', 'An toàn thông tin', 'Công nghệ kỹ thuật điều khiển và tự động hóa',
    'Quản trị nhà hàng và dịch vụ ăn uống', 'Ngôn ngữ Anh', 'Khoa học chế biến món ăn', 'Khoa học thủy sản',
    'Kinh doanh quốc tế', 'Luật kinh tế', 'Quản trị khách sạn', 'Ngôn ngữ Trung Quốc', 'Quản trị kinh doanh thực phẩm',
    'Marketing', 'Kỹ thuật hóa phân tích', 'Kinh doanh thời trang và Dệt may', 'Kỹ thuật Nhiệt', 'Quản lý năng lượng',
    'Thương mại điện tử', 'Công nghệ tài chính', 'Khoa học dữ liệu', 'Logistics và quản lý chuỗi cung ứng'

Lưu ý:
1. Luôn luôn query tất cả các cột (df[['id', 'ten_nganh', 'nam', 'phuong_thuc', 'loai_xet', 'diem_chuan']])
2. Đảm bảo truy vấn chính xác tên ngành từ danh sách
3. Sử dụng các phương thức pandas như .query(), .loc[], .iloc[] hoặc boolean indexing
4. Sắp xếp kết quả phù hợp với yêu cầu (thường là .sort_values()) và ưu tiên thời gian gần nhất
5. Với các câu hỏi không đề cập đến chữ nhất (cao nhất, thấp nhất, ...) và yêu cầu cụ thể thì hiển thị tối đa 50 dòng

Thời gian hiện tại: {current_time}

Input: 
Question: {question}
"""

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Build prompt and call LLM
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        llm_response = self._llm.with_structured_output(CommandPandasOutput).invoke(prompt)
        command = llm_response.command_pandas.strip()
        # Try to execute the generated pandas command
        try:
            result_df = eval(command, {"df": self._df})

            # Check if result is actually a DataFrame
            if isinstance(result_df, pd.DataFrame):
                result_string = result_df.to_string(index=False)
            else:
                result_string = str(result_df)

            return (
                f"Generated pandas command:\n{command}\n\n"
                f"Query result:\n{result_string}"
            )
        except Exception as e:
            # If there's an error, include the generated command in the message
            return (
                f"Error executing pandas command:\n"
                f"{e}\n\n"
                f"Generated command was:\n{command}"
            )

    async def _arun(
        self,
        question: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        # Build prompt and call LLM
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        llm_response = await self._llm.with_structured_output(CommandPandasOutput).ainvoke(prompt)
        command = llm_response.command_pandas.strip()
        # Try to execute the generated pandas command
        try:
            result_df = eval(command, {"df": self._df})

            # Check if result is actually a DataFrame
            if isinstance(result_df, pd.DataFrame):
                result_string = result_df.to_string(index=False)
            else:
                result_string = str(result_df)

            return (
                f"Generated pandas command:\n{command}\n\n"
                f"Query result:\n{result_string}"
            )
        except Exception as e:
            # If there's an error, include the generated command in the message
            return (
                f"Error executing pandas command:\n"
                f"{e}\n\n"
                f"Generated command was:\n{command}"
            )
