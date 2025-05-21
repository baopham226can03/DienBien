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
                    "df[['ma_nganh','ten_nganh','to_hop','mon']].query('ma_nganh == 7340101')"
    )

# Input schema for our tool
class ToHopXetTuyenInput(BaseModel):
    question: str = Field(description="Câu hỏi về các tổ hợp xét tuyển cần truy vấn")

class ToHopXetTuyenTool(BaseTool):
    """Tool để chuyển câu hỏi tuyển sinh thành và thực thi câu lệnh pandas."""
    name: str = "ToHopXetTuyenTool"
    description: str = (
        "Nhận câu hỏi về các tổ hợp xét tuyển đại học, "
        "sinh câu lệnh pandas để truy vấn DataFrame `df`, "
        "và trả về kết quả. Nếu lệnh pandas lỗi, in ra lỗi kèm lệnh."
    )
    args_schema: Optional[ArgsSchema] = ToHopXetTuyenInput
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
Bạn là trợ lý tư vấn tuyển sinh đại học chuyên về các tổ hợp xét tuyển đại học ở Việt Nam. 
Nhiệm vụ của bạn là chuyển câu hỏi người dùng thành câu lệnh pandas để truy vấn dữ liệu.

Bạn chỉ trả về câu lệnh pandas Python, không giải thích, không ghi chú.

DataFrame có tên df với các cột: ma_nganh, ten_nganh, to_hop, mon.

Cấu trúc dữ liệu:
- ma_nganh: '7......'
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
- to_hop: 'D14', 'D09', 'D01', 'A01', 'C01', 'A00', 'D07', 'B00', 'C02', 'C14', 'C03', 'C00', 'B08', 'K01', 'D15'
- mon: Các môn xét tuyển....

Lưu ý:
1. Đảm bảo truy vấn chính xác tên ngành từ danh sách
2. Sử dụng các phương thức pandas như .query(), .loc[], .iloc[] hoặc boolean indexing
3. Khi yêu cầu liệt kê các tổ hợp môn, nhớ query cả cột mon

Input: 
Question: {question}
"""

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Build prompt and call LLM
        prompt = self.PROMPT_TEMPLATE.format(question=question)
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
        prompt = self.PROMPT_TEMPLATE.format(question=question)
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
