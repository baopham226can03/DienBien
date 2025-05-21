import pandas as pd
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type


from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

# Pydantic schema for the LLM’s structured output
class RecommendationOutput(BaseModel):
    recommendations: list[str] = Field(
        ...,
        description="List of recommended documents or resources relevant to the user's question."
    )

# Input schema for the tool
class RecommendInput(BaseModel):
    question: str = Field(description="Câu hỏi cần gợi ý tài liệu liên quan")
    documents: str = Field(description="Danh sách tài liệu từ công cụ _retrieve")

class RecommendTool(BaseTool):
    """Tool để gợi ý các tài liệu liên quan dựa trên câu hỏi và danh sách tài liệu."""
    name: str = "RecommendTool"
    description: str = (
        "Nhận các tài liệu liên quan rồi phân tích, trả về gợi ý các tài liệu thật sự liên quan đến câu hỏi từ 10 tài liệu đã chọn lọc ở bước _retrieve"
    )
    args_schema: Type[BaseModel] = RecommendInput
    return_direct: bool = False

    # Non-field attributes for LLM
    _llm: Any = PrivateAttr()

    def __init__(self, llm: Any):
        """Khởi tạo tool với một LLM đầu vào."""
        super().__init__()
        object.__setattr__(self, '_llm', llm)

    # Template prompt
    PROMPT_TEMPLATE: str = """
Bạn là trợ lý gợi ý tài liệu chuyên nghiệp. 
Nhiệm vụ của bạn là dựa trên câu hỏi người dùng và danh sách tài liệu từ công cụ _retrieve để gợi ý các tài liệu phù hợp nhất.

Bạn chỉ trả về danh sách các tài liệu/tài nguyên liên quan~~Bạn chỉ trả về danh sách các tài liệu/tài nguyên liên quan dưới dạng danh sách Python, không giải thích, không ghi chú.
Mỗi mục trong danh sách là một chuỗi mô tả tài liệu/tài nguyên (ví dụ: tiêu đề, tên file, hoặc mô tả ngắn gọn).
Số lượng tài liệu gợi ý không vượt quá 10 mục.

Thời gian hiện tại: {current_time}

Input:
Question: {question}
Documents: {documents}
"""

    def _run(
        self,
        question: str,
        documents: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if not documents or "Không tìm thấy" in documents:
            return "Không có tài liệu nào phù hợp với câu hỏi."
        
        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            documents=documents,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        llm_response = self._llm.with_structured_output(RecommendationOutput).invoke(prompt)
        recommendations = llm_response.recommendations
        
        if not recommendations:
            return "Không tìm thấy tài liệu phù hợp sau khi phân tích."
        
        result_string = "\n".join([f"- {rec}" for rec in recommendations])
        return f"Danh sách tài liệu gợi ý:\n{result_string}"

    async def _arun(
        self,
        question: str,
        documents: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        if not documents or "Không tìm thấy" in documents:
            return "Không có tài liệu nào phù hợp với câu hỏi."
        
        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            documents=documents,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        llm_response = await self._llm.with_structured_output(RecommendationOutput).ainvoke(prompt)
        recommendations = llm_response.recommendations
        
        if not recommendations:
            return "Không tìm thấy tài liệu phù hợp sau khi phân tích."
        
        result_string = "\n".join([f"- {rec}" for rec in recommendations])
        return f"Danh sách tài liệu gợi ý:\n{result_string}"