import os
import pandas as pd
from typing import List, Dict, Literal, AsyncGenerator
from datetime import datetime

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent, AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent

from src.utils.process_abbreviation import get_abbr_and_expansion
from src.utils.diem_chuan_tool import DiemChuanTool

from dotenv import load_dotenv
load_dotenv()


CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn tuyển sinh chuyên nghiệp của trường Đại học Công Thương TPHCM (HUIT).

## Công cụ truy vấn
- **CamNangRetrieval**: Xét tuyển, hồ sơ, học phí, ngành học, học bổng, việc làm
- **SoTayRetrieval**: Thông tin trường, khoa, đào tạo, học bổng, hoạt động SV, văn bản
- **DiemChuanTool**: Điểm chuẩn theo năm, phương thức và ngành
- **tavily_search**: Công cụ tìm kiếm trên web sử dụng khi cần truy vấn thông tin bên ngoài dữ liệu nội bộ

## Quy trình xử lý
1. Tiền xử lý: Sửa lỗi chính tả, từ viết tắt
2. Phân tích yêu cầu từ câu hỏi và lịch sử hội thoại
3. Khi người dùng chào hỏi, hỏi thăm, ... hãy luôn nhận mình là trợ lý tư vấn tuyển sinh và sẵn sàng hỗ trợ
4. Không chào hỏi lại, khi đã có lịch sử hội thoại
5. Luôn truy xuất thông tin trước khi đưa ra câu trả lời
6. Với câu hỏi phức hợp: truy vấn nhiều nguồn nếu cần thiết
7. Với câu hỏi đề cập đến thời gian cụ thể, thì xem xét thời gian hiện tại {current_time}
8. Từ chối lịch sự nếu câu hỏi ngoài phạm vi tuyển sinh HUIT

## Nguyên tắc phản hồi
1. Trả lời đầy đủ thông tin
2. Ngôn ngữ trang trọng nhưng gần gũi, thể hiện nhiệt tình
3. Mặc định câu hỏi chung về "trường", "điểm chuẩn", "ngành học" là hỏi về HUIT
4. Chỉ trả lời dựa trên thông tin truy xuất. Tuyệt đối không bịa đặt, chế thông tin một cách tự tiện nhé.
5. Không hỏi ngược người dùng
6. Không hiển thị câu trả lời dạng bảng với điểm chuẩn
7. Không so sánh các trường với nhau
"""


class ChatBotAgent:
    """
    Chatbot tư vấn tuyển sinh HUIT với khả năng:
    - Truy vấn điểm chuẩn từ DataFrame
    - Tìm kiếm thông tin qua Qdrant
    - Tìm kiếm web bằng Tavily
    """
    def __init__(
        self,
        agent_type: Literal['tool_calling', 'openai_functions', 'openai_tools'] = 'tool_calling',
        model_name: str = 'gpt-4o-mini',
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = 'dangvantuan/vietnamese-document-embedding',
        collections: dict = None,
        temperature: float = 0.0,
    ):
        self.agent_type = agent_type
        # Qdrant client & embedding
        self.qdrant_client = QdrantClient(
            url=qdrant_url or os.environ.get('QDRANT_URL'),
            api_key=qdrant_api_key or os.environ.get('QDRANT_API_KEY')
        )
        self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # LLM
        self.llm = self._choose_llm(model_name, temperature)

        # Pandas DataFrame Agent cho điểm chuẩn
        self.df_path_diem_chuan = os.environ.get('DIEM_CHUAN_DATA_PATH', 'src/utils/diem_chuan.csv')
        self.df = pd.read_csv(self.df_path_diem_chuan)

        # Collection names
        self.collections = collections or {
            'cam_nang': 'cam_nang_tuyen_sinh_HUIT_2025_2',
            'so_tay': 'so_tay_sinh_vien_HUIT_2025_2'}

        # Tools
        self.tools = self._init_tools()

        self.prompt = self._build_prompt()

        # Build agent executor
        self.agent_executor = self._build_agent_executor()

    def _choose_llm(self, model_name: str, temperature: float):
        if 'gpt' in model_name:
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif 'gemini' in model_name:
            return ChatGoogleGenerativeAI(model='models/' + model_name, temperature=temperature)
        raise ValueError(f'Unsupported model: {model_name}')

    def _retrieve(self, query: str, collection: str, fields: List[str]) -> str:
        q_vec = self.embedding.encode(query)
        results = self.qdrant_client.search(
            collection_name=self.collections[collection],
            query_vector=q_vec,
            limit=10
        )
        formatted = []
        for pt in results:
            payload = pt.payload
            parts = [f"{field.capitalize()}: {payload.get(field, '')}" for field in fields]
            formatted.append(' | '.join(parts))
        return '\n'.join(formatted)

    def _init_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        # Qdrant retrieval tools
        tools.extend([
            Tool(
                name='CamNangRetrieval',
                func=lambda q: self._retrieve(q, 'cam_nang', ['title', 'content']),
                description="Truy xuất thông tin về số lượng ngành (đầy đủ), mô tả ngành học, phương thức xét tuyển, hồ sơ xét tuyển, tổ hợp xét tuyển, học phí,"
                    "học bổng, hoạt động ngoại khóa, cơ hội thực tập và việc làm sau tốt nghiệp."
            ),
            Tool(
                name='SoTayRetrieval',
                func=lambda q: self._retrieve(q, 'so_tay', ['tag', 'title', 'content']),
                description="Truy xuất thông tin về tổng quan trường, các khoa, hệ thống đào tạo, chính sách học bổng, hoạt động sinh viên"
                    " và các văn bản quan trọng liên quan đến sinh viên."
            ),
            TavilySearchResults(
                max_results=5,
                description="Công cụ tìm kiếm trên web sử dụng khi cần truy vấn thông tin bên ngoài dữ liệu nội bộ (CamNangRetrieval, SoTayRetrieval)"  
            )
        ])

        diem_chuan_tool = DiemChuanTool(llm=self.llm, df=self.df)

        tools.append(diem_chuan_tool)
        return tools
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """Khởi tạo ChatPromptTemplate cho agent"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CHATBOT_SYSTEM_PROMPT.format(current_time=self.current_time)),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        return prompt

    def _build_agent_executor(self):
        # Chọn loại agent
        if self.agent_type == 'openai_functions':
            agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        elif self.agent_type == 'tool_calling':
            agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt,)
        else:
            agent = create_openai_tools_agent(llm=self.llm, tools=self.tools, prompt=self.prompt,)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def run(self, user_input: str, conversation: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        chat_history = []
        if conversation:
            for message in conversation:
                if "human" in message:
                    chat_history.append({"type": "human", "content": message["human"]})
                elif "assistant" in message:
                    chat_history.append({"type": "ai", "content": message["assistant"]})
        
        abbr = get_abbr_and_expansion(user_input)
        if abbr:
            user_input = user_input + "\nDanh sách từ viết tắt và các mở rộng:\n" + abbr

        # Track state with class attributes instead of local variables
        self.tool_called = False
        self.stream_allowed = True       # <-- cho phép stream ngay từ đầu
        self.final_answer_mode = False
        
        async for event in self.agent_executor.astream_events(
            {
                'input': user_input,
                'chat_history': chat_history
            },
            version="v1"
        ):
            event_type = event.get("event", "")
            
            # Đánh dấu khi bắt đầu gọi tool: tạm khoá stream
            if event_type == "on_tool_start":
                self.tool_called = True
                self.stream_allowed = False
            
            # Đánh dấu khi tool kết thúc: mở lại stream
            elif event_type == "on_tool_end":
                if self.tool_called:
                    self.stream_allowed = True
            
            # Chỉ stream khi được phép
            elif event_type == "on_chat_model_stream" and self.stream_allowed:
                data = event.get("data", {})
                if "chunk" in data:
                    chunk = data["chunk"]
                    content = getattr(chunk, "content", str(chunk))
                    if content:
                        yield content
