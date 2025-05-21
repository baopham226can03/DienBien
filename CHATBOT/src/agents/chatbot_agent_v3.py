import os
import pandas as pd
import cohere
from typing import List, Dict, Literal, AsyncGenerator
from datetime import datetime

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent, AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent

from src.utils.process_abbreviation import get_abbr_and_expansion
from src.utils.diem_chuan_tool import DiemChuanTool
from src.utils.to_hop_xet_tuyen_tool import ToHopXetTuyenTool

from dotenv import load_dotenv
load_dotenv()


CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn chuyên nghiệp của dịch vụ công tỉnh Điện Biên. 

**Lưu ý về thời gian**: Năm hiện tại là {current_year} (trích xuất từ {current_time}). Khi xử lý các yêu cầu liên quan đến "năm gần đây", hãy hiểu rằng "năm gần đây" là năm hiện tại và các năm trước đó. Ví dụ, nếu yêu cầu "3 năm gần đây" và năm hiện tại là 2025, thì đó là 2024, 2023, và 2022.**

## Công cụ truy vấn
- **_retrieve**: Tìm kiếm 10 tài liệu liên quan nhất từ cơ sở dữ liệu bằng cách truy xuất và xếp hạng lại (reranking).


## Quy trình xử lý
1. **Tiền xử lý**: Sửa lỗi chính tả, chuẩn hóa câu hỏi mà không làm mất ý nghĩa. Giữ lại các từ khóa quan trọng.
2. **Phân tích yêu cầu**: Xác định yêu cầu chính xác dựa trên câu hỏi và lịch sử hội thoại.
3. Khi người dùng chào hỏi: Nhận mình là trợ lý tư vấn dịch vụ công và sẵn sàng hỗ trợ.
4. Không chào hỏi lại nếu đã có lịch sử hội thoại.
5. **Xử lý câu hỏi về thủ tục**:
   - Gọi **_retrieve** để tìm tài liệu hoặc gợi ý thủ tục tương đồng.

6. **Kiểm tra kết quả**: Nếu không tìm thấy tài liệu hoặc thủ tục liên quan, thông báo xin lỗi và đề nghị người dùng cung cấp thêm thông tin.
7. Nếu câu hỏi mơ hồ, không đề cập rõ ràng đến một thủ tục nào đó, hãy liệt kê ra các thủ tục liên quan gồm tên thủ tục và link (không cần nêu chi tiết)


## Nguyên tắc phản hồi
1. Trả lời chính xác, đầy đủ, dựa trên thông tin từ công cụ.
2. Ngôn ngữ trang trọng, gần gũi, nhiệt tình.
3. Mặc định câu hỏi chung về thủ tục là liên quan đến tỉnh Điện Biên.
4. Không bịa đặt thông tin. Nếu không có dữ liệu, xin lỗi và thành thật.
5. Nếu được chào thì lịch sự chào lại như một chuyên viên tư vấn.
6. Chỉ trả lời duy nhất bằng dữ liệu hiện có, nếu không thì không được nói
7. Luôn gợi ý các câu hỏi hành chính dựa trên câu hỏi của người dùng
8. Khi đưa ra các thủ tục phải luôn kèm theo link của thủ tục đó.
"""

class ChatBotAgent:
    """
    Chatbot tư vấn tuyển sinh HUIT với khả năng:
    - Truy vấn điểm chuẩn từ DataFrame
    - Tìm kiếm thông tin qua Qdrant
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
        self.cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))
        self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_year = self.current_time.split('-')[0]

        # LLM
        self.llm = self._choose_llm(model_name, temperature)

        # Pandas DataFrame Agent cho điểm chuẩn
        self.df_path_diem_chuan = os.environ.get('DIEM_CHUAN_DATA_PATH', 'src/utils/diem_chuan.csv')
        self.df_path_to_hop = os.environ.get('TO_HOP_DATA_PATH', 'src/utils/to_hop_xet_tuyen.csv')
        self.df_1 = pd.read_csv(self.df_path_diem_chuan)
        self.df_2 = pd.read_csv(self.df_path_to_hop)

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

    def _retrieve(self, query: str) -> str:
        """
        Retrieve up to 100 candidates from Qdrant and rerank top 20 using Cohere 3.5.
        """
        # Encode query and fetch top 100
        q_vec = self.embedding.encode(query)
        initial_hits = self.qdrant_client.search(
            collection_name=self.collections['dienbien'],
            query_vector=q_vec,
            limit=50
        )
        
        # Format documents for reranking
        formatted = []
        for hit in initial_hits:
            payload = hit.payload
            file_name = payload.get('file_name', '')
            content = payload.get('content', '')
            link_file = payload.get('link_file', '')
            formatted.append(f"File name: {file_name} | Content: {content} | Link file: {link_file}")
        
        # Rerank with Cohere 3.5
        rerank_response = self.cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=formatted,
            top_n=10
        )
        
        # Extract ranked results
        selected_hits = [initial_hits[result.index] for result in rerank_response.results]
        
        # Return formatted results
        return '\n\n'.join([f"Title: {hit.payload.get('title', '')} | Content: {hit.payload.get('content', '')} | Link file: {hit.payload.get('link_file', '')}" 
                        for hit in selected_hits])

    

    def _init_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        # Qdrant retrieval tools
        tools.extend([
            Tool(
                name='Retrieval',
                func=self._retrieve,
                description="Tìm ra các tài liệu liên quan nhất rồi reranking lại để chọn ra 10 tài liệu thật sự liên quan nhất"
            ),
        ])
        return tools
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """Khởi tạo ChatPromptTemplate cho agent"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CHATBOT_SYSTEM_PROMPT.format(current_time=self.current_time, current_year=self.current_year)),
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
        chat_history = conversation if conversation else []
        
        abbr = get_abbr_and_expansion(user_input)
        if abbr:
            user_input = user_input + "\nDanh sách từ viết tắt và các mở rộng:\n" + abbr

        self.tool_called = False
        self.stream_allowed = True
        self.final_answer_mode = False
        self.tool_count = 0  # Biến đếm số tool đang chạy

        async for event in self.agent_executor.astream_events(
            {
                'input': user_input,
                'chat_history': chat_history
            },
            version="v1"
        ):
            event_type = event.get("event", "")
            
            # Đánh dấu khi bắt đầu gọi tool: tạm khóa stream và tăng bộ đếm
            if event_type == "on_tool_start":
                self.tool_called = True
                self.stream_allowed = False
                self.tool_count += 1
            
            # Đánh dấu khi tool kết thúc: giảm bộ đếm, chỉ mở stream khi tất cả tool hoàn thành
            elif event_type == "on_tool_end":
                if self.tool_called:
                    self.tool_count -= 1
                    if self.tool_count == 0:
                        self.stream_allowed = True
                    continue  # Bỏ qua nội dung từ on_tool_end
            
            # Chỉ stream nội dung từ LLM khi stream_allowed và không còn tool nào chạy
            elif event_type == "on_chat_model_stream" and self.stream_allowed and self.tool_count == 0:
                data = event.get("data", {})
                if "chunk" in data:
                    chunk = data["chunk"]
                    content = getattr(chunk, "content", str(chunk))
                    if content:
                        yield content
    
    def invoke(self, user_input: str):
        return self.agent_executor.invoke({'input': user_input, 'history': []})
