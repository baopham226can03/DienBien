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

from dotenv import load_dotenv
load_dotenv()


# CHATBOT_SYSTEM_PROMPT = """
# Bạn là trợ lý tư vấn tuyển sinh chuyên nghiệp của trường Đại học Công Thương TPHCM (HUIT).

# ## Công cụ truy vấn
# - **CamNangRetrieval**: Xét tuyển, hồ sơ, học phí, ngành học, học bổng, việc làm
# - **SoTayRetrieval**: Thông tin trường, khoa, đào tạo, học bổng, hoạt động SV, văn bản
# - **DiemChuanTool**: Điểm chuẩn theo năm, phương thức và ngành
# - **tavily_search**: Công cụ tìm kiếm trên web; sử dụng khi CamNangRetrieval & SoTayRetrieval không đủ thông tin hoặc không tìm thấy kết quả để trả lời câu hỏi

# ## Quy trình xử lý
# 1. Tiền xử lý: Sửa lỗi chính tả, chuyển câu hỏi về dạng chuẩn dễ dàng đọc và xử lý mà không bóp méo hay thay đổi ý nghĩa.
# 2. Phân tích yêu cầu từ câu hỏi và lịch sử hội thoại
# 3. Khi người dùng chào hỏi, hỏi thăm: nhận mình là trợ lý tư vấn tuyển sinh và sẵn sàng hỗ trợ
# 4. Không chào hỏi lại khi đã có lịch sử hội thoại
# 5. Luôn truy xuất thông tin trước khi đưa ra câu trả lời. Khi
# 6. Với câu hỏi phức hợp: truy vấn nhiều nguồn
# 7. Với câu hỏi đề cập đến thời gian cụ thể: tham chiếu thời gian hiện tại {current_time}
# 8. Từ chối lịch sự nếu câu hỏi ngoài phạm vi tuyển sinh HUIT

# IMPORTANT: Khi gọi CamNangRetrieval hoặc SoTayRetrieval, nếu kết quả rỗng hoặc không đủ để trả lời câu hỏi, gọi tavily_search

# ## Nguyên tắc phản hồi
# 1. Trả lời chính xác, đầy đủ thông tin
# 2. Ngôn ngữ trang trọng nhưng gần gũi, thể hiện nhiệt tình
# 3. Mặc định câu hỏi chung về "trường", "điểm chuẩn", "ngành học" là hỏi về HUIT
# 4. Chỉ trả lời dựa trên thông tin truy xuất. Tuyệt đối không bịa đặt, chế thông tin một cách tự tiện nhé.
# 5. Khi thông tin truy xuất không nêu rõ về số lượng, thì không nên khẳng định.
# 6. Với dữ liệu điểm chuẩn không hiển thị dạng bảng, và phải đề cập đến loại xét
# 7. Không so sánh các trường với nhau
# """

CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn tuyển sinh chuyên nghiệp của trường Đại học Công Thương TPHCM (HUIT). Cơ bản thì trường này có 37 ngành đào tạo đại học. Chương trình kéo dài 3,5 năm cho cử nhân và 4 năm cho kỹ sư.

Thời gian hiện tại: {current_time}

**Lưu ý về thời gian**: Năm hiện tại là {current_year} (trích xuất từ {current_time}). Khi xử lý các yêu cầu liên quan đến "năm gần đây", hãy hiểu rằng "năm gần đây" là năm hiện tại và các năm trước đó. Ví dụ, nếu yêu cầu "3 năm gần đây" và năm hiện tại là 2025, thì đó là 2024, 2023, và 2022.**

## Công cụ truy vấn
- **CamNangRetrieval**: Xét tuyển, hồ sơ, học phí, ngành học (số lượng, mô tả), học bổng, việc làm
- **SoTayRetrieval**: Thông tin trường, khoa, đào tạo, học bổng, hoạt động SV, văn bản
- **DiemChuanTool**: Điểm chuẩn theo năm, phương thức và ngành.
- **tavily_search**: Công cụ tìm kiếm trên web; BẮT BUỘC sử dụng khi CamNangRetrieval, SoTayRetrieval và DiemChuanTool trả về kết quả rỗng, không liên quan, hoặc không đủ để trả lời câu hỏi. "Không liên quan" nghĩa là nội dung truy xuất không giải quyết được yêu cầu cụ thể của câu hỏi (ví dụ: hỏi về hiệu trưởng nhưng chỉ trả về thông tin ngành học). Ưu tiên các nguồn chính thức như https://ts.huit.edu.vn/ khi sử dụng tavily_search.

## Quy trình xử lý
1. Tiền xử lý: Sửa lỗi chính tả, chuyển câu hỏi về dạng chuẩn dễ dàng đọc và xử lý mà không bóp méo hay thay đổi ý nghĩa.
2. Phân tích yêu cầu từ câu hỏi và lịch sử hội thoại. Yêu cầu phải **THẬT SỰ CHUẨN XÁC** với nội dung của câu hỏi (Ví dụ hỏi "hiệu trưởng là ai" thì ý định phải là "thông tin về hiệu trưởng" chứ không chỉ là "thông tin trường").
3. Khi người dùng chào hỏi, hỏi thăm: nhận mình là trợ lý tư vấn tuyển sinh và sẵn sàng hỗ trợ.
4. Không chào hỏi lại khi đã có lịch sử hội thoại.
5. Luôn truy xuất thông tin trước khi đưa ra câu trả lời. Phải truy vấn sao cho THẬT CHUẨN với đầy đủ nội dung của câu hỏi, không làm mất mát thông tin.
6. Với câu hỏi phức hợp: truy vấn nhiều nguồn.
7. Từ chối lịch sự nếu câu hỏi ngoài phạm vi tuyển sinh, không liên quan đến HUIT, ...
8. Kiểm tra kết quả từ CamNangRetrieval, SoTayRetrieval, và DiemChuanTool:
   - Nếu kết quả rỗng, chứa "Không tìm thấy thông tin", hoặc không liên quan (ví dụ: nội dung không đề cập đến yêu cầu cụ thể như hiệu trưởng, tổ chức trường), BẮT BUỘC gọi tavily_search.
   - Nếu nội dung trả về không giải quyết được câu hỏi (ví dụ: hỏi về hiệu trưởng nhưng chỉ trả về thông tin ngành học), coi là không liên quan và gọi tavily_search.
   - Nếu tavily_search cũng không cung cấp thông tin, xin lỗi và thông báo không có dữ liệu.

IMPORTANT: 
- tavily_search là lựa chọn cuối cùng và BẮT BUỘC gọi khi các công cụ khác không đáp ứng. Không được dừng lại với thông báo "không có thông tin" nếu chưa thử tavily_search. Khi sử dụng tavily_search, ưu tiên các nguồn chính thức của HUIT như https://ts.huit.edu.vn/ hoặc các nguồn tin cậy như báo chí, trang web chính phủ.
- Luôn truy xuất trước khi đưa ra câu trả lời.

## Nguyên tắc phản hồi chi tiết đầy đủ
1. Trả lời chính xác, đầy đủ thông tin, trả lời thật sự đầy đủ bằng những gì có được.
2. Ngôn ngữ trang trọng nhưng gần gũi, thể hiện nhiệt tình.
3. Mặc định câu hỏi chung về "trường", "điểm chuẩn", "ngành học",... là hỏi về HUIT.
4. Tuyệt đối chỉ trả lời dựa trên thông tin truy xuất. Tuyệt đối không bịa đặt, không được tự trả lời.
5. Khi thông tin truy xuất không nêu rõ về số lượng, thì không nên khẳng định.
6. Với dữ liệu điểm chuẩn không hiển thị dạng bảng, và phải đề cập đến loại xét.
7. Không so sánh các trường với nhau.
8. Nếu không có thông tin sau khi đã thử tất cả công cụ (bao gồm tavily_search), xin lỗi và thành thật.
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
        self.cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))
        self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_year = self.current_time.split('-')[0]

        # LLM
        self.llm = self._choose_llm(model_name, temperature)

        # Pandas DataFrame Agent cho điểm chuẩn
        self.df_path_diem_chuan = os.environ.get('DIEM_CHUAN_DATA_PATH', 'diem_chuan.csv')
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

    def _cam_nang_retrieve(self, query: str) -> str:
        """
        Retrieve up to 100 candidates from Qdrant and rerank top 20 using Cohere 3.5.
        """
        # Encode query and fetch top 100
        q_vec = self.embedding.encode(query)
        initial_hits = self.qdrant_client.search(
            collection_name=self.collections['cam_nang'],
            query_vector=q_vec,
            limit=100
        )
        
        # Format documents for reranking
        formatted = []
        for hit in initial_hits:
            payload = hit.payload
            title = payload.get('title', '')
            content = payload.get('content', '')
            formatted.append(f"Title: {title} | Content: {content}")
        
        # Rerank with Cohere 3.5
        rerank_response = self.cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=formatted,
            top_n=20
        )
        
        # Extract ranked results
        selected_hits = [initial_hits[result.index] for result in rerank_response.results]
        
        # Return formatted results
        return '\n\n'.join([f"Title: {hit.payload.get('title', '')} | Content: {hit.payload.get('content', '')}" 
                        for hit in selected_hits])

    # def _so_tay_retrieve(self, query: str) -> str:
    #     """
    #     Retrieve top 10 SoTay documents from Qdrant without additional reranking.
    #     """
    #     q_vec = self.embedding.encode(query)
    #     hits = self.qdrant_client.search(
    #         collection_name=self.collections['so_tay'],
    #         query_vector=q_vec,
    #         limit=10
    #     )
    #     formatted = []
    #     for hit in hits:
    #         payload = hit.payload
    #         tag = payload.get('tag', '')
    #         title = payload.get('title', '')
    #         content = payload.get('content', '')
    #         formatted.append(f"Tag: {tag} | Title: {title} | Content: {content}")
    #     return '\n\n'.join(formatted)

    def _so_tay_retrieve(self, query: str) -> str:
        """
        Retrieve up to 100 candidates from Qdrant and rerank top 20 using Cohere 3.5.
        """
        # Encode query and fetch top 100
        q_vec = self.embedding.encode(query)
        initial_hits = self.qdrant_client.search(
            collection_name=self.collections['so_tay'],
            query_vector=q_vec,
            limit=50
        )
        
        # Format documents for reranking
        formatted = []
        for hit in initial_hits:
            payload = hit.payload
            tag = payload.get('tag', '')
            title = payload.get('title', '')
            content = payload.get('content', '')
            formatted.append(f"Tag: {tag} | Title: {title} | Content: {content}")
        
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
        return '\n\n'.join([f"Tag: {hit.payload.get('tag', '')} | Title: {hit.payload.get('title', '')} | Content: {hit.payload.get('content', '')}"
                        for hit in selected_hits])

    def _init_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        # Qdrant retrieval tools
        tools.extend([
            Tool(
                name='CamNangRetrieval',
                func=self._cam_nang_retrieve,
                description="Truy xuất thông tin về mô tả ngành học, số lượng ngành học, phương thức xét tuyển, hồ sơ xét tuyển, tổ hợp xét tuyển, mức học phí từng ngành, học bổng, hoạt động ngoại khóa, cơ hội thực tập và việc làm sau tốt nghiệp. Ưu tiên gọi trước tavily_search."
            ),
            Tool(
                name='SoTayRetrieval',
                func=self._so_tay_retrieve,
                description="Truy xuất thông tin về tổng quan trường, các khoa, hệ thống đào tạo, nghiên cứu khoa học, chính sách học bổng, các chính sách miễn giảm học phí, hoạt động sinh viên và các văn bản quan trọng liên quan đến sinh viên. Ưu tiên gọi trước tavily_search."
            ),
            TavilySearchResults(
                max_results=5,
                description="Công cụ tìm kiếm web, BẮT BUỘC gọi khi CamNangRetrieval, SoTayRetrieval, và DiemChuanTool trả về kết quả rỗng hoặc không liên quan đến câu hỏi. Luôn kiểm tra kết quả từ các công cụ khác trước khi gọi. Ưu tiên trích dẫn nguồn chính thức của Trường ĐH Công Thương TP.HCM, đặc biệt trang tuyển sinh https://ts.huit.edu.vn/."
            )
        ])

        diem_chuan_tool = DiemChuanTool(llm=self.llm, df=self.df)

        tools.append(diem_chuan_tool)
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
