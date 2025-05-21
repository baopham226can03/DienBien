import os
import cohere
import logging
import asyncio
from typing import List, Dict, Literal, AsyncGenerator
from datetime import datetime
from uuid import uuid4
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.utils.process_abbreviation import get_abbr_and_expansion

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn chuyên nghiệp của dịch vụ công tỉnh Điện Biên. 

**Lưu ý về thời gian**: Năm hiện tại là {current_year} (trích xuất từ {current_time}). Khi xử lý các yêu cầu liên quan đến "năm gần đây", hãy hiểu rằng "năm gần đây" là năm hiện tại và các năm trước đó. Ví dụ, nếu yêu cầu "3 năm gần đây" và năm hiện tại là 2025, thì đó là 2024, 2023, và 2022.**

## Công cụ truy vấn
- **dien_bien_retrieve**: Tìm kiếm 10 tài liệu liên quan nhất từ cơ sở dữ liệu tỉnh Điện Biên bằng cách truy xuất và xếp hạng lại (reranking). Trả về danh sách các tài liệu liên quan hoặc tên tài liệu (file_name) để sử dụng trong _extract.
- **_extract**: Trích xuất thông tin chi tiết (file_name, content, link_file) của một tài liệu cụ thể dựa trên tên tài liệu (file_name) được cung cấp từ dien_bien_retrieve.

## Quy trình xử lý
1. **Tiền xử lý**: Sửa lỗi chính tả, chuẩn hóa câu hỏi mà không làm mất ý nghĩa. Giữ lại các từ khóa quan trọng.
2. **Phân tích yêu cầu**: Xác định yêu cầu chính xác dựa trên câu hỏi và lịch sử hội thoại.
3. Khi người dùng chào hỏi: Nhận mình là trợ lý tư vấn dịch vụ công và sẵn sàng hỗ trợ.
4. Không chào hỏi lại nếu đã có lịch sử hội thoại.
5. **Xử lý câu hỏi về thủ tục**:
   - Gọi **dien_bien_retrieve** để tìm tài liệu hoặc gợi ý thủ tục tương đồng.
   - Nếu **dien_bien_retrieve** trả về tên tài liệu (file_name), tự động gọi **_extract** để lấy thông tin chi tiết.
   - Nếu **dien_bien_retrieve** trả về danh sách tài liệu, hiển thị các tài liệu đó.
6. **Kiểm tra kết quả**: Nếu không tìm thấy tài liệu hoặc thủ tục liên quan, thông báo xin lỗi và đề nghị người dùng cung cấp thêm thông tin.

## Nguyên tắc phản hồi
1. Trả lời chính xác, đầy đủ, dựa trên thông tin từ công cụ.
2. Ngôn ngữ trang trọng, gần gũi, nhiệt tình.
3. Mặc định câu hỏi chung về thủ tục là liên quan đến tỉnh Điện Biên.
4. Không bịa đặt thông tin. Nếu không có dữ liệu, xin lỗi và thành thật.
5. Nếu được chào thì lịch sự chào lại như một chuyên viên tư vấn.
"""

class ChatBotAgent:
    """
    Chatbot tư vấn dịch vụ công tỉnh Điện Biên với khả năng:
    - Tìm kiếm tài liệu và gợi ý thủ tục qua Qdrant và Cohere
    - Trích xuất thông tin chi tiết của tài liệu
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
        logger.info("Khởi tạo ChatBotAgent...")
        try:
            # Kiểm tra biến môi trường
            if not os.environ.get('QDRANT_URL') or not os.environ.get('QDRANT_API_KEY'):
                raise ValueError("QDRANT_URL và QDRANT_API_KEY phải được thiết lập.")
            if not os.environ.get('COHERE_API_KEY'):
                raise ValueError("COHERE_API_KEY phải được thiết lập.")

            self.agent_type = agent_type
            # Qdrant client & embedding
            logger.info("Kết nối đến Qdrant và khởi tạo embedding...")
            self.qdrant_client = QdrantClient(
                url=qdrant_url or os.environ.get('QDRANT_URL'),
                api_key=qdrant_api_key or os.environ.get('QDRANT_API_KEY')
            )
            self.cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))
            self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True)
            self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_year = self.current_time.split('-')[0]

            # Cache for query results
            self.query_cache = {}  # Dictionary to store query results

            # LLM
            logger.info(f"Khởi tạo LLM: {model_name} với temperature={temperature}")
            self.llm = self._choose_llm(model_name, temperature)

            # Collection names
            self.collections = collections or {'dien_bien': 'dienbien'}
            logger.info(f"Sử dụng collections: {self.collections}")

            # Tools
            self.tools = self._init_tools()
            logger.info("Đã khởi tạo công cụ dien_bien_retrieve và _extract")

            self.prompt = self._build_prompt()
            logger.info("Đã xây dựng prompt cho agent")

            # Build agent executor
            self.agent_executor = self._build_agent_executor()
            logger.info("Đã khởi tạo agent executor")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo ChatBotAgent: {str(e)}")
            raise

    def _choose_llm(self, model_name: str, temperature: float):
        logger.info(f"Chọn LLM: {model_name}")
        if 'gpt' in model_name:
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif 'gemini' in model_name:
            return ChatGoogleGenerativeAI(model='models/' + model_name, temperature=temperature)
        raise ValueError(f'Unsupported model: {model_name}')

    @lru_cache(maxsize=1000)
    def _encode_query(self, query: str) -> tuple:
        """Cache encoded query vectors."""
        logger.info(f"Mã hóa query: {query}")
        return tuple(self.embedding.encode(query))

    async def _dien_bien_retrieve(self, query: str) -> str:
        """
        Retrieve top 10 documents from the dien_bien collection and rerank using Cohere.
        Returns formatted results or a file_name for _extract.
        """
        logger.info(f"Thực hiện dien_bien_retrieve với query: {query}")
        try:
            # Check cache
            cache_key = f"dien_bien_{query}"
            if cache_key in self.query_cache:
                logger.info(f"Trả về kết quả từ cache cho query: {query}")
                return self.query_cache[cache_key]

            collection_name = self.collections['dien_bien']
            logger.info(f"Tìm kiếm trong collection: {collection_name}")

            # Encode query
            q_vec = self._encode_query(query)
            logger.info("Đã mã hóa query thành vector")

            # Search top 50 candidates
            initial_hits = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=q_vec,
                limit=50
            )
            logger.info(f"Tìm thấy {len(initial_hits)} tài liệu ban đầu")

            if not initial_hits:
                logger.warning(f"Không tìm thấy tài liệu trong {collection_name}")
                return "Không tìm thấy tài liệu liên quan."

            # Format documents for reranking
            formatted = []
            file_names = []
            for hit in initial_hits:
                payload = hit.payload
                title = payload.get('title', '')
                content = payload.get('content', '')
                file_name = payload.get('file_name', '')
                formatted.append(f"Title: {title} | Content: {content}")
                file_names.append(file_name)
            logger.info("Đã định dạng tài liệu cho reranking")

            # Rerank with Cohere
            rerank_response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=formatted,
                top_n=10
            )
            logger.info(f"Đã rerank, trả về {len(rerank_response.results)} tài liệu")

            # Extract top 10 reranked documents
            selected_hits = [initial_hits[result.index] for result in rerank_response.results]
            file_names = [file_names[result.index] for result in rerank_response.results]

            # Format results
            result = '\n\n'.join([f"Title: {hit.payload.get('title', '')} | Content: {hit.payload.get('content', '')} | File Name: {hit.payload.get('file_name', '')}" 
                                 for hit in selected_hits])

            # If only one highly relevant document, return its file_name
            if len(rerank_response.results) == 1 and rerank_response.results[0].relevance_score > 0.9:
                logger.info(f"Tìm thấy tài liệu có độ liên quan cao: {file_names[0]}")
                result = file_names[0]

            # Cache the result
            self.query_cache[cache_key] = result
            logger.info(f"Đã lưu kết quả vào cache cho query: {query}")

            return result if result else "Không tìm thấy tài liệu phù hợp sau khi phân tích."

        except Exception as e:
            logger.error(f"Lỗi trong dien_bien_retrieve: {str(e)}")
            return f"Lỗi khi truy xuất tài liệu: {str(e)}"

    def _dien_bien_retrieve_sync(self, query: str) -> str:
        """
        Synchronous wrapper for _dien_bien_retrieve to use in Tool.
        """
        return asyncio.run(self._dien_bien_retrieve(query))

    def _extract(self, document_name: str) -> str:
        """
        Extract detailed information (file_name, content, link_file) of a document from Qdrant.
        """
        logger.info(f"Thực hiện _extract với document_name: {document_name}")
        if not document_name or not isinstance(document_name, str):
            logger.warning("Tên tài liệu không hợp lệ")
            return "Vui lòng cung cấp tên tài liệu hợp lệ."
        
        try:
            search_result = self.qdrant_client.scroll(
                collection_name=self.collections['dien_bien'],
                scroll_filter={
                    "must": [
                        {
                            "key": "file_name",
                            "match": {"value": document_name}
                        }
                    ]
                },
                limit=1
            )
            hits = search_result[0]
            
            if not hits:
                logger.warning(f"Không tìm thấy tài liệu với tên: {document_name}")
                return f"Không tìm thấy tài liệu nào với tên: {document_name}"
            
            hit = hits[0]
            doc_info = hit.payload
            # Extract specific fields
            file_name = doc_info.get('file_name', 'Không có thông tin')
            content = doc_info.get('content', 'Không có nội dung')
            link_file = doc_info.get('link_file', 'Không có liên kết')
            formatted_info = f"Thông tin tài liệu:\n- Tên tài liệu: {file_name}\n- Nội dung: {content}\n- Liên kết: {link_file}"
            logger.info(f"Trích xuất thành công tài liệu: {file_name}")
            return formatted_info
        except Exception as e:
            logger.error(f"Lỗi khi truy xuất tài liệu '{document_name}': {str(e)}")
            return f"Lỗi khi truy xuất tài liệu '{document_name}': {str(e)}"

    def _init_tools(self) -> List[Tool]:
        logger.info("Khởi tạo công cụ dien_bien_retrieve và _extract")
        tools: List[Tool] = [
            Tool(
                name='dien_bien_retrieve',
                func=self._dien_bien_retrieve_sync,
                description="Tìm kiếm 10 tài liệu liên quan nhất từ collection dien_bien, xếp hạng lại bằng Cohere, và trả về danh sách tài liệu hoặc file_name để dùng trong _extract."
            ),
            Tool(
                name='_extract',
                func=self._extract,
                description="Trích xuất thông tin chi tiết (file_name, content, link_file) của một tài liệu cụ thể dựa trên file_name."
            )
        ]
        return tools
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """Khởi tạo ChatPromptTemplate cho agent"""
        logger.info("Xây dựng ChatPromptTemplate")
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
        logger.info("Xây dựng agent executor")
        try:
            agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            logger.info("Agent executor được xây dựng thành công")
            return executor
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng agent executor: {str(e)}")
            raise

    async def run(self, user_input: str, conversation: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        logger.info(f"Chạy agent với user_input: {user_input}")
        chat_history = conversation if conversation else []
        
        try:
            abbr = get_abbr_and_expansion(user_input)
            if abbr:
                logger.info(f"Đã xử lý từ viết tắt: {abbr}")
                user_input = user_input + "\nDanh sách từ viết tắt và các mở rộng:\n" + abbr

            self.tool_called = False
            self.stream_allowed = True
            self.tool_count = 0

            async for event in self.agent_executor.astream_events(
                {
                    'input': user_input,
                    'chat_history': chat_history
                },
                version="v1"
            ):
                event_type = event.get("event", "")
                logger.debug(f"Sự kiện: {event_type}")

                if event_type == "on_tool_start":
                    self.tool_called = True
                    self.stream_allowed = False
                    self.tool_count += 1
                    logger.info(f"Bắt đầu gọi công cụ, tool_count: {self.tool_count}")

                elif event_type == "on_tool_end":
                    if self.tool_called:
                        self.tool_count -= 1
                        if self.tool_count == 0:
                            self.stream_allowed = True
                        logger.info(f"Kết thúc công cụ, tool_count: {self.tool_count}")
                        continue
                
                elif event_type == "on_chat_model_stream" and self.stream_allowed and self.tool_count == 0:
                    data = event.get("data", {})
                    if "chunk" in data:
                        chunk = data["chunk"]
                        content = getattr(chunk, "content", str(chunk))
                        if content:
                            logger.debug(f"Stream nội dung: {content}")
                            yield content
                
                elif event_type == "on_error":
                    logger.error(f"Lỗi trong agent executor: {event.get('data', 'Unknown error')}")
                    yield f"Lỗi: {event.get('data', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Lỗi khi chạy agent: {str(e)}")
            yield f"Lỗi: {str(e)}"

    def invoke(self, user_input: str, conversation: List[Dict[str, str]] = None):
        logger.info(f"Gọi invoke với user_input: {user_input}")
        chat_history = conversation if conversation else []
        try:
            result = self.agent_executor.invoke({'input': user_input, 'chat_history': chat_history})
            logger.info(f"Kết quả invoke: {result}")
            return result
        except Exception as e:
            logger.error(f"Lỗi trong invoke: {str(e)}")
            raise