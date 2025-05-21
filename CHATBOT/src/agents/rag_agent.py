import uuid
import operator
import json
import datetime

from typing import List, Annotated, Dict, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

from src.utils.prompt_templates import create_queries_prompt, generate_answer_prompt,\
    classify_and_clarify_prompt, non_relation_response_prompt
from src.utils.process_abbreviation import get_abbr_and_expansion

from dotenv import load_dotenv
load_dotenv()

def combine_unique_docs(current: List[str], new: List[str]) -> List[str]:
    return list(dict.fromkeys(current + new))  # Preserves order while removing duplicates


# Định nghĩa state cho RAG workflow
class RAGState(TypedDict):
    history: str
    question: str
    clarified_question: str
    question_type: str
    queries: List[str]
    relevant_documents: Annotated[List[str], combine_unique_docs]
    answer: str


class QuestionClassification(BaseModel):
    question_type: Literal["greeting", "relation", "non-relation"]
    clarified_question: str = Field(
        description="Nội dung đã được làm rõ, giữ chính xác những gì người dùng đã hỏi sau khi thay thế từ viết tắt bằng cụm từ đầy đủ và làm rõ ngữ cảnh. Không thêm nội dung suy luận hay gợi ý."
    )


class CreateQueriesOutput(BaseModel):
   is_compound: bool = Field(description="boolean indicating if this is a compound question")
   queries: List[str] = Field(
       description="Array of decomposed simple questions (2-4) from the original compound question"
   )


class RAGAgent:
    def __init__(self, retrieve_app,
                 model_name="gpt-4o-mini",
                 temperature=0
                 ):
        """
        Khởi tạo RAGAgent với:
          - llm: đối tượng ngôn ngữ (LLM) có phương thức invoke(prompt).
          - retrieve_app: workflow hoặc agent chịu trách nhiệm truy xuất tài liệu,
                          ví dụ: retrieve_app từ RetrieveAgent đã được compile.
        """
        self.llm = self.choose_llm_model(model_name, temperature)
        self.retrieve_app = retrieve_app
        self.memory = MemorySaver()
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Xây dựng workflow RAG thông qua LangGraph
        self._build_workflow()
    
    def choose_llm_model(self, model_name, temperature):
        if "gpt" in model_name:
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif "gemini" in model_name:
            llm = ChatGoogleGenerativeAI(model="models/" + model_name, temperature=temperature)
        
        return llm
    
    def format_documents(self, documents):
        if not documents:
            return "Không có tài liệu liên quan."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_doc = f"Tài liệu {i}:\n{doc}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    async def classify_and_clarify_question(self, state: RAGState):
        abbr_and_expansion = get_abbr_and_expansion(state["question"])
        prompt = classify_and_clarify_prompt.format(
            question=state["question"], history=state["history"],
            current_time=self.current_time, abbr_and_expansion=abbr_and_expansion)
        
        response = await self.llm.with_structured_output(QuestionClassification).ainvoke(prompt)
        
        return {"question_type": response.question_type, "clarified_question": response.clarified_question}
    
    def should_query(self, state: RAGState):
        if state["question_type"] == "relation":
            return "create_queries"
        else:
            return "generate_answer"
    
    async def create_queries(self, state: RAGState):
        """
        Node tạo danh sách query phụ dựa trên câu hỏi.
        Gọi LLM với prompt tạo query và giải mã kết quả JSON trả về.
        """
        prompt = create_queries_prompt.format(question=state["clarified_question"])
        response = await self.llm.with_structured_output(CreateQueriesOutput).ainvoke(prompt)
        queries = response.queries
        
        return {"queries": queries}
    
    def continue_to_retriever(self, state: RAGState):
        """
        Node conditional chuyển sang retriever. Với mỗi query tạo một Send event.
        """
        return [Send("retriever", {"query": query, "current_round": 1, "max_round": 3,
                                   "top_k": [10], "threshold": [0.5]})
                for query in state["queries"]]
    
    def combine_documents(self, state: RAGState):
        """
        Node gom các tài liệu liên quan. Ở đây ta giả sử state['relevant_documents']
        đã được cập nhật qua retrieve_app và gộp các kết quả lại.
        """
        relevant_documents = state["relevant_documents"]
        return {"relevant_documents": relevant_documents}

    async def generate_answer(self, state: RAGState):
        """
        Node sinh câu trả lời cuối cùng dựa trên câu hỏi và các tài liệu đã gộp.
        """
        has_history = "true" if state["history"] else "false"
        if state["question_type"] != "relation":
            prompt = non_relation_response_prompt.format(
                question_type=state["question_type"],
                question=state["clarified_question"], 
                has_history=has_history,
            )
        else:
            relevant_docs_text = self.format_documents(state["relevant_documents"])
            prompt = generate_answer_prompt.format(
                question=state["clarified_question"],
                relevant_documents=relevant_docs_text,
                current_time=self.current_time,
                has_history=has_history,
            )
        response = await self.llm.ainvoke(prompt)
        return {"answer": response.content}

    def _build_workflow(self):
        # Khởi tạo state graph cho RAG workflow
        self.graph = StateGraph(RAGState)

        # --- Đăng ký các node vào graph ---
        self.graph.add_node("classify_and_clarify_question", self.classify_and_clarify_question)
        self.graph.add_node("create_queries", self.create_queries)
        # Sử dụng retrieve_app được truyền vào (workflow từ agent truy xuất)
        self.graph.add_node("retriever", self.retrieve_app)
        self.graph.add_node("combine_documents", self.combine_documents)
        self.graph.add_node("generate_answer", self.generate_answer)

        # --- Xây dựng các cạnh (edge) của graph ---
        self.graph.add_edge(START, "classify_and_clarify_question")
        self.graph.add_conditional_edges("classify_and_clarify_question", self.should_query)
        # Dùng conditional edge từ create_queries đến retriever theo danh sách query
        self.graph.add_conditional_edges("create_queries", self.continue_to_retriever, ["retriever"])
        self.graph.add_edge("retriever", "combine_documents")
        self.graph.add_edge("combine_documents", "generate_answer")
        self.graph.add_edge("generate_answer", END)

        # Compile workflow thành ứng dụng có thể invoke
        self.app = self.graph.compile(checkpointer=self.memory)

    def display_graph(self):
        """
        Hiển thị đồ thị workflow RAG thông qua việc render Mermaid.
        """
        display(Image(self.app.get_graph(xray=True).draw_mermaid_png()))

    async def run(self, question: str, conversation: List[Dict[str, str]]):
        """
        Khởi tạo state với câu hỏi và chạy workflow RAG.
        Trả về state kết quả chứa câu trả lời cuối cùng.
        """
        history = ""
        for item in conversation:
            for role, msg in item.items():
                history += f"<{role}>\n{msg}\n</{role}>\n\n"
        
        initial_state: RAGState = {
            "question": question,
            "queries": [],
            "relevant_documents": [],
            "answer": "",
            "history": history,
            "question_type": "",
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        # return self.app.invoke(initial_state, config)

        async for msg, metadata in self.app.astream(initial_state, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "generate_answer":
                yield msg.content
