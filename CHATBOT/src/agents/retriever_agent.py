import uuid
import operator
import json
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from src.utils.prompt_templates import grade_relevant_prompt, rewrite_prompt, select_collection_prompt
from src.vector_store.load_qdrant_vector_db import VectorDBCollectionRetriever

from dotenv import load_dotenv
load_dotenv()


# Định nghĩa kiểu state cho workflow
class MultiCollectionRetrieveState(TypedDict):
    query: str
    selected_collection: str
    retrieved_documents: List[str]
    relevant_documents: List[str]
    sufficient_information: Annotated[List[bool], operator.add]
    current_round: int
    max_round: int
    display_fields: List[str]
    top_k: Annotated[List[int], operator.add]
    threshold: Annotated[List[float], operator.add]


class GradeRelevantOutput(BaseModel):
    relevant_indices: List[int] = Field(
        description="Array containing indices of all documents relevant to the question"
    )
    sufficient_information: bool = Field(
        description="Boolean indicating if the relevant documents collectively provide enough information to answer the question"
    )


class RewritePromptOutput(BaseModel):
    query: str = Field(
        description="The rewritten query in Vietnamese, optimized for better search performance."
    )
    top_k: int = Field(
        description="The new top_k value, increased by 10-15 compared to the previous value to broaden search results."
    )
    threshold: float = Field(
        description="The new threshold value, decreased by 0.1–0.2 compared to the previous value to include more potential results."
    )


class MultiCollectionRetrieveAgent:
    def __init__(self, 
                 collections: List[VectorDBCollectionRetriever],
                 model_name="gpt-4o-mini",
                 temperature=0,
                 max_round: int = 3):
        """
        Khởi tạo MultiCollectionRetrieveAgent với các tham số:
          - collections: danh sách các VectorDBCollectionRetriever để lựa chọn
          - model_name: tên mô hình LLM sử dụng
          - max_round: số vòng tối đa cho quá trình rewrite
        """
        self.collections = {c.collection_name: c for c in collections}
        self.collection_descriptions = {c.collection_name: c.description for c in collections}
        self.collection_display_fields = {c.collection_name: c.display_fields for c in collections}
        self.llm = self.choose_llm_model(model_name, temperature)
        self.max_round = max_round
        self.memory = MemorySaver()

        # Xây dựng workflow sử dụng LangGraph
        self._build_workflow()

    def choose_llm_model(self, model_name, temperature):
        if "gpt" in model_name:
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif "gemini" in model_name:
            llm = ChatGoogleGenerativeAI(model="models/" + model_name, temperature=temperature)
        
        return llm
    
    # ---- Các hàm xử lý trong workflow ----
    async def select_collection(self, state: MultiCollectionRetrieveState):
        """
        Chọn collection phù hợp dựa trên câu truy vấn.
        """
        prompt = select_collection_prompt.format(
            query=state["query"],
            collections=json.dumps(self.collection_descriptions, indent=2, ensure_ascii=False))

        response = await self.llm.ainvoke(prompt)
        selected_collection = response.content.strip()
        display_fields = self.collection_display_fields.get(selected_collection, [])
        
        return {
            "selected_collection": selected_collection,
            "display_fields": display_fields
        }

    async def retrieve_query(self, state: MultiCollectionRetrieveState):
        """
        Gọi retriever tương ứng với collection được chọn để lấy các documents.
        """
        retriever = self.collections[state["selected_collection"]]
        retrieved_results = await retriever.search_with_threshold(
            query_text=state["query"], 
            limit=state["top_k"][-1], 
            score_threshold=state["threshold"][-1]
        )
        retrieved_documents = []
        for x in retrieved_results:
            document = "\n".join([f"{x['payload'][field]}" for field in state["display_fields"]])
            retrieved_documents.append(document)
        
        return {"retrieved_documents": retrieved_documents}

    async def grade_relevant(self, state: MultiCollectionRetrieveState):
        """
        Đánh giá tính liên quan của các documents đã truy xuất.
        """
        format_docs = {i: doc for i, doc in enumerate(state["retrieved_documents"])}
        
        prompt = grade_relevant_prompt.format(query=state["query"], format_docs=format_docs)
        
        response = await self.llm.with_structured_output(GradeRelevantOutput).ainvoke(prompt)
        relevant_indices = response.relevant_indices
        
        # Lấy các document liên quan dựa trên indices
        relevant_documents = [state["retrieved_documents"][i] for i in relevant_indices if i < len(state["retrieved_documents"])]
        
        return {
            "relevant_documents": relevant_documents, 
            "sufficient_information": [response.sufficient_information]
        }

    async def rewrite_query(self, state: MultiCollectionRetrieveState):
        """
        Viết lại câu truy vấn dựa trên thông tin từ các document liên quan.
        """
        prompt = rewrite_prompt.format(
            query=state["query"], relevant_documents=state["relevant_documents"],
            previous_top_k=state["top_k"][-1], previous_threshold=state["threshold"][-1])
        response = await self.llm.with_structured_output(RewritePromptOutput).ainvoke(prompt)
        
        # Tăng số vòng hiện tại sau mỗi lần rewrite
        return {"query": response.query, "current_round": state["current_round"] + 1,
                "top_k": [response.top_k], "threshold": [response.threshold]}

    def decide_to_rewrite(self, state: MultiCollectionRetrieveState):
        """
        Quyết định có tiếp tục rewrite hay không.
        """
        if not state["sufficient_information"][-1] and state["current_round"] < state["max_round"]:
            return "rewrite_query"
        else:
            return "end"

    def _build_workflow(self):
        """
        Xây dựng workflow của retrieve agent sử dụng StateGraph.
        """
        self.workflow = StateGraph(MultiCollectionRetrieveState)
        
        # Thêm các node
        self.workflow.add_node("select_collection", self.select_collection)
        self.workflow.add_node("retrieve", self.retrieve_query)
        self.workflow.add_node("grade_relevant", self.grade_relevant)
        self.workflow.add_node("rewrite_query", self.rewrite_query)

        # Thêm các edge
        self.workflow.add_edge(START, "select_collection")
        self.workflow.add_edge("select_collection", "retrieve")
        self.workflow.add_edge("retrieve", "grade_relevant")
        self.workflow.add_conditional_edges(
            "grade_relevant", 
            self.decide_to_rewrite, 
            {"rewrite_query": "rewrite_query", "end": END}
        )
        self.workflow.add_edge("rewrite_query", "retrieve")
        
        # Compile workflow
        self.app = self.workflow.compile(checkpointer=self.memory)

    def display_graph(self):
        """
        Hiển thị đồ thị workflow.
        """
        display(Image(self.app.get_graph(xray=True).draw_mermaid_png()))

    async def run(self, query: str):
        """
        Khởi tạo state với câu truy vấn và chạy retrieve agent.
        """
        initial_state: MultiCollectionRetrieveState = {
            "query": query,
            "selected_collection": "",
            "retrieved_documents": [],
            "relevant_documents": [],
            "sufficient_information": [],
            "current_round": 0,
            "max_round": self.max_round,
            "display_fields": [],
            "top_k": [10],
            "threshold": [0.5]
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        return await self.app.ainvoke(initial_state, config)
