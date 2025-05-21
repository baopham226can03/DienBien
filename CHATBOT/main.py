import sys
import os
import asyncio
from uuid import uuid4
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from typing import Optional
from fastapi import FastAPI, Response, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv


from src.agents.cache_bot import CacheBot
from src.agents.chatbot_agent_v3 import ChatBotAgent

from src.config.settings import LLM_MODEL_NAME, TEMPERATURE, AGENT_TYPE,\
    QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL_NAME, LOP_1_SCORE_THRESHOLD
from typing import List, Dict


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QuestionRequest(BaseModel):
    question: str
    userid: str
    conversation: List[Dict[str, str]]


# Initialize collections and agents
@app.on_event("startup")
async def startup_event():
    global chatbot_agent

    try:
        logger.info("Initializing Cache Bot...")



        logger.info("Initializing Chatbot Agents...")

        chatbot_agent = ChatBotAgent(
            agent_type = AGENT_TYPE,
            model_name = LLM_MODEL_NAME,
            qdrant_url = QDRANT_URL,
            qdrant_api_key = QDRANT_API_KEY,
            embedding_model = EMBEDDING_MODEL_NAME,
            collections = {
                'cam_nang': 'cam_nang_tuyen_sinh_HUIT_2025_2',
                'so_tay': 'so_tay_sinh_vien_HUIT_2025_2',
                'dienbien': 'dienbien'},
            temperature=TEMPERATURE)

    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise

# Store session histories
session_histories = {}

@app.get("/")
def root():
    return {"response": "Hello World"}

@app.post("/process")
async def process(
    query: QuestionRequest,
    response: Response,
    session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    logger.info(f"Received question: {query.question}")
    logger.info(f"Parameters request: {query}")
    logger.info(f"Session ID request: {session_id}")
    question = query.question.strip()
    if not question:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Question cannot be empty"}

    try:
        # Generate a session ID if not provided
        if session_id is None:
            session_id = str(uuid4())
        logger.info(f"Session ID After request: {session_id}")
        history = session_histories.get(session_id, "")



        async def rag_stream():
            try:
                full_answer = ""
                result = chatbot_agent.run(question, query.conversation)
                async for chunk in result:  # Dùng sync for
                    full_answer += chunk
                    yield chunk
                session_histories[session_id] = history + f"\nUser: {question}\nAssistant: {full_answer}\n"
                timestamp = datetime.utcnow().isoformat() + "Z"
                escaped_answer = full_answer.replace("\n", "\\n")
                logger.info(
                    f"[{timestamp}] [END] "
                    f"sessionId={session_id} "
                    f"userid={query.userid} "
                    f"question=\"{question}\" "
                    f"fullAnswer=\"{escaped_answer}\""
                )
            except Exception as e:
                yield f"\n[Lỗi]: {str(e)}"
            # try:
            #     full_answer = ""
            #     async for chunk in chatbot_agent.run(question, query.conversation):  # Dùng sync for
            #         full_answer += chunk
            #         print("chunk", chunk)
            #         yield chunk
            #     session_histories[session_id] = history + f"\nUser: {question}\nAssistant: {full_answer}\n"
            # except Exception as e:
            #     yield f"\n[Lỗi]: {str(e)}"

        return StreamingResponse(
            content=rag_stream(),
            media_type="text/plain",
            headers={"X-Session-ID": session_id}
        )


    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": f"Failed to process question: {str(e)}"}
