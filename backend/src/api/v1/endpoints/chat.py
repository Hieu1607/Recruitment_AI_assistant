from fastapi import APIRouter, HTTPException
import uuid
from src.schemas.ai_schema import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Mock AI response
    bot_reply = f"Mock reply to: {request.message}"
    
    return ChatResponse(
        response=bot_reply,
        session_id=session_id
    )
