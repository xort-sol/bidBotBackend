from fastapi import APIRouter, Body
from models.schema import MessageEditRequest
from models.schema import MessageSuggestionRequest
from services.response_service import edit_response_suggestion # type: ignore
from services.response_service import generate_response_suggestion

router = APIRouter()

@router.post("/suggestResponse", response_model=str)
async def suggest_response(request: MessageSuggestionRequest = Body(...)):
    response_text = await generate_response_suggestion(request)
    return response_text

@router.post("/editResponse", response_model=str)
async def edit_response(request: MessageEditRequest = Body(...)):
    response_text = await edit_response_suggestion(request)
    return response_text
