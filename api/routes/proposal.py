from fastapi import APIRouter, Body, HTTPException
from models.schema import ProposalRequest, ProposalResponse
from models.schema import EditProposalRequest, EditProposalResponse
from services.groq_service import generate_proposal_with_groq, edit_proposal_with_groq
from config.settings import settings

router = APIRouter(
    prefix="/api",
    tags=["proposal"]
)

@router.post("/generateProposal", response_model=ProposalResponse)
async def generate_proposal(request: ProposalRequest = Body(...)):
    """
    Generate a job proposal based on the provided job description using the Groq API.
    
    Args:
        request: ProposalRequest model containing job description and parameters
        
    Returns:
        ProposalResponse: Generated proposal and status
    """
    try:
        # Get model to use (either from request or default)
        model_used = request.model if request.model else settings.DEFAULT_MODEL
        
        # Call the Groq service to generate the proposal
        proposal_text = await generate_proposal_with_groq(request)
        
        # Return the response
        return ProposalResponse(
            proposal=proposal_text,
            status="success",
            model_used=model_used
        )
    except HTTPException:
        # Re-raise HTTP exceptions (they already have status code and detail)
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to generate proposal: {str(e)}")
    

@router.post("/editProposal", response_model=EditProposalResponse)
async def edit_proposal(request: EditProposalRequest = Body(...)):
    """
    Edit an existing proposal based on the provided instructions using the Groq API.
    
    Args:
        request: EditProposalRequest model containing original proposal and edit instructions
        
    Returns:
        EditProposalResponse: Edited proposal and status information
    """
    try:
        # Get model to use (either from request or default)
        model_used = request.model if request.model else settings.DEFAULT_MODEL
        
        # Call the Groq service to edit the proposal
        edited_response = await edit_proposal_with_groq(request)
        
        # Return the response (already formatted by the edit_proposal_with_groq function)
        return edited_response
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have status code and detail)
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to edit proposal: {str(e)}")