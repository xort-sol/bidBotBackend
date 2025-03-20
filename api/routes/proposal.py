from fastapi import APIRouter, Body, HTTPException
from models.schema import ProposalRequest, ProposalResponse
from services.groq_service import generate_proposal_with_groq
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