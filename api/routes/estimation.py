from fastapi import APIRouter, Body, HTTPException
from models.estimation_schema import EstimationRequest, EstimationResponse
from services.estimation_service import (
    analyze_job_with_llm,
    estimate_time,
    estimate_cost,
    estimate_resources,
    generate_reasoning
)
from config.settings import settings

router = APIRouter(
    prefix="/api",
    tags=["estimation"]
)

@router.post("/estimateProject", response_model=EstimationResponse)
async def estimate_project(request: EstimationRequest = Body(...)):
    """
    Generate comprehensive project estimates based on the job description.
    
    Args:
        request: EstimationRequest model containing job description and parameters
        
    Returns:
        EstimationResponse: Time, cost, and resource estimates with reasoning
    """
    try:
        # Get model to use (either from request or default)
        model_used = settings.DEFAULT_MODEL
        
        # Step 1: Analyze the job to extract key parameters
        job_analysis = await analyze_job_with_llm(request)
        
        # Initialize response dictionary
        estimates = {}
        
        # Step 2: Generate estimates based on estimation_type
        if request.estimation_type in ["time", "all"]:
            time_estimate = await estimate_time(request, job_analysis)
            estimates["time"] = time_estimate
            
        if request.estimation_type in ["cost", "all"]:
            # Ensure we have a time estimate for the cost calculation
            if "time" not in estimates:
                time_estimate = await estimate_time(request, job_analysis)
                estimates["time"] = time_estimate
                
            cost_estimate = await estimate_cost(request, job_analysis, estimates["time"])
            estimates["cost"] = cost_estimate
            
        if request.estimation_type in ["resources", "all"]:
            resource_estimate = await estimate_resources(request, job_analysis)
            estimates["resources"] = resource_estimate
            
        # Step 3: Generate explanation of estimates
        reasoning = await generate_reasoning(request, job_analysis, estimates)
        
        # Step 4: Return the response
        response = EstimationResponse(
            reasoning=reasoning,
            status="success",
            model_used=model_used
        )
        
        # Add estimates to the response if they exist
        if "time" in estimates:
            response.time = estimates["time"]
        if "cost" in estimates:
            response.cost = estimates["cost"]
        if "resources" in estimates:
            response.resources = estimates["resources"]
            
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to generate estimates: {str(e)}")