from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class EstimationRequest(BaseModel):
    """Request model for project estimations."""
    job_description: str = Field(..., description="Job description to estimate")
    additional_context: Optional[str] = Field(None, description="Additional context for estimation")
    region: Optional[str] = Field("US", description="Region for cost estimation (affects rates)")
    currency: Optional[str] = Field("USD", description="Currency for cost estimation")
    skill_level: Optional[str] = Field("Intermediate", description="Skill level (Beginner, Intermediate, Expert)")
    estimation_type: Optional[str] = Field("all", description="Type of estimation (time, cost, resources, or all)")

class TimeEstimate(BaseModel):
    """Time estimation details."""
    min_hours: float
    max_hours: float
    estimated_duration: str
    confidence: str

class CostEstimate(BaseModel):
    """Cost estimation details."""
    min_amount: float
    max_amount: float
    currency: str
    hourly_rate_range: str

class ResourceEstimate(BaseModel):
    """Resource estimation details."""
    required_skills: List[str]
    recommended_tools: List[str] 
    team_size: str
    skill_level_needed: str

class EstimationResponse(BaseModel):
    """Response model for all types of estimations."""
    time: Optional[TimeEstimate] = None
    cost: Optional[CostEstimate] = None
    resources: Optional[ResourceEstimate] = None
    reasoning: str = Field(..., description="Explanation of the estimation factors")
    status: str = Field("success", description="Status of the request")
    model_used: str = Field(..., description="Model used for generation")