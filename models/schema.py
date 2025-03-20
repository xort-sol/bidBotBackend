from pydantic import BaseModel, Field
from typing import List, Optional

class PreviousProposal(BaseModel):
    """Model for a previous proposal related to the job."""
    proposal_text: str = Field(..., description="Text of the previous proposal")
    submission_date: str = Field(..., description="Date when the proposal was submitted")
    status: str = Field(..., description="Status of the proposal (e.g., accepted, rejected, pending)")

class UserProject(BaseModel):
    """Model for a user's previous project."""
    project_name: str = Field(..., description="Name of the project")
    headline: str = Field(..., description="Headline or summary of the project")
    description: str = Field(..., description="Detailed description of the project")

class ProposalRequest(BaseModel):
    """Request model for proposal generation."""
    job_description: str = Field(..., description="Job description to generate a proposal for")
    additional_context: Optional[str] = Field(None, description="Additional context for the proposal")
    tone: Optional[str] = Field("Professional", description="Tone of the proposal")
    max_length: Optional[int] = Field(500, description="Maximum length of the proposal in tokens")
    model: Optional[str] = Field(None, description="Model to use for generation (if different from default)")
    
    previous_proposals: List[PreviousProposal] = Field([], description="List of previous proposals related to this job")
    associated_files: List[str] = Field([], description="List of file URLs or paths associated with the job")
    job_tags: List[str] = Field([], description="Tags related to the job (e.g., 'Python', 'Machine Learning', 'Web Development')")
    job_type: Optional[str] = Field(None, description="Type of job (e.g., 'Fixed Price', 'Hourly')")
    user_previous_projects: List[UserProject] = Field([], description="List of user's previous projects including name, description, and headline")


class ProposalResponse(BaseModel):
    """Response model for proposal generation."""
    proposal: str = Field(..., description="Generated proposal text")
    status: str = Field(..., description="Status of the request")
    model_used: str = Field(..., description="Model used for generation")

class EstimationResponse(BaseModel):
    time_range: dict = {"min_hours": int, "max_hours": int}
    cost_range: dict = {"min_amount": float, "max_amount": float, "currency": str}
    resources: list = ["required_skill_1", "required_skill_2"]
    confidence: str = Field(..., description="Low/Medium/High confidence in this estimate")
    reasoning: str = Field(..., description="Brief explanation of estimate factors")


class ChatMessage(BaseModel):
    sender: str = Field(..., description="Sender of the message: 'client' or 'freelancer'")
    message: str = Field(..., description="The message content")

class MessageSuggestionRequest(BaseModel):
    conversation: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    job_context: str = Field(..., description="Context of the job for better response generation")
    tone: str = Field("Professional", description="Desired tone of the response")

class MessageEditRequest(BaseModel):
    """Request model for editing a message suggestion."""
    original_message: str = Field(..., description="The original suggested message to edit")
    conversation: List[ChatMessage] = Field(..., description="List of messages in the conversation for context")
    job_context: str = Field(..., description="Context of the job related to this message")
    edit_instructions: str = Field(..., description="Specific instructions on how to edit the message")
    