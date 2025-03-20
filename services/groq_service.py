import httpx
import re
import random
import time
from fastapi import HTTPException
from models.schema import ProposalRequest

from config.settings import settings


def clean_llm_response(text: str) -> str:
    """
    Clean LLM output by removing common prefixes, suffixes, and formatting issues.
    
    Args:
        text: The raw text response from the LLM
        
    Returns:
        Cleaned text ready for use
    """
    # Remove <think>...</think> blocks (including the tags and their contents)
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extended list of prefixes to remove (case insensitive)
    prefixes = [
        r"^Here(?:'s| is)(?: your| the)?(?: generated| sample| draft| proposed| custom)? proposal:?\s*",
        r"^I\'ve created(?: a| the)?(?: proposal| tailored proposal| custom proposal| response)(?: for you| based on your requirements| as requested):?\s*",
        r"^Based on your job description,?\s*(?:here(?:'s| is)(?: a| the)? proposal:?\s*)?",
        r"^Sure[,!]?\s*(?:I(?:'ll| will) create|here(?:'s| is))(?: a| the)?(?: proposal| response):?\s*",
        r"^Let me (?:create|draft|write)(?: a| the)?(?: proposal| response):?\s*",
        r"^(?:Alright|Okay|Got it)[,!]?\s*(?:here is|Below is|Following is)(?: a| the)?(?: proposal| response| Upwork proposal):?\s*",
        r"^Here is a concise and professional Upwork proposal that stands out from AI-generated content:?\s*",
        r"^\*\*Proposal for.*?\*\*\s*",
        r"^\"(?:\*\*)?(?:Proposal|Subject)(?:\*\*)?:.*?\"\s*",
        r"^\"(?:\*\*)?.*?(?:\*\*)?\"\s*",
        r"^\".*?\"\s*",
        r"^Subject:.*?\s*",
    ]
 
    # Apply prefixes removal
    for prefix in prefixes:
        cleaned_text = re.sub(prefix, '', cleaned_text, flags=re.IGNORECASE)

    # Common suffixes to remove (case insensitive)
    suffixes = [
        r'\s*(?:Let me know|Please let me know) if you (?:need|want|would like|require) any (?:changes|revisions|modifications|edits|adjustments)\.?',
        r'\s*I look forward to (?:hearing|discussing|working) with you\.?',
        r'\s*(?:Looking|I\'m looking) forward to your (?:response|reply)\.?',
        r'\s*(?:Thank you|Thanks) for your (?:consideration|time|opportunity)\.?'
    ]
    
    # Apply suffixes removal
    for suffix in suffixes:
        cleaned_text = re.sub(suffix, '', cleaned_text, flags=re.IGNORECASE)

    # Remove any leading/trailing quotes
    cleaned_text = re.sub(r'^["\']+|["\']+$', '', cleaned_text)
    
    # Remove potential markdown or formatting quotes at beginning or end
    cleaned_text = re.sub(r'^```.*?\s+|```$', '', cleaned_text)

    # Clean up extra whitespace, newlines, and tabs
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text
def create_proposal_prompt(request: ProposalRequest) -> str:
    """
    Create an optimized prompt for the Groq API based on the request.
    
    Args:
        request: ProposalRequest containing job description and parameters
        
    Returns:
        str: Formatted prompt for the LLM
    """
    system_prompt = "You are an expert freelancer who creates personalized, authentic-sounding proposals that win contracts. Respond ONLY with the final proposal text. Do not include any explanations, introductions, or meta-commentary."
    
    user_prompt = f"""
    I need a concise, professional proposal for the following job description. Output ONLY the final proposal - no introduction text, no 'here is the proposal', just the raw proposal text.

    [JOB POST]
    {request.job_description}
    [/JOB POST]

    Core Requirements:
    - Maximum length: 200-250 words (approximately 3 paragraphs)
    - Write in a {request.tone} tone
    - Be specific to the job details, not generic
    - Show expertise through specific approaches
    - DO NOT include 'Here is the proposal' or similar phrases at the beginning
    - DO NOT use quotes around the proposal
    - DO NOT include any meta-commentary before or after the proposal

    Paragraph Structure:
    - First Paragraph: Open with an insight about a specific challenge related to their project
    - Second Paragraph: Highlight relevant experience and technical approach
    - Third Paragraph: Include a question about their needs and suggest a next step

    Human-Like Writing Elements:
    - Use contractions naturally (I've, you're, doesn't)
    - Include a specific technical detail that shows advanced knowledge
    - Avoid generic phrases like "I am writing to express my interest"
    """

    if request.previous_proposals:
        user_prompt += "\n\nPrevious Proposals:\n"
        for proposal in request.previous_proposals:
            user_prompt += f"- Submitted on {proposal.submission_date}: {proposal.status}\n"

    if request.associated_files:
        user_prompt += "\n\nAssociated Files:\n"
        for file in request.associated_files:
            user_prompt += f"- {file}\n"

    if request.job_tags:
        user_prompt += "\n\nJob Tags: " + ", ".join(request.job_tags) + "\n"

    if request.job_type:
        user_prompt += f"\nJob Type: {request.job_type}\n"

    if request.user_previous_projects:
        user_prompt += "\n\nUser's Previous Projects:\n"
        for project in request.user_previous_projects:
            user_prompt += f"- {project.project_name}: {project.headline}\n  {project.description}\n"

    if request.additional_context:
        user_prompt += f"\n\nAdditional Context:\n{request.additional_context}\n"

    return {
        "system": system_prompt,
        "user": user_prompt
    }


async def generate_proposal_with_groq(request: ProposalRequest) -> str:
    """
    Call Groq API to generate a proposal based on the job description.
    
    Args:
        request: ProposalRequest model containing job description and parameters
        
    Returns:
        str: Generated proposal text
        
    Raises:
        HTTPException: If there's an error communicating with Groq API
    """
    # Validate that API key exists
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ API key not configured")
    
    # Create the prompt
    prompts = create_proposal_prompt(request)
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Set up the payload
    payload = {
        "model": settings.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ],
        "temperature": 0.7,
        "max_tokens": min(request.max_length, 4000)
    }
    
    # Retry parameters
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    for retry in range(max_retries):
        try:
            # Make the API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    settings.GROQ_API_URL,
                    headers=headers,
                    json=payload
                )
                
                # If rate limited, wait and retry
                if response.status_code == 429:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                    continue
                
                # Check if the response is successful
                if response.status_code != 200:
                    error_detail = f"Groq API error: {response.status_code}"
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_detail += f" - {error_json['error']['message']}"
                    except:
                        pass
                    raise HTTPException(status_code=500, detail=error_detail)
                
                # Parse the response
                response_data = response.json()
                
                # Extract the generated proposal
                raw_proposal_text = response_data["choices"][0]["message"]["content"]
                
                # Clean the response
                clean_proposal = clean_llm_response(raw_proposal_text)
                
                return clean_proposal
                
        except httpx.TimeoutException:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise HTTPException(status_code=504, detail="Request to Groq API timed out")
        except httpx.RequestError as e:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail=f"Error communicating with Groq API: {str(e)}")
        except Exception as e:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    # If we've exhausted all retries
    raise HTTPException(status_code=429, detail="Maximum retries exceeded due to rate limiting.")



