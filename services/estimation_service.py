import asyncio
import httpx
import json
import re
import random
import time
from fastapi import HTTPException
from models.estimation_schema import EstimationRequest, TimeEstimate, CostEstimate, ResourceEstimate
from config.settings import settings

async def analyze_job_with_llm(request: EstimationRequest) -> dict:
    """
    Initial analysis of the job description to extract key parameters.
    
    Args:
        request: EstimationRequest with job description
        
    Returns:
        dict: Job analysis with extracted parameters
    """
    system_prompt = "You are an expert freelancer who accurately analyzes project requirements and extracts key information for estimation purposes."
    
    user_prompt = f"""
Analyze this job description and extract the following information in JSON format:

[JOB DESCRIPTION]
{request.job_description}
[/JOB DESCRIPTION]

Extract and return ONLY a JSON object with the following keys:
1. "primary_skills": [list of up to 5 technical skills required]
2. "complexity": A rating of "Simple", "Moderate", or "Complex" with brief reasoning
3. "deliverable_types": [list of required deliverables]
4. "timeline_constraints": Any deadlines or time requirements mentioned, or "None specified"
5. "quality_expectations": Any quality standards mentioned, or "Standard quality"
6. "project_type": The category of work (e.g., "Web Development", "Mobile App", "Content Writing", "Design")

Return ONLY the JSON object with no additional text. Ensure the JSON is valid.
"""
    
    if request.additional_context:
        user_prompt += f"""
Additional Context:
{request.additional_context}
"""
    
    # Make API call
    response_text = await query_groq({
        "system": system_prompt,
        "user": user_prompt
    })
    
    # Extract JSON from response
    try:
        # Remove any non-JSON text before or after the JSON object
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            analysis = json.loads(clean_json)
            return analysis
        else:
            # If no JSON found, try to parse the whole response
            analysis = json.loads(response_text)
            return analysis
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse job analysis response")

async def estimate_time(request: EstimationRequest, job_analysis: dict) -> TimeEstimate:
    """
    Estimate time required for the project.
    
    Args:
        request: EstimationRequest with job details
        job_analysis: Pre-analyzed job parameters
        
    Returns:
        TimeEstimate: Estimated time range and explanation
    """
    system_prompt = "You are an expert project estimator who provides accurate time estimates for freelance projects."
    
    user_prompt = f"""
Based on this job analysis:
{json.dumps(job_analysis, indent=2)}

Estimate the time required to complete this project.

Return ONLY a JSON object with these exact keys:
1. "min_hours": (number) minimum hours required
2. "max_hours": (number) maximum hours required  
3. "estimated_duration": (string) human-readable time estimate (e.g., "2-3 days")
4. "confidence": (string) "Low", "Medium", or "High" confidence in estimate

Consider setup time, development, testing, and client revisions in your estimate.
Your response must be ONLY the valid JSON object with no additional text.
"""
    
    # Make API call
    response_text = await query_groq({
        "system": system_prompt,
        "user": user_prompt
    })
    
    # Parse response
    try:
        # Extract JSON from response text
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            estimate_data = json.loads(clean_json)
        else:
            estimate_data = json.loads(response_text)
            
        return TimeEstimate(
            min_hours=float(estimate_data["min_hours"]),
            max_hours=float(estimate_data["max_hours"]),
            estimated_duration=estimate_data["estimated_duration"],
            confidence=estimate_data["confidence"]
        )
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=500, detail="Failed to generate time estimate")

async def estimate_cost(request: EstimationRequest, job_analysis: dict, time_estimate: TimeEstimate) -> CostEstimate:
    """
    Estimate cost based on time estimate and region.
    
    Args:
        request: EstimationRequest with job details
        job_analysis: Pre-analyzed job parameters
        time_estimate: Previously calculated time estimate
        
    Returns:
        CostEstimate: Estimated cost range and rates
    """
    system_prompt = "You are an expert freelance pricing strategist who provides accurate cost estimates for projects."
    
    user_prompt = f"""
Based on this job analysis:
{json.dumps(job_analysis, indent=2)}

And this time estimate:
- Min hours: {time_estimate.min_hours}
- Max hours: {time_estimate.max_hours}
- Duration: {time_estimate.estimated_duration}

Estimate an appropriate cost range for a {request.skill_level} freelancer in {request.region}.

Return ONLY a JSON object with these exact keys:
1. "min_amount": (number) minimum project cost in {request.currency}
2. "max_amount": (number) maximum project cost in {request.currency}
3. "currency": "{request.currency}"
4. "hourly_rate_range": (string) e.g., "$50-70"

Your response must be ONLY the valid JSON object with no additional text.
"""
    
    # Make API call
    response_text = await query_groq({
        "system": system_prompt,
        "user": user_prompt
    })
    
    # Parse response
    try:
        # Extract JSON from response text
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            estimate_data = json.loads(clean_json)
        else:
            estimate_data = json.loads(response_text)
            
        return CostEstimate(
            min_amount=float(estimate_data["min_amount"]),
            max_amount=float(estimate_data["max_amount"]),
            currency=estimate_data["currency"],
            hourly_rate_range=estimate_data["hourly_rate_range"]
        )
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=500, detail="Failed to generate cost estimate")

async def estimate_resources(request: EstimationRequest, job_analysis: dict) -> ResourceEstimate:
    """
    Estimate resources required for the project.
    
    Args:
        request: EstimationRequest with job details
        job_analysis: Pre-analyzed job parameters
        
    Returns:
        ResourceEstimate: Required skills, tools, and team size
    """
    system_prompt = "You are an expert project planner who accurately assesses resource requirements for freelance projects."
    
    user_prompt = f"""
Based on this job analysis:
{json.dumps(job_analysis, indent=2)}

Estimate the resources required to complete this project.

Return ONLY a JSON object with these exact keys:
1. "required_skills": [list of specific skills needed]
2. "recommended_tools": [list of software/tools recommended]
3. "team_size": (string) e.g., "1 person", "2-3 people"
4. "skill_level_needed": (string) "Beginner", "Intermediate", "Expert", or "Mixed"

Your response must be ONLY the valid JSON object with no additional text.
"""
    
    # Make API call
    response_text = await query_groq({
        "system": system_prompt,
        "user": user_prompt
    })
    
    # Parse response
    try:
        # Extract JSON from response text
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            estimate_data = json.loads(clean_json)
        else:
            estimate_data = json.loads(response_text)
            
        return ResourceEstimate(
            required_skills=estimate_data["required_skills"],
            recommended_tools=estimate_data["recommended_tools"],
            team_size=estimate_data["team_size"],
            skill_level_needed=estimate_data["skill_level_needed"]
        )
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=500, detail="Failed to generate resource estimate")

async def generate_reasoning(request: EstimationRequest, job_analysis: dict, estimates: dict) -> str:
    """
    Generate an explanation for the estimates.
    
    Args:
        request: EstimationRequest with job details
        job_analysis: Pre-analyzed job parameters
        estimates: Dictionary containing any estimates that were generated
        
    Returns:
        str: Human-readable explanation of estimates
    """
    system_prompt = "You are an expert freelancer who explains project estimates clearly and concisely."
    
    # Build a summary of the estimates for the prompt
    estimate_summary = ""
    if "time" in estimates:
        estimate_summary += f"""
Time Estimate:
- {estimates['time'].min_hours} to {estimates['time'].max_hours} hours
- Estimated duration: {estimates['time'].estimated_duration}
- Confidence: {estimates['time'].confidence}
"""
    
    if "cost" in estimates:
        estimate_summary += f"""
Cost Estimate:
- {estimates['cost'].min_amount} to {estimates['cost'].max_amount} {estimates['cost'].currency}
- Hourly rate range: {estimates['cost'].hourly_rate_range}
"""
    
    if "resources" in estimates:
        estimate_summary += f"""
Resource Estimate:
- Required skills: {', '.join(estimates['resources'].required_skills)}
- Team size: {estimates['resources'].team_size}
- Skill level: {estimates['resources'].skill_level_needed}
"""
    
    user_prompt = f"""
Based on this job analysis:
{json.dumps(job_analysis, indent=2)}

And these estimates:
{estimate_summary}

Generate a concise explanation (2-3 paragraphs) of these estimates. Explain the key factors that influenced the estimates and any important considerations the freelancer should keep in mind.

Your response should be ONLY the explanation text with no additional formatting or meta-commentary.
"""
    
    # Make API call
    response_text = await query_groq({
        "system": system_prompt,
        "user": user_prompt
    })
    
    return response_text

async def query_groq(prompts: dict) -> str:
    """
    Call Groq API with system and user prompts.
    
    Args:
        prompts: Dictionary with "system" and "user" prompts
        
    Returns:
        str: LLM response text
        
    Raises:
        HTTPException: If there's an error communicating with Groq API
    """
    # Validate that API key exists
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ API key not configured")
    
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
        "temperature": 0.3,  # Lower temperature for more consistent results
        "max_tokens": 2000
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
                    await asyncio.sleep(delay)
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
                
                # Extract the generated text
                return response_data["choices"][0]["message"]["content"]
                
        except httpx.TimeoutException:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            raise HTTPException(status_code=504, detail="Request to Groq API timed out")
        except httpx.RequestError as e:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail=f"Error communicating with Groq API: {str(e)}")
        except Exception as e:
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    # If we've exhausted all retries
    raise HTTPException(status_code=429, detail="Maximum retries exceeded due to rate limiting.")