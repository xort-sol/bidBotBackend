from typing import List
import httpx
import re
import random
import time
from fastapi import HTTPException
from models.schema import ProposalRequest
from models.schema import EditProposalRequest
from models.schema import EditProposalResponse
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
    Create a prompt that generates genuinely human-like proposals with varied formatting.
    
    Args:
        request: ProposalRequest containing job description and parameters
        
    Returns:
        dict: System and user prompts for the LLM
    """
    # Random greeting selection to add variation
    greetings = [
        "Hi there!",
        "Hey,",
        "Hello,",
        "Hi,",
        "Morning/Afternoon/Evening (depending on time),",
        "Howdy,"
    ]
    
    # A more casual, authentic system prompt
    system_prompt = "You are a freelancer quickly typing up a proposal for a job. Write like a real person with natural formatting, occasional typos, and genuinely human style. Include a casual greeting."
    
    user_prompt = f"""
    Write a quick proposal for this job as if you're typing it directly into an application form:

    [JOB POST]
    {request.job_description}
    [/JOB POST]

    Make it genuinely human by:
    - Starting with a casual greeting like "{random.choice(greetings)}"
    - Writing in a {request.tone} style, but inconsistently (formal in some parts, casual in others)
    - Including 1-2 authentic typos or missing words (common human errors)
    - Using some incomplete sentences or run-ons where natural
    - Adding personal details that sound real and specific
    - Using varied formatting - sometimes paragraphs, sometimes short 1-liners
    - Including a list or bullet points in ONE section (but make it look casual, not perfectly formatted)
    - Maybe using a dash or asterisk instead of a proper bullet point
    - Having some paragraphs with just 1-2 sentences
    - Breaking up text with natural spacing (an extra line break occasionally)
    - Adding natural thought jumps (changing topics mid-paragraph sometimes)
    - Using 1-2 filler phrases ("anyway", "you know", "actually", "btw")
    - Being specific about availability ("free Thursday after 2pm" instead of "anytime")
    
    Absolutely avoid:
    - Perfect paragraph structure or even paragraph lengths
    - Starting with a problem-solution structure
    - Using formal transition words (furthermore, moreover, additionally)
    - Listing technical skills in sequence
    - Perfect grammar throughout the entire proposal
    - Generic closing questions
    - Looking too organized or perfectly structured
    
    Keep it under 250 words but don't count exactly. Write it like you're typing quickly without much editing.
    """

    # Add any additional context from the request
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
    
    # Use higher temperature for more randomness and natural output
    payload = {
        "model": settings.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ],
        "temperature": 0.85,  # Increased for more randomness
        "top_p": 0.95,  # Slightly wider sampling
        "max_tokens": min(request.max_length, 4000)
    }
    
    # Rest of the function remains the same...
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
                    time.sleep(delay)
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
                
                # Apply occasional post-processing for even more human feel
                if random.random() < 0.3:  # 30% chance
                    clean_proposal = apply_human_quirks(clean_proposal)
                
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


def apply_human_quirks(text: str) -> str:
    """
    Apply additional human-like quirks to the generated text.
    This adds another layer of randomness to make detection even harder.
    
    Args:
        text: The cleaned proposal text
        
    Returns:
        Text with additional human quirks
    """
    # Only apply some changes randomly to avoid patterns
    result = text
    
    # Occasionally double a letter (common typing error)
    if random.random() < 0.4:
        words = result.split()
        if len(words) > 5:
            word_idx = random.randint(0, len(words) - 1)
            if len(words[word_idx]) > 3:
                char_idx = random.randint(1, len(words[word_idx]) - 2)
                char = words[word_idx][char_idx]
                words[word_idx] = words[word_idx][:char_idx] + char + words[word_idx][char_idx:]
                result = ' '.join(words)
    
    # Occasionally add an extra space
    if random.random() < 0.3:
        words = result.split()
        if len(words) > 3:
            word_idx = random.randint(0, len(words) - 2)
            words[word_idx] = words[word_idx] + "  "  # Double space
            result = ' '.join(words)
    
    # Sometimes forget to capitalize a sentence
    if random.random() < 0.25:
        sentences = re.split(r'(?<=[.!?])\s+', result)
        if len(sentences) > 2:
            sent_idx = random.randint(1, len(sentences) - 1)
            if len(sentences[sent_idx]) > 0 and sentences[sent_idx][0].isupper():
                sentences[sent_idx] = sentences[sent_idx][0].lower() + sentences[sent_idx][1:]
                result = ' '.join(sentences)
    
    return result





def create_edit_proposal_prompt(request: EditProposalRequest) -> dict:
    """
    Create a prompt for editing an existing proposal in a human-like way.
    
    Args:
        request: EditProposalRequest containing original proposal and edit instructions
        
    Returns:
        dict: System and user prompts for the LLM
    """
    system_prompt = "You are a freelancer quickly editing a proposal you already wrote. Make edits that look natural and human, preserving the original style but improving based on the instructions."
    
    user_prompt = f"""
    I need to edit this proposal I already submitted. Here's my original proposal and what I want to change:
    
    [ORIGINAL PROPOSAL]
    {request.original_proposal}
    [/ORIGINAL PROPOSAL]
    
    [EDIT INSTRUCTIONS]
    {request.edit_instructions}
    [/EDIT INSTRUCTIONS]
    
    When editing, make sure to:
    - Keep my original casual/messy writing style and tone
    - Preserve any typos, run-ons, or informal elements UNLESS they're part of what I want to fix
    - Maintain the original greeting and overall structure if possible
    - Make edits look natural, like I quickly revised it myself
    - Keep any bullet points, short paragraphs, or unique formatting from the original
    - Don't make it look "better" or more polished than the original
    - If adding new content, match the style of the original (including occasional natural imperfections)
    - Be specific about any technical details you add (don't be generic)
    """
    
    # Add original job description if available
    if request.job_description:
        user_prompt += f"""
        
        For reference, here was the original job posting:
        [JOB POST]
        {request.job_description}
        [/JOB POST]
        """
    
    # Add tone instructions if different from original
    if request.tone and request.tone != "Same as original":
        user_prompt += f"""
        
        While preserving the original style, adjust the overall tone to be more {request.tone}.
        """
    
    # Add any additional context
    if request.associated_files:
        user_prompt += "\n\nAssociated Files:\n"
        for file in request.associated_files:
            user_prompt += f"- {file}\n"

    if request.job_tags:
        user_prompt += "\n\nJob Tags: " + ", ".join(request.job_tags) + "\n"

    if request.job_type:
        user_prompt += f"\nJob Type: {request.job_type}\n"
    
    user_prompt += """
    
    Don't explain your changes. Just give me the revised proposal text that I can directly copy and paste.
    """
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


async def edit_proposal_with_groq(request: EditProposalRequest) -> EditProposalResponse:
    """
    Call Groq API to edit an existing proposal based on the instructions.
    
    Args:
        request: EditProposalRequest model containing original proposal and edit instructions
        
    Returns:
        EditProposalResponse: Response containing edited proposal
        
    Raises:
        HTTPException: If there's an error communicating with Groq API
    """
    # Validate that API key exists
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ API key not configured")
    
    # Create the prompt
    prompts = create_edit_proposal_prompt(request)
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Set up the payload with randomized temperature for more human-like output
    payload = {
        "model": request.model or settings.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ],
        "temperature": random.uniform(0.85, 1.05),  # Randomized high temperature
        "top_p": 0.92,  # Slightly wider sampling
        "max_tokens": min(request.max_length, 4000),
        "frequency_penalty": 0.5,  # Reduce repetition patterns
        "presence_penalty": 0.3   # Encourage diversity
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
                    time.sleep(delay)
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
                raw_edited_text = response_data["choices"][0]["message"]["content"]
                
                # Clean the response
                clean_edited_proposal = clean_llm_response(raw_edited_text)
                
                # Determine what changes were made (simplified version)
                changes = determine_changes(request.original_proposal, clean_edited_proposal)
                
                # Apply occasional post-processing for even more human feel if preserve_quirks is True
                if request.preserve_quirks and random.random() < 0.3:  # 30% chance
                    clean_edited_proposal = apply_human_quirks(clean_edited_proposal)
                
                return EditProposalResponse(
                    original_proposal=request.original_proposal,
                    edited_proposal=clean_edited_proposal,
                    changes_made=changes,
                    status="success",
                    model_used=request.model or settings.DEFAULT_MODEL
                )
                
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


def determine_changes(original: str, edited: str) -> List[str]:
    """
    Simple function to determine key changes made between original and edited versions.
    
    Args:
        original: Original proposal text
        edited: Edited proposal text
        
    Returns:
        List of change descriptions
    """
    changes = []
    
    # Length change
    original_words = len(original.split())
    edited_words = len(edited.split())
    word_diff = edited_words - original_words
    
    if abs(word_diff) > 5:
        if word_diff > 0:
            changes.append(f"Added approximately {word_diff} words")
        else:
            changes.append(f"Removed approximately {abs(word_diff)} words")
    
    # Simple checks for common changes
    if len(edited) > len(original) * 1.2:
        changes.append("Significantly expanded content")
    elif len(edited) < len(original) * 0.8:
        changes.append("Significantly condensed content")
    
    # Very basic content change detection (could be enhanced with more sophisticated diff algorithms)
    orig_paragraphs = original.split('\n\n')
    edit_paragraphs = edited.split('\n\n')
    
    if len(orig_paragraphs) != len(edit_paragraphs):
        changes.append("Adjusted paragraph structure")
    
    # Check for bullet points
    if ('- ' in edited or '* ' in edited) and not ('- ' in original or '* ' in original):
        changes.append("Added bullet points")
    
    # If we couldn't detect any specific changes
    if not changes:
        changes.append("Made minor revisions throughout")
    
    return changes