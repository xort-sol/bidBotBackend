# In services/response_service.py

import httpx
import random
import asyncio
import re
from fastapi import HTTPException
from models.schema import MessageSuggestionRequest
from models.schema import MessageEditRequest
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


async def generate_response_suggestion(request: MessageSuggestionRequest) -> str:
    # Build the conversation string from the history
    conversation_history = ""
    for msg in request.conversation:
        conversation_history += f"{msg.sender.capitalize()}: {msg.message}\n"

    # Define the prompt for the API request
    prompt = f"""
    Respond to the following conversation with a concise and professional message. Avoid lengthy introductions, salutations like "Dear client", and unnecessary formalities. Keep the response focused, realistic, and to the point:
    and the response should be less than 130 tokens
    {conversation_history}

    Context: {request.job_context}

    Desired tone: {request.tone}
    """

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": settings.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping to create message responses."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150  # Limiting the response length
    }

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

                # If rate-limited, wait and retry
                if response.status_code == 429:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                
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
                if 'choices' in response_data:
                    raw_response = response_data["choices"][0]["message"]["content"]
                    raw_response = clean_llm_response(raw_response)
                    return raw_response.strip()
                else:
                    raise HTTPException(status_code=500, detail="API response does not contain 'choices' field.")

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
    
    raise HTTPException(status_code=429, detail="Maximum retries exceeded due to rate limiting.")



async def edit_response_suggestion(request: MessageEditRequest) -> str:
    """
    Edit a previously generated response based on specific instructions.
    Includes comprehensive error handling and retries.
    and the response should be less than 180 tokens
    
    Args:
        request: MessageEditRequest containing original message and edit instructions
        
    Returns:
        str: Edited response text
    """
    # Build the conversation string from the history
    conversation_history = ""
    for msg in request.conversation:
        conversation_history += f"{msg.sender.capitalize()}: {msg.message}\n"

    # Define the prompt for the API request
    prompt = f"""
    Edit the following message based on the provided instructions.
    
    CONVERSATION CONTEXT:
    {conversation_history}
    
    JOB CONTEXT:
    {request.job_context}
    
    ORIGINAL MESSAGE:
    {request.original_message}
    
    EDIT INSTRUCTIONS:
    {request.edit_instructions}
    
    Return only the edited message without explanations or notes.
    """

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": settings.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping to edit message responses."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 200  # Slightly higher limit for edited responses
    }

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

                # If rate-limited, wait and retry
                if response.status_code == 429:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                
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
                if 'choices' in response_data:
                    raw_response = response_data["choices"][0]["message"]["content"]
                    raw_response = clean_llm_response(raw_response)
                    return raw_response.strip()
                else:
                    raise HTTPException(status_code=500, detail="API response does not contain 'choices' field.")

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
    
    raise HTTPException(status_code=429, detail="Maximum retries exceeded due to rate limiting.")


