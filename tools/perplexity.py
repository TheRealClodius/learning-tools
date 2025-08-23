"""
Perplexity API Integration Tools

This module provides functions to interact with Perplexity's search and research APIs.
Includes both quick search capabilities and comprehensive research operations.

Functions:
- perplexity_search: Quick search using Sonar models
- perplexity_research: Comprehensive research using Sonar Deep Research

API Documentation: https://docs.perplexity.ai/
"""

import httpx
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env.local
load_dotenv('.env.local')

from execute import (
    execute_perplexity_search_input, execute_perplexity_search_output,
    execute_perplexity_research_input, execute_perplexity_research_output
)

logger = logging.getLogger(__name__)

class PerplexityAPIError(Exception):
    """Raised when Perplexity API returns an error"""
    pass

class PerplexityAuthError(Exception):
    """Raised when API key is invalid or missing"""
    pass

# API Configuration
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_CHAT_ENDPOINT = f"{PERPLEXITY_BASE_URL}/chat/completions"

async def perplexity_search(input_data: Dict[str, Any], stream_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Execute a quick search query using Perplexity's Sonar models
    
    Args:
        input_data: Pydantic model instance with search parameters
        
    Returns:
        Pydantic model instance with search results
    """
    logger.info(f"Executing Perplexity search: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        # Get API key from environment, not from input (like weather tools)
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            return {
                "success": False,
                "message": "Perplexity API key not configured in environment variables",
                "data": {}
            }
        
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Build the request payload
        payload = {
            "model": input_data.get("model", "sonar"),
            "messages": [
                {
                    "role": "user",
                    "content": input_data.get("query")
                }
            ],
            "temperature": input_data.get("temperature", 0.2),
            "max_tokens": input_data.get("max_tokens", 1000),
        }
        
        # Enable streaming if callback provided
        if stream_callback:
            payload["stream"] = True
        
        # Add optional search parameters
        if input_data.get('search_domain_filter'):
            payload["search_domain_filter"] = input_data.get('search_domain_filter')
            
        if input_data.get('search_recency_filter'):
            payload["search_recency_filter"] = input_data.get('search_recency_filter')
            
        if input_data.get('return_images'):
            payload["return_images"] = input_data.get('return_images')
            
        if input_data.get('return_related_questions'):
            payload["return_related_questions"] = input_data.get('return_related_questions')
            
        if input_data.get('search_context_size'):
            payload["web_search_options"] = {
                "search_context_size": input_data.get('search_context_size')
            }
        
        logger.debug(f"Sending request to Perplexity API: {payload}")
        
        # Make the API request
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for search
            if stream_callback:
                # Streaming mode
                full_response = []
                async with client.stream("POST", PERPLEXITY_CHAT_ENDPOINT, headers=headers, json=payload) as response:
                    if response.status_code == 401:
                        raise PerplexityAuthError("Invalid API key")
                    if response.status_code != 200:
                        error_msg = f"Perplexity API error {response.status_code}: {await response.aread()}"
                        logger.error(error_msg)
                        raise PerplexityAPIError(error_msg)
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                await stream_callback(chunk)
                                full_response.append(chunk)
                            except json.JSONDecodeError:
                                continue
                
                # Build final response from accumulated chunks
                if full_response:
                    answer = ""
                    for chunk in full_response:
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                answer += delta["content"]
                    
                    # Use the last chunk for metadata
                    last_chunk = full_response[-1] if full_response else {}
                    api_response = {
                        "choices": [{"message": {"content": answer}}],
                        "model": last_chunk.get("model", input_data.get("model", "sonar")),
                        "usage": last_chunk.get("usage", {})
                    }
                else:
                    raise PerplexityAPIError("No response chunks received")
            else:
                # Non-streaming mode
                response = await client.post(
                    PERPLEXITY_CHAT_ENDPOINT,
                    headers=headers,
                    json=payload
                )
                
                # Check for authentication errors
                if response.status_code == 401:
                    raise PerplexityAuthError("Invalid API key")
                
                # Check for other API errors
                if response.status_code != 200:
                    error_msg = f"Perplexity API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise PerplexityAPIError(error_msg)
                
                # Parse response
                api_response = response.json()
                logger.debug(f"Perplexity API response: {api_response}")
            
            # Extract the answer and metadata
            answer = api_response["choices"][0]["message"]["content"]
            model = api_response.get("model", input_data.get("model"))
            usage = api_response.get("usage", {})
            
            # Extract citations if available
            citations = []
            if "citations" in api_response:
                # Citations are returned as a list of URL strings, not dictionaries
                for cite_url in api_response["citations"]:
                    citations.append({
                        "url": cite_url,
                        "title": "",  # Not provided in this format
                        "snippet": ""  # Not provided in this format
                    })
            
            # Extract related questions if available  
            related_questions = []
            if "related_questions" in api_response:
                related_questions = api_response["related_questions"]
            
            # Extract images if available
            images = []
            if "images" in api_response:
                for img in api_response["images"]:
                    if isinstance(img, dict):
                        images.append({
                            "url": img.get("url", ""),
                            "description": img.get("description", "")
                        })
                    else:
                        # If images are also just URLs
                        images.append({
                            "url": str(img),
                            "description": ""
                        })
            
            # Build response data
            response_data = {
                "answer": answer,
                "model": model,
                "usage": usage,
                "citations": citations,
                "related_questions": related_questions,
                "images": images
            }
            
            return {
                "success": True,
                "message": f"Search completed successfully using {model}",
                "data": response_data
            }
            
    except PerplexityAuthError as e:
        logger.error(f"Authentication error: {str(e)}")
        return {
            "success": False,
            "message": f"Authentication failed: {str(e)}",
            "data": {"answer": "", "model": ""}
        }
    except PerplexityAPIError as e:
        logger.error(f"API error: {str(e)}")
        return {
            "success": False,
            "message": f"API error: {str(e)}",
            "data": {"answer": "", "model": ""}
        }
    except Exception as e:
        logger.error(f"Unexpected error in perplexity_search: {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "data": {"answer": "", "model": ""}
        }

async def perplexity_research(input_data: Dict[str, Any], stream_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Execute comprehensive research using Perplexity's Sonar Deep Research model
    
    Args:
        input_data: Pydantic model instance with research parameters
        
    Returns:
        Pydantic model instance with research results
    """
    logger.info(f"Executing Perplexity research: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        # Get API key from environment, not from input (like weather tools)
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            logger.error("PERPLEXITY-RESEARCH: API key not configured in environment variables")
            return {
                "success": False,
                "message": "Perplexity API key not configured in environment variables",
                "data": {}
            }
        
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Build the request payload for deep research
        payload = {
            "model": input_data.get("model", "sonar-deep-research"),
            "messages": [
                {
                    "role": "user",
                    "content": input_data.get("directive")
                }
            ],
            "temperature": input_data.get("temperature", 0.1),
            "max_tokens": input_data.get("max_tokens", 4000),
        }
        
        # Enable streaming if callback provided
        if stream_callback:
            payload["stream"] = True
        
        # Add optional research parameters
        if input_data.get('search_domain_filter'):
            payload["search_domain_filter"] = input_data.get('search_domain_filter')
            
        if input_data.get('search_recency_filter'):
            payload["search_recency_filter"] = input_data.get('search_recency_filter')
            
        if input_data.get('search_after_date_filter'):
            payload["search_after_date_filter"] = input_data.get('search_after_date_filter')
            
        if input_data.get('search_before_date_filter'):
            payload["search_before_date_filter"] = input_data.get('search_before_date_filter')
            
        if input_data.get('return_related_questions'):
            payload["return_related_questions"] = input_data.get('return_related_questions')
            
        if input_data.get('search_context_size'):
            payload["web_search_options"] = {
                "search_context_size": input_data.get('search_context_size')
            }
        
        logger.debug(f"Sending research request to Perplexity API: {payload}")
        
        # Add streaming flag if stream_callback is provided
        if stream_callback:
            payload["stream"] = True
        
        # Make the API request (longer timeout for research)
        async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min timeout for research
            if stream_callback:
                # Streaming mode
                full_response = []
                logger.info(f"PERPLEXITY-RESEARCH: Starting streaming request to {PERPLEXITY_CHAT_ENDPOINT}")
                async with client.stream("POST", PERPLEXITY_CHAT_ENDPOINT, headers=headers, json=payload) as response:
                    if response.status_code == 401:
                        logger.error(f"PERPLEXITY-RESEARCH: Authentication failed - Invalid API key")
                        raise PerplexityAuthError("Invalid API key")
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_msg = f"Perplexity Research API error {response.status_code}: {error_body}"
                        logger.error(f"PERPLEXITY-RESEARCH: {error_msg}")
                        raise PerplexityAPIError(error_msg)
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                await stream_callback(chunk)
                                full_response.append(chunk)
                            except json.JSONDecodeError:
                                continue
                
                # Build final response from accumulated chunks
                if full_response:
                    # Extract content from the last chunk or combine chunks
                    research_report = ""
                    for chunk in full_response:
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                research_report += delta["content"]
                    
                    # Use the last chunk for metadata
                    last_chunk = full_response[-1] if full_response else {}
                    api_response = {
                        "choices": [{"message": {"content": research_report}}],
                        "model": last_chunk.get("model", input_data.get("model", "sonar-deep-research")),
                        "usage": last_chunk.get("usage", {})
                    }
                else:
                    raise PerplexityAPIError("No response chunks received")
            else:
                # Non-streaming mode
                response = await client.post(
                    PERPLEXITY_CHAT_ENDPOINT,
                    headers=headers,
                    json=payload
                )
                
                # Check for authentication errors
                if response.status_code == 401:
                    raise PerplexityAuthError("Invalid API key")
                
                # Check for other API errors
                if response.status_code != 200:
                    error_msg = f"Perplexity API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise PerplexityAPIError(error_msg)
                
                # Parse response
                api_response = response.json()
                logger.debug(f"Perplexity research API response: {api_response}")
            
            # Extract the research report and metadata
            research_report = api_response["choices"][0]["message"]["content"]
            model = api_response.get("model", input_data.get("model"))
            usage = api_response.get("usage", {})
            
            # Extract enhanced citations for research
            citations = []
            if "citations" in api_response:
                # Citations are returned as a list of URL strings, not dictionaries
                for cite_url in api_response["citations"]:
                    citations.append({
                        "url": cite_url,
                        "title": "",  # Not provided in this format
                        "snippet": "",  # Not provided in this format
                        "published_date": "",  # Not provided in this format
                        "domain": ""  # Not provided in this format
                    })
            
            # Extract related questions for further research
            related_questions = []
            if "related_questions" in api_response:
                related_questions = api_response["related_questions"]
            
            # Extract key findings from the research report
            key_findings = _extract_key_findings(research_report)
            
            # Build research methodology info
            research_methodology = {
                "search_strategy": f"Deep research using {model} model",
                "sources_evaluated": len(citations),
                "research_duration": "Variable (typically 5-30 minutes)"
            }
            
            # Add search count if available in usage
            if "search_count" in usage:
                research_methodology["search_strategy"] += f" with {usage['search_count']} searches"
            
            # Build response data
            response_data = {
                "research_report": research_report,
                "model": model,
                "usage": usage,
                "citations": citations,
                "related_questions": related_questions,
                "key_findings": key_findings,
                "research_methodology": research_methodology
            }
            
            return {
                "success": True,
                "message": f"Research completed successfully using {model}",
                "data": response_data
            }
            
    except PerplexityAuthError as e:
        logger.error(f"Authentication error: {str(e)}")
        return {
            "success": False,
            "message": f"Authentication failed: {str(e)}",
            "data": {"research_report": "", "model": ""}
        }
    except PerplexityAPIError as e:
        logger.error(f"API error: {str(e)}")
        return {
            "success": False,
            "message": f"API error: {str(e)}",
            "data": {"research_report": "", "model": ""}
        }
    except Exception as e:
        logger.error(f"PERPLEXITY-RESEARCH: Unexpected error in perplexity_research: {str(e)}")
        logger.error(f"PERPLEXITY-RESEARCH: Error type: {type(e).__name__}")
        import traceback
        logger.error(f"PERPLEXITY-RESEARCH: Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"Perplexity research failed: {str(e)}",
            "data": {"research_report": "", "model": ""}
        }



def _extract_key_findings(research_report: str) -> List[str]:
    """
    Extract key findings from a research report
    
    Args:
        research_report: The full research report text
        
    Returns:
        List of key findings
    """
    key_findings = []
    
    # Simple extraction logic - look for bullet points, numbered lists, or key sections
    lines = research_report.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for bullet points, numbered items, or sentences with key indicators
        if (line.startswith('•') or 
            line.startswith('-') or 
            line.startswith('*') or
            line.startswith(tuple(f'{i}.' for i in range(1, 10))) or
            'key finding' in line.lower() or
            'important' in line.lower() or
            'significant' in line.lower()):
            
            # Clean up the finding
            finding = line.lstrip('•-*0123456789. ').strip()
            if finding and len(finding) > 10:  # Avoid very short fragments
                key_findings.append(finding)
    
    # Limit to most relevant findings
    return key_findings[:10]

# Tool registration information
TOOL_INFO = {
    "perplexity.search": {
        "function": perplexity_search,
        "description": "Quick search using Perplexity's Sonar models",
        "category": "search"
    },
    "perplexity.research": {
        "function": perplexity_research,
        "description": "Comprehensive research using Perplexity's Deep Research model",
        "category": "research"
    }
} 