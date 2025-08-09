"""
Rate Limit Handler for API Calls

Provides intelligent rate limit handling with:
- Exponential backoff for retries
- Request queuing to prevent bursts
- User-friendly error messages
- Automatic retry with progressive delays
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import yaml
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class RateLimitConfig:
    """Configuration for rate limit handling"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    
    # Token bucket configuration
    tokens_per_minute: int = 10000  # Adjust based on your plan
    burst_size: int = 2000  # Maximum burst allowed
    
    # Queue configuration
    max_queue_size: int = 100
    queue_timeout: float = 300.0  # 5 minutes

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, tokens_per_minute: int, burst_size: int):
        self.tokens_per_minute = tokens_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int) -> tuple[bool, float]:
        """
        Try to consume tokens from the bucket.
        Returns (success, wait_time_if_failed)
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            tokens_to_add = elapsed * (self.tokens_per_minute / 60.0)
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculate wait time needed
                tokens_needed = tokens - self.tokens
                wait_time = (tokens_needed / self.tokens_per_minute) * 60.0
                return False, wait_time

class RateLimitHandler:
    """Handles rate limiting with intelligent retry and queueing"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.token_bucket = TokenBucket(
            self.config.tokens_per_minute,
            self.config.burst_size
        )
        self.request_queue: deque = deque()
        self.processing = False
        self._queue_processor_task = None
    
    async def start(self):
        """Start the queue processor"""
        if not self._queue_processor_task:
            self._queue_processor_task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop the queue processor"""
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_queue(self):
        """Process queued requests"""
        while True:
            try:
                if not self.request_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next request from queue
                request_data = self.request_queue.popleft()
                func, args, kwargs, future, token_estimate = request_data
                
                # Check if request has timed out
                if hasattr(future, '_timeout') and time.time() > future._timeout:
                    future.set_exception(TimeoutError("Request timed out in queue"))
                    continue
                
                # Wait for tokens to be available
                success, wait_time = await self.token_bucket.consume(token_estimate)
                if not success:
                    await asyncio.sleep(wait_time)
                    # Re-add to front of queue
                    self.request_queue.appendleft(request_data)
                    continue
                
                # Execute the request
                try:
                    result = await func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        estimate_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        """Execute a function with rate limit handling and retry logic"""
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # If we have token estimate, check bucket first
                if estimate_tokens:
                    success, wait_time = await self.token_bucket.consume(estimate_tokens)
                    if not success:
                        # Add to queue instead of immediate retry
                        future = asyncio.Future()
                        future._timeout = time.time() + self.config.queue_timeout
                        
                        self.request_queue.append((
                            func, args, kwargs, future, estimate_tokens
                        ))
                        
                        # Ensure queue processor is running
                        await self.start()
                        
                        return await future
                
                # Execute the function
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_error = e
                error_info = self._parse_error(e)
                
                if error_info['is_rate_limit']:
                    retry_count += 1
                    
                    if retry_count > self.config.max_retries:
                        raise RateLimitError(
                            self._get_user_friendly_message(error_info, retry_count)
                        )
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.config.initial_delay * (self.config.backoff_factor ** (retry_count - 1)),
                        self.config.max_delay
                    )
                    
                    # If error provides retry_after, use that instead
                    if error_info.get('retry_after'):
                        delay = error_info['retry_after']
                    
                    logger.warning(
                        f"Rate limit hit, retrying in {delay}s "
                        f"(attempt {retry_count}/{self.config.max_retries})"
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    # Not a rate limit error, re-raise
                    raise
        
        # Exhausted retries
        raise RateLimitError(
            self._get_user_friendly_message(
                self._parse_error(last_error),
                self.config.max_retries + 1
            )
        )
    
    def _parse_error(self, error: Exception) -> Dict[str, Any]:
        """Parse error to extract rate limit information"""
        error_str = str(error)
        error_info = {
            'is_rate_limit': False,
            'error_type': type(error).__name__,
            'message': error_str,
            'retry_after': None
        }
        
        # Check for rate limit indicators
        rate_limit_indicators = [
            'rate_limit_error',
            'rate limit',
            '429',
            'too many requests',
            'exceeded quota',
            'usage limit'
        ]
        
        error_lower = error_str.lower()
        for indicator in rate_limit_indicators:
            if indicator in error_lower:
                error_info['is_rate_limit'] = True
                break
        
        # Try to extract retry-after time if available
        # This is a simple pattern match, adjust based on actual error format
        import re
        retry_match = re.search(r'retry[_\s-]?after[:\s]+(\d+)', error_lower)
        if retry_match:
            error_info['retry_after'] = int(retry_match.group(1))
        
        return error_info
    
    def _get_user_friendly_message(self, error_info: Dict[str, Any], attempt: int) -> str:
        """Generate user-friendly error message"""
        if attempt == 1:
            return (
                "🔄 I'm experiencing high demand right now. "
                "Let me try again in just a moment..."
            )
        elif attempt <= self.config.max_retries:
            return (
                f"⏳ Still working on your request (attempt {attempt}). "
                "The service is quite busy, but I'll keep trying..."
            )
        else:
            return (
                "😔 I'm sorry, but the service is currently overloaded and I couldn't complete your request. "
                "This usually happens during peak usage times. Please try again in a few minutes, "
                "or try breaking your request into smaller parts."
            )

class RateLimitError(Exception):
    """Custom exception for rate limit errors with user-friendly messages"""
    pass

# Decorator for easy rate limit handling
def with_rate_limit(
    handler: Optional[RateLimitHandler] = None,
    estimate_tokens: Optional[Union[int, Callable]] = None
):
    """Decorator to add rate limit handling to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal handler
            if handler is None:
                handler = RateLimitHandler()
            
            # Calculate token estimate
            tokens = None
            if estimate_tokens:
                if callable(estimate_tokens):
                    tokens = estimate_tokens(*args, **kwargs)
                else:
                    tokens = estimate_tokens
            
            return await handler.execute_with_retry(
                func,
                *args,
                estimate_tokens=tokens,
                **kwargs
            )
        return wrapper
    return decorator

# Global rate limit handler instance
_global_handler = None

def load_config_from_yaml(tier: Optional[str] = None) -> RateLimitConfig:
    """Load rate limit configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'rate_limits.yaml')
    
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Start with default configuration
        config_dict = yaml_config.get('default', {})
        
        # Override with tier-specific settings if provided
        if tier and 'tiers' in yaml_config and tier in yaml_config['tiers']:
            tier_config = yaml_config['tiers'][tier]
            config_dict.update(tier_config)
        
        # Create RateLimitConfig with values from YAML
        return RateLimitConfig(
            max_retries=config_dict.get('max_retries', 3),
            initial_delay=config_dict.get('initial_delay', 2.0),
            max_delay=config_dict.get('max_delay', 60.0),
            backoff_factor=config_dict.get('backoff_factor', 2.0),
            tokens_per_minute=config_dict.get('tokens_per_minute', 40000),
            burst_size=config_dict.get('burst_size', 8000),
            max_queue_size=config_dict.get('max_queue_size', 100),
            queue_timeout=config_dict.get('queue_timeout', 300.0)
        )
    except Exception as e:
        logger.warning(f"Failed to load rate limit config from YAML: {e}. Using defaults.")
        return RateLimitConfig()

def get_global_handler() -> RateLimitHandler:
    """Get or create global rate limit handler"""
    global _global_handler
    if _global_handler is None:
        # Load configuration from environment or use default
        tier = os.getenv('ANTHROPIC_TIER', 'default')
        config = load_config_from_yaml(tier)
        _global_handler = RateLimitHandler(config)
    return _global_handler

async def initialize_rate_limiting(config: Optional[RateLimitConfig] = None):
    """Initialize global rate limiting"""
    global _global_handler
    _global_handler = RateLimitHandler(config)
    await _global_handler.start()

async def shutdown_rate_limiting():
    """Shutdown global rate limiting"""
    global _global_handler
    if _global_handler:
        await _global_handler.stop()
        _global_handler = None