"""
Prompt Caching Module for Claude API

This module provides intelligent caching for Claude API prompts to reduce costs and improve performance.
It supports both in-memory caching and Redis-based persistent caching, with special handling for
Anthropic's prompt caching features.

Features:
- System prompt caching with cache_control blocks
- Message history caching
- Tool schema caching
- Configurable TTL and cache sizes
- Fallback from Redis to in-memory caching
- Cache hit/miss metrics
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from functools import wraps

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Statistics for cache performance monitoring"""
    hits: int = 0
    misses: int = 0
    system_prompt_hits: int = 0
    system_prompt_misses: int = 0
    message_cache_hits: int = 0
    message_cache_misses: int = 0
    total_tokens_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging/monitoring"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "system_prompt_hits": self.system_prompt_hits,
            "system_prompt_misses": self.system_prompt_misses,
            "message_cache_hits": self.message_cache_hits,
            "message_cache_misses": self.message_cache_misses,
            "total_tokens_saved": self.total_tokens_saved
        }

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    content: Any
    created_at: float
    ttl: int
    token_count: int = 0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()

class PromptCache:
    """
    Intelligent prompt caching system for Claude API with support for:
    - System prompt caching with Anthropic's cache_control
    - Message history caching
    - Tool schema caching
    - Both in-memory and Redis storage
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_memory_entries: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        system_prompt_ttl: int = 86400,  # 24 hours
        enable_anthropic_caching: bool = True
    ):
        """
        Initialize the prompt cache
        
        Args:
            redis_url: Redis connection URL (optional)
            max_memory_entries: Maximum entries in in-memory cache
            default_ttl: Default TTL for cache entries in seconds
            system_prompt_ttl: TTL for system prompts in seconds
            enable_anthropic_caching: Enable Anthropic's prompt caching features
        """
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.system_prompt_ttl = system_prompt_ttl
        self.enable_anthropic_caching = enable_anthropic_caching
        
        # In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        # Redis cache setup
        self._redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(
                    redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"),
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Connected to Redis for prompt caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory caching.")
                self._redis_client = None
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Cache key prefixes
        self.SYSTEM_PROMPT_PREFIX = "sys_prompt:"
        self.MESSAGE_PREFIX = "messages:"
        self.TOOL_SCHEMA_PREFIX = "tools:"
        
        logger.info(f"PromptCache initialized with Redis: {self._redis_client is not None}, "
                   f"Memory limit: {max_memory_entries}, Default TTL: {default_ttl}s")
    
    def _generate_cache_key(self, content: Any, prefix: str = "") -> str:
        """Generate a stable cache key from content"""
        if isinstance(content, dict):
            # Sort keys for consistent hashing
            content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        elif isinstance(content, list):
            content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        else:
            content_str = str(content)
        
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return f"{prefix}{hash_obj.hexdigest()[:16]}"
    
    def _cleanup_memory_cache(self):
        """Remove expired entries and enforce size limits"""
        # Remove expired entries
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._memory_cache[key]
        
        # Enforce size limit by removing least recently used entries
        if len(self._memory_cache) > self.max_memory_entries:
            # Sort by last_accessed time
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            # Remove oldest entries
            remove_count = len(self._memory_cache) - self.max_memory_entries
            for key, _ in sorted_entries[:remove_count]:
                del self._memory_cache[key]
    
    async def _get_from_redis(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis cache"""
        if not self._redis_client:
            return None
        
        try:
            data = self._redis_client.get(key)
            if data:
                entry_data = json.loads(data)
                entry = CacheEntry(
                    content=entry_data['content'],
                    created_at=entry_data['created_at'],
                    ttl=entry_data['ttl'],
                    token_count=entry_data.get('token_count', 0),
                    access_count=entry_data.get('access_count', 0),
                    last_accessed=entry_data.get('last_accessed', time.time())
                )
                if not entry.is_expired:
                    entry.touch()
                    return entry
                else:
                    # Clean up expired entry
                    self._redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Error reading from Redis cache: {e}")
        
        return None
    
    async def _set_to_redis(self, key: str, entry: CacheEntry):
        """Set entry in Redis cache"""
        if not self._redis_client:
            return
        
        try:
            entry_data = {
                'content': entry.content,
                'created_at': entry.created_at,
                'ttl': entry.ttl,
                'token_count': entry.token_count,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed
            }
            self._redis_client.setex(
                key,
                entry.ttl,
                json.dumps(entry_data, separators=(',', ':'))
            )
        except Exception as e:
            logger.warning(f"Error writing to Redis cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached content by key"""
        # Try memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                entry.touch()
                self.stats.hits += 1
                self.stats.total_tokens_saved += entry.token_count
                return entry.content
            else:
                # Remove expired entry
                del self._memory_cache[key]
        
        # Try Redis cache
        entry = await self._get_from_redis(key)
        if entry:
            # Store in memory cache for faster future access
            self._memory_cache[key] = entry
            self.stats.hits += 1
            self.stats.total_tokens_saved += entry.token_count
            return entry.content
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, content: Any, ttl: Optional[int] = None, token_count: int = 0):
        """Set content in cache with TTL"""
        ttl = ttl or self.default_ttl
        entry = CacheEntry(
            content=content,
            created_at=time.time(),
            ttl=ttl,
            token_count=token_count
        )
        
        # Store in memory cache
        self._memory_cache[key] = entry
        
        # Store in Redis if available
        await self._set_to_redis(key, entry)
        
        # Cleanup memory cache if needed
        self._cleanup_memory_cache()
    
    def add_cache_control_to_system_prompt(self, system_prompt: str) -> str:
        """
        Add Anthropic's cache_control directive to system prompt for prompt caching
        
        Args:
            system_prompt: The system prompt text
            
        Returns:
            System prompt with cache control if caching is enabled
        """
        if not self.enable_anthropic_caching or not system_prompt.strip():
            return system_prompt
        
        # Add cache control instruction at the end of system prompt
        cached_prompt = system_prompt.rstrip()
        if not cached_prompt.endswith('\n'):
            cached_prompt += '\n'
        
        # Note: Anthropic's cache_control is applied via the messages API parameter,
        # not as text in the prompt. This method is for marking which prompts
        # should have cache control applied.
        return cached_prompt
    
    async def get_cached_system_prompt(self, system_prompt: str) -> Tuple[str, bool]:
        """
        Get system prompt from cache or mark for caching
        
        Args:
            system_prompt: The system prompt text
            
        Returns:
            Tuple of (prompt, is_from_cache)
        """
        if not system_prompt.strip():
            return system_prompt, False
        
        cache_key = self._generate_cache_key(system_prompt, self.SYSTEM_PROMPT_PREFIX)
        cached_prompt = await self.get(cache_key)
        
        if cached_prompt:
            self.stats.system_prompt_hits += 1
            logger.debug(f"System prompt cache hit: {cache_key[:8]}...")
            return cached_prompt, True
        
        # Process prompt for caching
        processed_prompt = self.add_cache_control_to_system_prompt(system_prompt)
        
        # Cache the processed prompt
        await self.set(
            cache_key, 
            processed_prompt, 
            ttl=self.default_ttl,  # Use default TTL, not system_prompt_ttl for consistency
            token_count=len(system_prompt.split()) * 1.3  # Rough token estimate
        )
        
        self.stats.system_prompt_misses += 1
        logger.debug(f"System prompt cached: {cache_key[:8]}...")
        return processed_prompt, False
    
    async def get_cached_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get message history from cache
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (messages, is_from_cache)
        """
        if not messages:
            return messages, False
        
        cache_key = self._generate_cache_key(messages, self.MESSAGE_PREFIX)
        cached_messages = await self.get(cache_key)
        
        if cached_messages:
            self.stats.message_cache_hits += 1
            logger.debug(f"Message cache hit: {cache_key[:8]}...")
            return cached_messages, True
        
        # Cache the messages
        total_tokens = sum(
            len(str(msg.get('content', '')).split()) * 1.3 
            for msg in messages
        )
        await self.set(
            cache_key,
            messages,
            ttl=self.default_ttl,
            token_count=int(total_tokens)
        )
        
        self.stats.message_cache_misses += 1
        logger.debug(f"Messages cached: {cache_key[:8]}...")
        return messages, False
    
    async def get_cached_tools(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get tool schemas from cache
        
        Args:
            tools: List of tool schema dictionaries
            
        Returns:
            Tuple of (tools, is_from_cache)
        """
        if not tools:
            return tools, False
        
        cache_key = self._generate_cache_key(tools, self.TOOL_SCHEMA_PREFIX)
        cached_tools = await self.get(cache_key)
        
        if cached_tools:
            logger.debug(f"Tool schema cache hit: {cache_key[:8]}...")
            return cached_tools, True
        
        # Cache the tool schemas
        tool_tokens = len(json.dumps(tools).split()) * 1.3
        await self.set(
            cache_key,
            tools,
            ttl=self.system_prompt_ttl,  # Tools change less frequently
            token_count=int(tool_tokens)
        )
        
        logger.debug(f"Tool schemas cached: {cache_key[:8]}...")
        return tools, False
    
    def prepare_claude_request_with_caching(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare Claude API request with optimal caching configuration
        
        Args:
            model: Claude model name
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            system_prompt: System prompt text
            messages: Message history
            tools: Tool schemas
            **kwargs: Additional API parameters
            
        Returns:
            Dictionary of API request parameters with caching optimizations
        """
        request_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs
        }
        
        # Add system prompt with caching if enabled
        if system_prompt and self.enable_anthropic_caching:
            # For Anthropic's prompt caching, we need to add cache_control
            # to the system parameter in the API call
            request_params["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        elif system_prompt:
            request_params["system"] = system_prompt
        
        # Add tools if provided
        if tools:
            request_params["tools"] = tools
        
        return request_params
    
    async def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache entries matching pattern or all if no pattern
        
        Args:
            pattern: Cache key pattern to match (optional)
        """
        if pattern:
            # Clear matching entries from memory
            keys_to_remove = [
                key for key in self._memory_cache.keys()
                if pattern in key
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            # Clear matching entries from Redis
            if self._redis_client:
                try:
                    matching_keys = self._redis_client.keys(f"*{pattern}*")
                    if matching_keys:
                        self._redis_client.delete(*matching_keys)
                except Exception as e:
                    logger.warning(f"Error clearing Redis cache: {e}")
        else:
            # Clear all
            self._memory_cache.clear()
            if self._redis_client:
                try:
                    self._redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Error flushing Redis cache: {e}")
        
        logger.info(f"Cache cleared{f' (pattern: {pattern})' if pattern else ''}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        stats_dict = self.stats.to_dict()
        stats_dict.update({
            "memory_cache_size": len(self._memory_cache),
            "redis_available": self._redis_client is not None,
            "anthropic_caching_enabled": self.enable_anthropic_caching,
            "max_memory_entries": self.max_memory_entries,
            "default_ttl": self.default_ttl,
            "system_prompt_ttl": self.system_prompt_ttl
        })
        return stats_dict
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")


def cache_claude_request(cache_instance: Optional[PromptCache] = None):
    """
    Decorator to add caching to Claude API requests
    
    Args:
        cache_instance: PromptCache instance to use (optional)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if cache_instance:
                # Extract and cache prompts/messages from kwargs
                if 'system' in kwargs:
                    system_prompt = kwargs['system']
                    if isinstance(system_prompt, str):
                        cached_prompt, _ = await cache_instance.get_cached_system_prompt(system_prompt)
                        kwargs['system'] = cached_prompt
                
                if 'messages' in kwargs:
                    messages = kwargs['messages']
                    cached_messages, _ = await cache_instance.get_cached_messages(messages)
                    kwargs['messages'] = cached_messages
                
                if 'tools' in kwargs:
                    tools = kwargs['tools']
                    cached_tools, _ = await cache_instance.get_cached_tools(tools)
                    kwargs['tools'] = cached_tools
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator