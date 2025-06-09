"""Base agent with common functionality for all optimized agents."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
import asyncio
import aiohttp
import backoff
import json
import hashlib
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class CacheEntry:
    data: Any
    expires_at: datetime
    etag: Optional[str] = None

class BaseAgent(ABC, Generic[T]):
    """Base class for all agents with common functionality."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", **kwargs):
        """Initialize the base agent.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.cache: Dict[str, CacheEntry] = {}
        self.rate_limits: Dict[str, float] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Initialize rate limiting
        self.rate_limits = {
            'default': 10,  # 10 requests per second
            'gemini': 2,    # 2 requests per second for Gemini
        }
        
        # Request tracking
        self._last_request: Dict[str, float] = {}
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def __del__(self):
        """Ensure session is closed when object is destroyed."""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            if self._session._loop.is_running():
                asyncio.create_task(self.close())
    
    @asynccontextmanager
    async def rate_limiter(self, endpoint: str = 'default'):
        """Context manager for rate limiting."""
        last_request = self._last_request.get(endpoint, 0)
        current_time = asyncio.get_event_loop().time()
        
        # Calculate time to wait
        min_interval = 1.0 / self.rate_limits.get(endpoint, self.rate_limits['default'])
        time_since_last = current_time - last_request
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self._last_request[endpoint] = asyncio.get_event_loop().time()
        yield
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        key_dict = {
            'args': args,
            'kwargs': {k: v for k, v in kwargs.items() if k != 'self'}
        }
        return hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
    
    async def get_cached(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get a value from cache if it exists and hasn't expired."""
        if key in self.cache:
            entry = self.cache[key]
            if entry.expires_at > datetime.now():
                return entry.data
            del self.cache[key]
        return None
    
    async def set_cached(self, key: str, value: Any, ttl: int = 3600):
        """Set a value in the cache with a TTL."""
        self.cache[key] = CacheEntry(
            data=value,
            expires_at=datetime.now() + timedelta(seconds=ttl)
        )
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
    async def cached_call(self, func, *args, ttl: int = 3600, **kwargs):
        """Call a function with caching."""
        cache_key = self.cache_key(func.__name__, *args, **kwargs)
        
        # Try to get from cache
        cached = await self.get_cached(cache_key, ttl)
        if cached is not None:
            logger.debug(f"Cache hit for {func.__name__}")
            return cached
        
        # Call the function
        logger.debug(f"Cache miss for {func.__name__}, executing...")
        result = await func(*args, **kwargs)
        
        # Cache the result
        await self.set_cached(cache_key, result, ttl)
        return result
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> T:
        """Process the input and return the result."""
        pass
    
    async def __call__(self, *args, **kwargs) -> T:
        """Make the agent callable."""
        return await self.process(*args, **kwargs)
