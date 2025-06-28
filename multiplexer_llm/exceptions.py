"""Custom exceptions for the multiplexer package."""

from typing import Optional


class MultiplexerError(Exception):
    """Base exception for all multiplexer errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelSelectionError(MultiplexerError):
    """Raised when no models are available for selection."""
    
    def __init__(self, message: str = "No models available for selection") -> None:
        super().__init__(message)


class APIError(MultiplexerError):
    """Base exception for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        endpoint: Optional[str] = None, 
        model_name: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.model_name = model_name


class ModelNotFoundError(APIError):
    """Raised when a model is not found (404)."""
    
    def __init__(
        self, 
        status_code: Optional[int] = None, 
        endpoint: Optional[str] = None, 
        model_name: Optional[str] = None,
        original_message: Optional[str] = None
    ) -> None:
        if original_message:
            message = f"Model '{model_name}' not found at endpoint {endpoint} (HTTP {status_code}): {original_message}"
        else:
            message = f"Model '{model_name}' not found at endpoint {endpoint} (HTTP {status_code})"
        super().__init__(message, status_code, endpoint, model_name)


class AuthenticationError(APIError):
    """Raised for authentication issues (401/403)."""
    
    def __init__(
        self, 
        status_code: Optional[int] = None, 
        endpoint: Optional[str] = None, 
        model_name: Optional[str] = None,
        original_message: Optional[str] = None
    ) -> None:
        if original_message:
            message = f"Authentication failed for model '{model_name}' at {endpoint} (HTTP {status_code}): {original_message}"
        else:
            message = f"Authentication failed for model '{model_name}' at {endpoint} (HTTP {status_code})"
        super().__init__(message, status_code, endpoint, model_name)


class RateLimitError(APIError):
    """Raised when rate limited with retry information."""
    
    def __init__(
        self, 
        status_code: Optional[int] = None, 
        endpoint: Optional[str] = None, 
        model_name: Optional[str] = None,
        retry_after: Optional[float] = None,
        original_message: Optional[str] = None
    ) -> None:
        if retry_after:
            message = f"Rate limit exceeded for model '{model_name}' at {endpoint}. Retry after {retry_after}s"
        elif original_message:
            message = f"Rate limit exceeded for model '{model_name}' at {endpoint}: {original_message}"
        else:
            message = f"Rate limit exceeded for model '{model_name}' at {endpoint} (HTTP {status_code})"
        super().__init__(message, status_code, endpoint, model_name)
        self.retry_after = retry_after


class ServiceUnavailableError(APIError):
    """Raised for 5xx server errors and connection issues."""
    
    def __init__(
        self, 
        status_code: Optional[int] = None, 
        endpoint: Optional[str] = None, 
        model_name: Optional[str] = None,
        original_message: Optional[str] = None
    ) -> None:
        if original_message:
            message = f"Service unavailable for model '{model_name}' at {endpoint} (HTTP {status_code}): {original_message}"
        else:
            message = f"Service unavailable for model '{model_name}' at {endpoint} (HTTP {status_code})"
        super().__init__(message, status_code, endpoint, model_name)


class AllModelsFailedError(MultiplexerError):
    """Raised when all models have failed and no more models are available to try."""
    
    def __init__(self, message: str = "All models have failed") -> None:
        super().__init__(message)
