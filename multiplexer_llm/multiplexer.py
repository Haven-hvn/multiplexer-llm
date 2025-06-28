"""Main Multiplexer class for load balancing across multiple LLM providers."""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Union

from .exceptions import (
    AllModelsFailedError,
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    ModelSelectionError,
    MultiplexerError,
    RateLimitError,
    ServiceUnavailableError,
)
from .types import (
    CompletionOptions,
    CompletionResult,
    ModelStats,
    OpenAICompatibleClient,
    WeightedModel,
)

# Set up logging
logger = logging.getLogger(__name__)


class ChatCompletions:
    """Chat completions interface for the multiplexer."""
    
    def __init__(self, multiplexer: "Multiplexer") -> None:
        self._multiplexer = multiplexer
    
    async def create(
        self,
        *,
        messages: Any,
        model: str = "placeholder",
        **kwargs: Any,
    ) -> CompletionResult:
        """Create a chat completion using the multiplexer."""
        return await self._multiplexer._create_completion(
            messages=messages,
            model=model,
            **kwargs,
        )


class Chat:
    """Chat interface for the multiplexer."""
    
    def __init__(self, multiplexer: "Multiplexer") -> None:
        self.completions = ChatCompletions(multiplexer)


class Multiplexer:
    """
    A multiplexer for Large Language Model APIs that combines quotas from multiple
    models and automatically uses fallback models when primary models are rate limited.
    """
    
    def __init__(self) -> None:
        self._weighted_models: List[WeightedModel] = []
        self._fallback_models: List[WeightedModel] = []
        self._model_timeouts: Dict[str, asyncio.Task[None]] = {}
        self.chat = Chat(self)
    
    def _select_from_pool(self, pool: List[WeightedModel]) -> Optional[WeightedModel]:
        """Select a model from a pool using weighted random selection."""
        if not pool:
            return None
            
        total_weight = sum(wm.weight for wm in pool)
        random_weight = random.random() * total_weight
        
        for model in pool:
            random_weight -= model.weight
            if random_weight <= 0:
                return model
        
        return pool[-1]
    
    def _select_weighted_model(self) -> WeightedModel:
        """Selects an active weighted model entry based on weight."""
        now = time.time()
        
        # Filter active primary models
        active_primary = [
            wm for wm in self._weighted_models
            if wm.disabled_until is None or wm.disabled_until < now
        ]
        
        # Try primary models first
        selected = self._select_from_pool(active_primary)
        if selected:
            return selected
        
        # If no active primary, try fallbacks
        active_fallbacks = [
            wm for wm in self._fallback_models
            if wm.disabled_until is None or wm.disabled_until < now
        ]
        
        selected = self._select_from_pool(active_fallbacks)
        if selected:
            return selected
        
        # Check if there are models but all disabled
        if self._weighted_models or self._fallback_models:
            raise ModelSelectionError("All models are temporarily rate limited.")
        raise ModelSelectionError("No models available in the multiplexer.")
    
    async def _disable_model_temporarily(
        self, model_name: str, duration_ms: float
    ) -> None:
        """Disables a model temporarily."""
        # Check in primary models first
        model_index = -1
        model_array = self._weighted_models
        
        for i, wm in enumerate(self._weighted_models):
            if wm.model_name == model_name:
                model_index = i
                break
        
        # If not found in primary models, check in fallback models
        if model_index == -1:
            for i, wm in enumerate(self._fallback_models):
                if wm.model_name == model_name:
                    model_index = i
                    model_array = self._fallback_models
                    break
        
        # If model not found in either array, return
        if model_index == -1:
            return
        
        model = model_array[model_index]
        model.disabled_until = time.time() + (duration_ms / 1000.0)
        
        # Cancel existing timeout for this model if any
        existing_task = self._model_timeouts.get(model_name)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            try:
                await existing_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling

        # Set a new timeout to re-enable the model
        async def re_enable_model() -> None:
            try:
                await asyncio.sleep(duration_ms / 1000.0)
                model.disabled_until = None
                logger.info(f"Model {model_name} re-enabled after temporary disable.")
            except asyncio.CancelledError:
                logger.debug(f"Re-enable task for {model_name} was cancelled")
                raise
            finally:
                # Clean up the task reference
                self._model_timeouts.pop(model_name, None)

        task = asyncio.create_task(re_enable_model())
        self._model_timeouts[model_name] = task
        
        logger.warning(
            f"Model {model_name} temporarily disabled for "
            f"{duration_ms / 1000.0}s."
        )

    async def _create_completion(
        self,
        *,
        messages: Any,
        model: str = "placeholder",
        **kwargs: Any,
    ) -> CompletionResult:
        """Create a chat completion with automatic failover."""
        last_error: Optional[Exception] = None
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Attempt to select an available model
                selected = self._select_weighted_model()
            except ModelSelectionError as selection_error:
                # If model selection fails (e.g., all rate limited or no models configured)
                if last_error:
                    # Map the last error to our custom exceptions and raise it
                    # We don't have model info here since selection failed, so pass None
                    raise self._map_error_to_custom_exception(last_error, None, None)
                # Otherwise, re-throw the selection error (no models available/configured)
                raise selection_error

            # Prepare parameters with the selected model name
            final_params = {
                "messages": messages,
                "model": selected.model_name,  # Use the selected model's name
                **kwargs,
            }

            try:
                # Attempt the API call
                result = await selected.model.chat.completions.create(**final_params)

                # For mock tests, result might be a plain dict, not an object
                choices = getattr(result, 'choices', None) or result.get('choices')
                
                # Check if result is None (shouldn't happen but let's be safe)
                if result is None:
                    selected.fail_fast_count += 1
                    raise RuntimeError(f"Client for model {selected.model_name} returned None")

                # Validate that the response has the expected structure
                if choices is None:
                    selected.fail_fast_count += 1
                    error_msg = f"Invalid response from model {selected.model_name}: missing or null 'choices' field"
                    if hasattr(result, 'error'):
                        error_msg += f". Error: {result.error}"
                    elif 'error' in result:
                        error_msg += f". Error: {result['error']}"
                    raise RuntimeError(error_msg)

                selected.success_count += 1  # Increment success count
                
                # Create a new result object with the correct model name
                # We need to handle this carefully since Pydantic models are immutable
                try:
                    # Try to create a copy with the updated model name
                    if hasattr(result, 'model_copy'):
                        # Pydantic v2 style
                        updated_result = result.model_copy(update={'model': selected.model_name})
                    elif hasattr(result, 'copy'):
                        # Pydantic v1 style
                        updated_result = result.copy(update={'model': selected.model_name})
                    else:
                        # Fallback: try direct assignment (might work for some objects)
                        updated_result = result
                        updated_result.model = selected.model_name
                    return updated_result
                except Exception as copy_error:
                    # If we can't update the model name, log it but still return the result
                    logger.warning(f"Could not update model name for {selected.model_name}: {copy_error}")
                    return result
            except Exception as error:
                # Check if it's a rate limit error (429 or 529)
                if self._is_rate_limit_error(error):
                    logger.warning(
                        f"Model {selected.model_name} hit rate limit. Trying next model."
                    )
                    selected.rate_limit_count += 1  # Increment rate limit count
                    await self._disable_model_temporarily(
                        selected.model_name, 60 * 1000
                    )  # Disable for 1 minute
                    last_error = error  # Store the 429 error
                    continue  # Continue the loop to try another model
                elif self._is_persistent_error(error):
                    # Handle persistent errors (connection issues, service unavailable)
                    logger.warning(
                        f"Model {selected.model_name} failed with persistent error: {error}. "
                        "Disabling temporarily and trying next model."
                    )
                    selected.fail_fast_count += 1  # Increment fail-fast count
                    await self._disable_model_temporarily(
                        selected.model_name, 15 * 1000
                    )  # Disable for 15 seconds
                    last_error = error
                    continue  # Continue the loop to try another model
                else:
                    selected.fail_fast_count += 1  # Increment fail-fast count
                    last_error = error
                    logger.warning(f"Model {selected.model_name} failed with non-rate-limit error: {error}")
                    # For non-rate-limit errors, we stop retrying and propagate the error
                    break

        # If we've exhausted all retries, raise the last error or a generic error
        if last_error:
            if isinstance(last_error, RateLimitError):
                raise RateLimitError("Rate limit exceeded") from last_error
            raise self._map_error_to_custom_exception(last_error, selected.model_name, selected.base_url)
        raise ModelSelectionError("All models failed after maximum retries")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error (429 or 529)."""
        # Check for OpenAI-style errors
        if hasattr(error, "status_code"):
            return error.status_code in (429, 529)

        # Check for requests-style errors
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            return error.response.status_code in (429, 529)

        # Check error class name for rate limit indicators (for mock tests)
        error_class_name = error.__class__.__name__.lower()
        if "ratelimit" in error_class_name:
            return True

        # Check error message for rate limit indicators
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "429",
            "529",
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _is_persistent_error(self, error: Exception) -> bool:
        """Check if an error is a persistent connection/service error."""
        # Check for OpenAI-style errors with status codes
        if hasattr(error, "status_code"):
            # Consider 5xx errors and connection-related errors as persistent
            return error.status_code >= 500 or error.status_code in (408, 499)

        # Check for requests-style errors
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            status_code = error.response.status_code
            return status_code >= 500 or status_code in (408, 499)
        
        # Check for connection-related error types
        error_type = type(error).__name__
        if error_type in ["APIConnectionError", "ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout"]:
            return True
            
        # Check error message for connection-related keywords
        error_str = str(error).lower()
        persistent_indicators = [
            "connection", "connect", "timeout", "unavailable", "refused", 
            "network", "unreachable", "host", "dns", "resolve"
        ]
        return any(indicator in error_str for indicator in persistent_indicators)

    def _map_error_to_custom_exception(
        self, 
        error: Exception, 
        model_name: Optional[str], 
        base_url: Optional[str]
    ) -> MultiplexerError:
        """Map an underlying error to a custom multiplexer exception."""
        # Import OpenAI exceptions for type checking
        try:
            import openai
        except ImportError:
            # If OpenAI is not available, fall back to generic error mapping
            return MultiplexerError(f"API error: {str(error)}")

        # Extract status code from error
        status_code = getattr(error, "status_code", None)
        original_message = str(error)

        # Map specific OpenAI exceptions
        if isinstance(error, openai.NotFoundError):
            return ModelNotFoundError(
                status_code=status_code,
                endpoint=base_url,
                model_name=model_name,
                original_message=original_message
            )
        elif isinstance(error, openai.AuthenticationError):
            return AuthenticationError(
                status_code=status_code,
                endpoint=base_url,
                model_name=model_name,
                original_message=original_message
            )
        elif isinstance(error, openai.RateLimitError) or self._is_rate_limit_error(error):
            retry_after = getattr(error, "retry_after", None)
            # Simplify the error message when model_name is None
            if model_name is None:
                return RateLimitError("Rate limit exceeded")
            return RateLimitError(
                status_code=status_code,
                endpoint=base_url,
                model_name=model_name,
                retry_after=retry_after,
                original_message=original_message
            )
        elif isinstance(error, (openai.APIConnectionError, openai.InternalServerError)):
            return ServiceUnavailableError(
                status_code=status_code,
                endpoint=base_url,
                model_name=model_name,
                original_message=original_message
            )
        elif isinstance(error, openai.APIError):
            # Generic API error - check status code
            if status_code == 404:
                return ModelNotFoundError(
                    status_code=status_code,
                    endpoint=base_url,
                    model_name=model_name,
                    original_message=original_message
                )
            elif status_code in (401, 403):
                return AuthenticationError(
                    status_code=status_code,
                    endpoint=base_url,
                    model_name=model_name,
                    original_message=original_message
                )
            elif status_code and 500 <= status_code < 600:
                return ServiceUnavailableError(
                    status_code=status_code,
                    endpoint=base_url,
                    model_name=model_name,
                    original_message=original_message
                )
        
        # For any other error, wrap in generic MultiplexerError
        return MultiplexerError(f"Unexpected error for model {model_name}: {original_message}")

    def add_model(
        self,
        model: OpenAICompatibleClient,
        weight: int,
        model_name: str,
        base_url: Optional[str] = None,
    ) -> None:
        """Add a primary model to the multiplexer."""
        if not isinstance(weight, int) or weight <= 0:
            raise ValueError("Weight must be a positive integer.")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        # Check for duplicate model names
        all_models = self._weighted_models + self._fallback_models
        if any(wm.model_name == model_name for wm in all_models):
            logger.warning(
                f"Attempted to add a model with the same name '{model_name}' "
                f"multiple times. Skipping."
            )
            return

        # Add model with disabled_until initialized to None and stats to 0
        weighted_model = WeightedModel(model, weight, model_name, base_url)
        self._weighted_models.append(weighted_model)

    def add_fallback_model(
        self,
        model: OpenAICompatibleClient,
        weight: int,
        model_name: str,
        base_url: Optional[str] = None,
    ) -> None:
        """Add a fallback model to the multiplexer."""
        if not isinstance(weight, int) or weight <= 0:
            raise ValueError("Weight must be a positive integer.")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        # Check for duplicate model names
        all_models = self._weighted_models + self._fallback_models
        if any(wm.model_name == model_name for wm in all_models):
            logger.warning(
                f"Attempted to add a model with the same name '{model_name}' "
                f"multiple times. Skipping."
            )
            return

        # Add fallback model with disabled_until initialized to None and stats to 0
        weighted_model = WeightedModel(model, weight, model_name, base_url)
        self._fallback_models.append(weighted_model)

    def reset(self) -> None:
        """Reset the multiplexer, clearing all models and pending timeouts."""
        # Cancel all pending timeout tasks
        for task in self._model_timeouts.values():
            if not task.done():
                task.cancel()

        self._model_timeouts.clear()

        # Reset model lists
        self._weighted_models = []
        self._fallback_models = []

    async def async_reset(self) -> None:
        """Async version of reset that properly waits for task cancellation."""
        # Create a copy of tasks to avoid modification during iteration
        tasks_to_cancel = list(self._model_timeouts.values())
        
        # Cancel all tasks
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation to complete with proper handling
        if tasks_to_cancel:
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    logger.debug("Task was properly cancelled")
                elif isinstance(result, Exception):
                    logger.warning(f"Task raised exception during cancellation: {result}")

        # Clear data structures
        self._model_timeouts.clear()
        self._weighted_models = []
        self._fallback_models = []
        
        logger.info("Multiplexer has been fully reset")

    def get_stats(self) -> ModelStats:
        """Get usage statistics for all models."""
        stats: ModelStats = {}
        all_models = self._weighted_models + self._fallback_models

        for wm in all_models:
            stats[wm.model_name] = {
                "success": wm.success_count,
                "rateLimited": wm.rate_limit_count,
                "failed": wm.fail_fast_count,
            }

        return stats

    async def __aenter__(self) -> "Multiplexer":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - properly cleanup resources."""
        await self.async_reset()
