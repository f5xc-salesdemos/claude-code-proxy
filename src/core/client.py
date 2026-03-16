"""Async OpenAI / Azure OpenAI client with cancellation and retry support."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from src.core.config import config

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async OpenAI client with cancellation support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 90,
        api_version: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}

        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0",
        }

        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}

        # Detect if using Azure and instantiate the appropriate client
        self.client: Union[AsyncAzureOpenAI, AsyncOpenAI]
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers,
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Shared error handling
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _map_openai_errors(
        self, request_id: Optional[str] = None
    ) -> AsyncGenerator[None, None]:
        """Map OpenAI SDK exceptions to FastAPI HTTPExceptions.

        Also cleans up active request tracking on exit.
        """
        try:
            yield
        except AuthenticationError as e:
            logger.error("Upstream AuthenticationError: %s", e)
            raise HTTPException(
                status_code=401, detail=self.classify_openai_error(str(e))
            ) from e
        except RateLimitError as e:
            logger.error("Upstream RateLimitError: %s", e)
            raise HTTPException(
                status_code=429, detail=self.classify_openai_error(str(e))
            ) from e
        except BadRequestError as e:
            logger.error("Upstream BadRequestError: %s", e)
            raise HTTPException(
                status_code=400, detail=self.classify_openai_error(str(e))
            ) from e
        except APIError as e:
            logger.error(
                "Upstream APIError (%s): %s", getattr(e, "status_code", "unknown"), e
            )
            status_code = getattr(e, "status_code", 500)
            raise HTTPException(
                status_code=status_code, detail=self.classify_openai_error(str(e))
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Unexpected upstream error: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}"
            ) from e
        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    # ------------------------------------------------------------------
    # Chat completion (non-streaming)
    # ------------------------------------------------------------------

    async def create_chat_completion(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        async with self._map_openai_errors(request_id):
            completion_task = asyncio.create_task(
                self.client.chat.completions.create(**request)
            )

            if request_id:
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(
                        status_code=499, detail="Request cancelled by client"
                    )

                completion = await completion_task
            else:
                completion = await completion_task

            result: Dict[str, Any] = completion.model_dump()
            return result

    # ------------------------------------------------------------------
    # Chat completion (streaming)
    # ------------------------------------------------------------------

    async def create_chat_completion_stream(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True

            # Retry for rate limits on connection
            max_retries = config.max_retries
            last_error: Optional[RateLimitError] = None
            streaming_completion = None
            for attempt in range(max_retries + 1):
                try:
                    streaming_completion = await self.client.chat.completions.create(
                        **request
                    )
                    last_error = None
                    break
                except RateLimitError as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = 2 ** (attempt + 1)
                        logger.warning(
                            "Upstream rate limited on connect, retry %d/%d after %ds: %s",
                            attempt + 1,
                            max_retries,
                            backoff,
                            e,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(
                            "Upstream RateLimitError after %d retries: %s",
                            max_retries,
                            e,
                        )

            if last_error is not None:
                raise last_error

            async for chunk in streaming_completion:  # type: ignore[union-attr]
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(
                            status_code=499, detail="Request cancelled by client"
                        )

                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            yield "data: [DONE]"

        except AuthenticationError as e:
            logger.error("Upstream AuthenticationError: %s", e)
            raise HTTPException(
                status_code=401, detail=self.classify_openai_error(str(e))
            ) from e
        except RateLimitError as e:
            logger.error("Upstream RateLimitError: %s", e)
            raise HTTPException(
                status_code=429, detail=self.classify_openai_error(str(e))
            ) from e
        except BadRequestError as e:
            logger.error("Upstream BadRequestError: %s", e)
            raise HTTPException(
                status_code=400, detail=self.classify_openai_error(str(e))
            ) from e
        except APIError as e:
            logger.error(
                "Upstream APIError (%s): %s", getattr(e, "status_code", "unknown"), e
            )
            status_code = getattr(e, "status_code", 500)
            raise HTTPException(
                status_code=status_code, detail=self.classify_openai_error(str(e))
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Unexpected upstream error: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}"
            ) from e

        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        logger.debug("Classifying upstream error: %s", error_detail)
        error_str = str(error_detail).lower()

        if (
            "unsupported_country_region_territory" in error_str
            or "country, region, or territory not supported" in error_str
        ):
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."

        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."

        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."

        if "model" in error_str and (
            "not found" in error_str or "does not exist" in error_str
        ):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."

        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        return str(error_detail)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
