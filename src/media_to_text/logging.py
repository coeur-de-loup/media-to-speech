"""Structured logging configuration with Axiom integration and request tracing."""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from axiom import Client as AxiomClient

# Context variables for request tracing
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Sensitive data patterns to filter from logs
SENSITIVE_PATTERNS = [
    "password", "secret", "token", "key", "auth", "authorization",
    "credential", "api_key", "openai_api_key", "axiom_token"
]


class AxiomProcessor:
    """Axiom log processor for sending logs to Axiom dataset."""
    
    def __init__(self, token: Optional[str] = None, dataset: str = "media-to-text-logs"):
        self.token = token or os.getenv("AXIOM_TOKEN")
        self.dataset = dataset
        self.client = None
        self.enabled = bool(self.token)
        
        if self.enabled:
            try:
                self.client = AxiomClient(token=self.token)
                structlog.get_logger().info("Axiom logging enabled", dataset=dataset)
            except Exception as e:
                structlog.get_logger().warning("Failed to initialize Axiom client", error=str(e))
                self.enabled = False
    
    def __call__(self, logger, method_name, event_dict):
        """Process log entry for Axiom."""
        if not self.enabled or not self.client:
            return event_dict
        
        try:
            # Prepare log entry for Axiom
            axiom_entry = {
                "_time": datetime.utcnow().isoformat() + "Z",
                "level": event_dict.get("level", "info").upper(),
                "event": event_dict.get("event", ""),
                "service": "media-to-text",
                "version": "0.1.0",
                **event_dict
            }
            
            # Add trace context if available
            trace_id = trace_id_var.get("")
            if trace_id:
                axiom_entry["trace_id"] = trace_id
            
            request_id = request_id_var.get("")
            if request_id:
                axiom_entry["request_id"] = request_id
            
            user_id = user_id_var.get("")
            if user_id:
                axiom_entry["user_id"] = user_id
            
            # Send to Axiom asynchronously (fire and forget)
            asyncio.create_task(self._send_to_axiom(axiom_entry))
            
        except Exception as e:
            # Don't let Axiom errors break the application
            structlog.get_logger().warning("Failed to send log to Axiom", error=str(e))
        
        return event_dict
    
    async def _send_to_axiom(self, entry: Dict[str, Any]) -> None:
        """Send log entry to Axiom dataset."""
        try:
            if self.client:
                await asyncio.to_thread(self.client.ingest, self.dataset, [entry])
        except Exception:
            # Silently fail - don't spam logs with Axiom errors
            pass


def filter_sensitive_data(logger, method_name, event_dict):
    """Filter sensitive data from log entries."""
    def _filter_value(value):
        if isinstance(value, str):
            # Check if any sensitive pattern is in the key or value
            value_lower = value.lower()
            for pattern in SENSITIVE_PATTERNS:
                if pattern in value_lower and len(value) > 10:  # Only filter longer strings
                    return f"***{value[-4:]}"  # Show last 4 characters
        elif isinstance(value, dict):
            return {k: _filter_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_filter_value(item) for item in value]
        return value
    
    # Filter the entire event dict
    filtered_dict = {}
    for key, value in event_dict.items():
        key_lower = key.lower()
        
        # Check if key itself is sensitive
        if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
            if isinstance(value, str) and len(value) > 4:
                filtered_dict[key] = f"***{value[-4:]}"
            else:
                filtered_dict[key] = "***"
        else:
            filtered_dict[key] = _filter_value(value)
    
    return filtered_dict


def add_trace_context(logger, method_name, event_dict):
    """Add trace context to log entries."""
    # Add trace information if available
    trace_id = trace_id_var.get("")
    if trace_id:
        event_dict["trace_id"] = trace_id
    
    request_id = request_id_var.get("")
    if request_id:
        event_dict["request_id"] = request_id
    
    user_id = user_id_var.get("")
    if user_id:
        event_dict["user_id"] = user_id
    
    # Add service metadata
    event_dict["service"] = "media-to-text"
    event_dict["version"] = "0.1.0"
    
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    enable_axiom: bool = True,
    axiom_token: Optional[str] = None,
    axiom_dataset: str = "media-to-text-logs"
) -> None:
    """
    Configure structured logging with Axiom integration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_axiom: Whether to enable Axiom logging
        axiom_token: Axiom API token (defaults to AXIOM_TOKEN env var)
        axiom_dataset: Axiom dataset name
    """
    # Configure processors
    processors: List[Any] = [
        filter_sensitive_data,
        add_trace_context,
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Add Axiom processor if enabled
    if enable_axiom:
        axiom_processor = AxiomProcessor(token=axiom_token, dataset=axiom_dataset)
        processors.append(axiom_processor)
    
    # Add console formatting
    if sys.stderr.isatty():
        # Development mode - colored output
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # Production mode - JSON output
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str = "") -> structlog.BoundLogger:
    """Get a configured structlog logger."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """Set trace ID in context and return it."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in context and return it."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def set_user_id(user_id: str) -> None:
    """Set user ID in context."""
    user_id_var.set(user_id)


def clear_trace_context() -> None:
    """Clear all trace context variables."""
    trace_id_var.set("")
    request_id_var.set("")
    user_id_var.set("")


class LoggerMixin:
    """Mixin class to add structured logging to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger bound to this class."""
        class_name = self.__class__.__name__
        return get_logger(class_name)


# Convenience functions for common log levels
def debug(event: str, **kwargs) -> None:
    """Log debug message with context."""
    get_logger().debug(event, **kwargs)


def info(event: str, **kwargs) -> None:
    """Log info message with context."""
    get_logger().info(event, **kwargs)


def warning(event: str, **kwargs) -> None:
    """Log warning message with context."""
    get_logger().warning(event, **kwargs)


def error(event: str, **kwargs) -> None:
    """Log error message with context."""
    get_logger().error(event, **kwargs)


def critical(event: str, **kwargs) -> None:
    """Log critical message with context."""
    get_logger().critical(event, **kwargs)


# FastAPI middleware for request tracing
class LoggingMiddleware:
    """FastAPI middleware for request tracing and logging."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Set up trace context for this request
            trace_id = set_trace_id()
            request_id = set_request_id()
            
            # Log request start
            logger = get_logger("http")
            logger.info(
                "HTTP request started",
                method=scope.get("method"),
                path=scope.get("path"),
                trace_id=trace_id,
                request_id=request_id
            )
            
            # Process request
            await self.app(scope, receive, send)
            
            # Clear context after request
            clear_trace_context()
        else:
            await self.app(scope, receive, send)