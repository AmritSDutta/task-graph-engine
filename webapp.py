"""Custom FastAPI app for LangGraph server with additional REST endpoints.

This app provides custom REST endpoints that are automatically merged with
the LangGraph core API routes. LangGraph will automatically combine this app
with its own routes - no manual mounting needed.

Available endpoints:
- GET /api/health - Health check
- GET /api/models - List all models with capabilities and costs
- GET /api/models/{model_name} - Get specific model details
- GET /api/statistics - Get runtime statistics (model usage, costs)
- GET /api/config - Get current configuration settings

Authentication:
- Protected endpoints require API key via Bearer token when REQUIRE_AUTH=true
- Set API_KEY and REQUIRE_AUTH=true in environment variables
"""

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from task_agent.config import settings
from task_agent.llms.simple_llm_selector.models import (
    MODEL_CAPABILITIES,
    MODEL_COST,
    CODING_MODEL_PRIORITY,
    get_model_capabilities,
    get_model_cost,
)
from task_agent.utils.model_live_usage import get_model_usage_singleton

logger = logging.getLogger(__name__)

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Verify API key for protected endpoints.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        The API key if valid

    Raises:
        HTTPException: If authentication is required and token is invalid
    """
    if not settings.REQUIRE_AUTH:
        # Auth disabled - allow through
        return credentials.credentials if credentials else ""

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide Authorization: Bearer <API_KEY>",
        )

    if settings.API_KEY and credentials.credentials == settings.API_KEY:
        return credentials.credentials

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


# ============================================================================
# FastAPI App - LangGraph will automatically merge this with its routes
# ============================================================================

app = FastAPI(
    title="Task Graph Engine API",
    description="LangGraph-based task planning system with intelligent LLM selection",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Custom REST API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Task Graph Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "langgraph_api": "/runs",  # LangGraph endpoints are at root
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status and basic system information
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "auth_required": settings.REQUIRE_AUTH,
        "models_loaded": len(MODEL_CAPABILITIES),
    }


@app.get("/api/models")
async def list_models(
    api_key: Annotated[str, Depends(verify_api_key)],
):
    """List all available models with their capabilities and costs.

    Requires:
        API key authentication (if REQUIRE_AUTH=true)

    Returns:
        Dictionary with model names as keys and their details as values
    """
    models = {}
    for model_name in MODEL_CAPABILITIES.keys():
        models[model_name] = {
            "capabilities": sorted(get_model_capabilities(model_name)),
            "cost": get_model_cost(model_name),
            "is_coding_priority": model_name in CODING_MODEL_PRIORITY,
        }
    return {
        "count": len(models),
        "models": models,
    }


@app.get("/api/models/{model_name}")
async def get_model_details(
    model_name: str,
    api_key: Annotated[str, Depends(verify_api_key)],
):
    """Get detailed information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model details including capabilities and cost

    Raises:
        HTTPException: If model not found
    """
    if model_name not in MODEL_CAPABILITIES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    return {
        "name": model_name,
        "capabilities": sorted(get_model_capabilities(model_name)),
        "cost": get_model_cost(model_name),
        "is_coding_priority": model_name in CODING_MODEL_PRIORITY,
    }


@app.get("/api/statistics")
async def get_statistics(
    api_key: Annotated[str, Depends(verify_api_key)],
):
    """Get runtime statistics including model usage and cost information.

    Requires:
        API key authentication (if REQUIRE_AUTH=true)

    Returns:
        Dictionary with usage statistics and model information
    """
    model_usage = get_model_usage_singleton()

    # Get usage for all known models
    usage_data = {}
    for model_name in MODEL_CAPABILITIES.keys():
        usage_count = model_usage.get_model_usage(model_name)
        if usage_count > 0:
            base_cost = get_model_cost(model_name)
            # Calculate derived cost with exponential penalty
            import math

            factor = math.exp(settings.COST_SPREADING_FACTOR * usage_count)
            derived_cost = base_cost * factor
            usage_data[model_name] = {
                "usage_count": usage_count,
                "base_cost": base_cost,
                "derived_cost": round(derived_cost, 4),
                "penalty_factor": round(factor, 4),
            }

    return {
        "cost_spreading_factor": settings.COST_SPREADING_FACTOR,
        "total_models_with_usage": len(usage_data),
        "models": usage_data,
    }


@app.get("/api/config")
async def get_config(
    api_key: Annotated[str, Depends(verify_api_key)],
):
    """Get current configuration settings (non-sensitive).

    Requires:
        API key authentication (if REQUIRE_AUTH=true)

    Returns:
        Dictionary with configuration settings
    """
    return {
        "inference_model": settings.INFERENCE_MODEL,
        "moderation_api_check_req": settings.MODERATION_API_CHECK_REQ,
        "cost_spreading_factor": settings.COST_SPREADING_FACTOR,
        "model_cost_csv_path": settings.MODEL_COST_CSV_PATH,
        "model_capability_csv_path": settings.MODEL_CAPABILITY_CSV_PATH,
    }


# Log startup
logger.info("Custom FastAPI app loaded with endpoints at /api/*")
