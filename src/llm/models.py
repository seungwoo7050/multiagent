"""
Model registry for LLM models and their capabilities.
"""

from typing import Any, Dict, List, Optional, Set, Union, cast
from functools import lru_cache

from src.config.settings import get_settings
from src.config.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Model information registry
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "token_limit": 128000,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "capabilities": ["general", "reasoning", "coding", "math", "vision"],
        "description": "Latest and most powerful OpenAI model with vision capabilities",
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "token_limit": 128000,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "capabilities": ["general", "reasoning", "coding", "math"],
        "description": "Powerful model with large context window",
    },
    "gpt-4": {
        "provider": "openai",
        "token_limit": 8192,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "capabilities": ["general", "reasoning", "coding", "math"],
        "description": "Powerful reasoning model",
    },
    "gpt-4-32k": {
        "provider": "openai",
        "token_limit": 32768,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "capabilities": ["general", "reasoning", "coding", "math"],
        "description": "Powerful reasoning model with extended context window",
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "token_limit": 16385,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "capabilities": ["general", "coding", "math"],
        "description": "Fast and cost-effective model",
    },
    
    # Anthropic models
    "claude-3-opus": {
        "provider": "anthropic",
        "token_limit": 200000,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_json_mode": False,
        "capabilities": ["general", "reasoning", "coding", "math", "vision"],
        "description": "Anthropic's most powerful model with vision capabilities",
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "token_limit": 180000,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_json_mode": False,
        "capabilities": ["general", "reasoning", "coding", "math", "vision"],
        "description": "Balance of intelligence and speed from Anthropic",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "token_limit": 150000,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_json_mode": False,
        "capabilities": ["general", "coding", "math", "vision"],
        "description": "Fast and efficient model from Anthropic",
    },
}

def register_model(
    model_name: str,
    provider: str,
    token_limit: int,
    capabilities: List[str],
    description: str,
    supports_streaming: bool = True,
    supports_function_calling: bool = False,
    supports_json_mode: bool = False,
    additional_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register a new model in the registry.
    
    Args:
        model_name: Name of the model
        provider: Provider name
        token_limit: Maximum token limit
        capabilities: List of model capabilities
        description: Short description of the model
        supports_streaming: Whether the model supports streaming
        supports_function_calling: Whether the model supports function calling
        supports_json_mode: Whether the model supports JSON mode
        additional_properties: Any additional model properties
    """
    global _MODEL_REGISTRY
    
    model_info = {
        "provider": provider,
        "token_limit": token_limit,
        "supports_streaming": supports_streaming,
        "supports_function_calling": supports_function_calling,
        "supports_json_mode": supports_json_mode,
        "capabilities": capabilities,
        "description": description,
    }
    
    # Add any additional properties
    if additional_properties:
        model_info.update(additional_properties)
    
    _MODEL_REGISTRY[model_name] = model_info
    logger.debug(f"Registered model {model_name} ({provider})")

@lru_cache(maxsize=128)
def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict[str, Any]: Model information or empty dict if not found
    """
    # Try exact match
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name].copy()
    
    # Try prefix match
    for registry_model, info in _MODEL_REGISTRY.items():
        if model_name.startswith(registry_model):
            return info.copy()
    
    # Try to determine provider from settings
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model_name)
    
    if provider:
        # Return generic info based on provider
        provider_defaults = {
            "openai": {
                "provider": "openai",
                "token_limit": 4096,  # Conservative default
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "capabilities": ["general"],
                "description": "Unrecognized OpenAI model",
            },
            "anthropic": {
                "provider": "anthropic",
                "token_limit": 100000,  # Conservative default
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "capabilities": ["general"],
                "description": "Unrecognized Anthropic model",
            }
        }
        
        if provider in provider_defaults:
            logger.warning(f"Using default info for unrecognized model {model_name} ({provider})")
            return provider_defaults[provider].copy()
    
    # Return empty dict if not found
    logger.warning(f"No information found for model {model_name}")
    return {}

def get_token_limit(model_name: str) -> int:
    """
    Get the token limit for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        int: Token limit or default conservative value
    """
    model_info = get_model_info(model_name)
    return model_info.get("token_limit", 4096)  # Conservative default

def model_supports_feature(model_name: str, feature: str) -> bool:
    """
    Check if a model supports a specific feature.
    
    Args:
        model_name: Name of the model
        feature: Feature name to check
        
    Returns:
        bool: True if the model supports the feature
    """
    model_info = get_model_info(model_name)
    return model_info.get(f"supports_{feature}", False)

def model_has_capability(model_name: str, capability: str) -> bool:
    """
    Check if a model has a specific capability.
    
    Args:
        model_name: Name of the model
        capability: Capability to check
        
    Returns:
        bool: True if the model has the capability
    """
    model_info = get_model_info(model_name)
    return capability in model_info.get("capabilities", [])

def list_available_models(
    provider: Optional[str] = None,
    min_token_limit: Optional[int] = None,
    required_capabilities: Optional[List[str]] = None,
    required_features: Optional[List[str]] = None,
) -> List[str]:
    """
    List available models with filtering options.
    
    Args:
        provider: Filter by provider
        min_token_limit: Minimum token limit
        required_capabilities: List of required capabilities
        required_features: List of required features
        
    Returns:
        List[str]: List of matching model names
    """
    result = []
    
    for model_name, info in _MODEL_REGISTRY.items():
        # Apply provider filter
        if provider and info["provider"] != provider:
            continue
            
        # Apply token limit filter
        if min_token_limit and info["token_limit"] < min_token_limit:
            continue
            
        # Apply capabilities filter
        if required_capabilities:
            model_capabilities = set(info["capabilities"])
            if not set(required_capabilities).issubset(model_capabilities):
                continue
                
        # Apply features filter
        if required_features:
            model_features = {
                feature for feature in ["streaming", "function_calling", "json_mode"]
                if info.get(f"supports_{feature}", False)
            }
            if not set(required_features).issubset(model_features):
                continue
                
        # Model passed all filters
        result.append(model_name)
    
    return sorted(result)

def get_provider_models(provider: str) -> List[str]:
    """
    Get all models for a specific provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List[str]: List of model names for the provider
    """
    return list_available_models(provider=provider)

def find_alternative_model(
    model_name: str,
    same_provider: bool = True,
) -> Optional[str]:
    """
    Find an alternative model if the requested one is unavailable.
    
    Args:
        model_name: Original model name
        same_provider: Whether to restrict to the same provider
        
    Returns:
        Optional[str]: Alternative model name or None if not found
    """
    # Get original model info
    original_info = get_model_info(model_name)
    
    if not original_info:
        return None
        
    provider = original_info["provider"]
    capabilities = original_info.get("capabilities", [])
    
    # Find models from the same provider with similar capabilities
    candidates = []
    
    for candidate_name, info in _MODEL_REGISTRY.items():
        # Skip the original model
        if candidate_name == model_name:
            continue
            
        # Check provider constraint
        if same_provider and info["provider"] != provider:
            continue
            
        # Check capabilities (at least half should match)
        candidate_capabilities = set(info.get("capabilities", []))
        original_capabilities = set(capabilities)
        
        if not original_capabilities:
            # If no capabilities specified, any model is fine
            pass
        elif len(original_capabilities.intersection(candidate_capabilities)) < len(original_capabilities) / 2:
            continue
            
        # Calculate a score based on similarity
        # Higher is better
        score = 0
        
        # Prefer same provider
        if info["provider"] == provider:
            score += 100
            
        # Prefer similar token limit
        token_limit_ratio = min(info["token_limit"], original_info["token_limit"]) / max(info["token_limit"], original_info["token_limit"])
        score += 50 * token_limit_ratio
        
        # Prefer similar capabilities
        if original_capabilities:
            capability_overlap = len(original_capabilities.intersection(candidate_capabilities)) / len(original_capabilities)
            score += 30 * capability_overlap
        
        # Store with score
        candidates.append((candidate_name, score))
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best alternative, or None if no candidates
    return candidates[0][0] if candidates else None

def initialize_models():
    """Initialize the model registry with settings."""
    # Ensure models from settings are registered
    for model in settings.ENABLED_MODELS_SET:
        if model not in _MODEL_REGISTRY:
            provider = settings.LLM_MODEL_PROVIDER_MAP.get(model)
            if provider:
                logger.warning(f"Model {model} not in registry, adding with defaults")
                register_model(
                    model_name=model,
                    provider=provider,
                    token_limit=4096,  # Conservative default
                    capabilities=["general"],
                    description=f"Model {model} from {provider}",
                )

# Initialize models when module is imported
# 명시적 호출필요
# initialize_models()