# -*- coding: utf-8 -*-
"""
Configuration for Semantic Understanding System
================================================

This module provides configuration settings for the semantic understanding features,
including fuzzy matching thresholds, LLM settings, and feature flags for
incremental rollout.
"""

from dataclasses import dataclass
from typing import Optional
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

@dataclass
class SemanticConfig:
    """Configuration for semantic understanding features."""

    # ===== Feature Flags =====
    enable_semantic_classifier: bool = True
    """Enable the semantic question classifier (vs. regex-only)"""

    enable_fuzzy_matching: bool = True
    """Enable fuzzy matching for typo tolerance"""

    enable_llm_fallback: bool = True
    """Enable LLM fallback for complex queries (can disable for initial rollout)"""

    enable_regex_fallback: bool = True
    """Enable regex fallback as ultimate safety net"""

    # ===== Fuzzy Matching Thresholds =====
    fuzzy_high_confidence: float = 85.0
    """Threshold for high confidence matches (use directly without LLM)"""

    fuzzy_medium_confidence: float = 70.0
    """Threshold for medium confidence matches (use with logging)"""

    fuzzy_low_confidence: float = 50.0
    """Threshold for low confidence matches (trigger LLM fallback)"""

    fuzzy_top_k: int = 5
    """Number of top matches to return from fuzzy search"""

    # ===== LLM Settings (Groq API) =====
    llm_provider: str = "groq"
    """LLM provider: 'groq', 'openai', or 'anthropic'"""

    llm_model: str = "groq/llama-3.3-70b-versatile"
    """Model name for Groq API. Options: groq/llama-3.3-70b-versatile, groq/mixtral-8x7b-32768"""

    llm_max_tokens: int = 2000
    """Maximum tokens for LLM response"""

    llm_temperature: float = 0.0
    """Temperature for LLM (0.0 = deterministic, good for classification)"""

    groq_api_key: Optional[str] = None
    """Groq API key (will be read from environment if not provided)"""

    # ===== Performance Settings =====
    cache_fuzzy_results: bool = True
    """Enable caching of fuzzy match results"""

    cache_llm_results: bool = True
    """Enable caching of LLM responses"""

    cache_ttl_seconds: int = 3600
    """Cache time-to-live in seconds (1 hour default)"""

    # ===== Logging Settings =====
    log_classification_decisions: bool = True
    """Log which classification method was used"""

    log_fuzzy_scores: bool = False
    """Log detailed fuzzy matching scores (verbose)"""

    log_llm_prompts: bool = False
    """Log LLM prompts and responses (verbose, use for debugging)"""

    log_timing: bool = True
    """Log timing information for performance monitoring"""

    def __post_init__(self):
        """Set API key from environment if not provided."""
        if self.groq_api_key is None:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError if configuration is invalid
        """
        # Check thresholds are in valid range
        if not (0 <= self.fuzzy_low_confidence <= self.fuzzy_medium_confidence <= self.fuzzy_high_confidence <= 100):
            raise ValueError(
                "Fuzzy confidence thresholds must be in range [0, 100] and "
                "low <= medium <= high"
            )

        # Check LLM API key if LLM is enabled
        if self.enable_llm_fallback and not self.groq_api_key:
            raise ValueError(
                "LLM fallback is enabled but GROQ_API_KEY is not set. "
                "Set the environment variable or disable LLM fallback."
            )

        # Check LLM provider
        if self.llm_provider not in ["groq", "openai", "anthropic"]:
            raise ValueError(
                f"Invalid LLM provider: {self.llm_provider}. "
                "Must be 'groq', 'openai', or 'anthropic'"
            )

        return True


# ================================================================================
# ========================= PREDEFINED CONFIGURATIONS ============================
# ================================================================================

# Default configuration (recommended for production)
DEFAULT_CONFIG = SemanticConfig()

# Conservative configuration (for initial rollout - no LLM)
CONSERVATIVE_CONFIG = SemanticConfig(
    enable_llm_fallback=False,  # No LLM initially to avoid API costs
    fuzzy_high_confidence=90.0,  # Stricter thresholds
    fuzzy_medium_confidence=80.0,
    fuzzy_low_confidence=60.0,
    log_classification_decisions=True,
    log_fuzzy_scores=True  # Verbose logging for validation
)

# Aggressive configuration (for full features after validation)
AGGRESSIVE_CONFIG = SemanticConfig(
    enable_llm_fallback=True,
    fuzzy_high_confidence=80.0,  # More permissive thresholds
    fuzzy_medium_confidence=65.0,
    fuzzy_low_confidence=45.0,
    log_classification_decisions=True,
    log_fuzzy_scores=True,
    log_llm_prompts=True  # Full logging for analysis
)

# Development configuration (for testing)
DEV_CONFIG = SemanticConfig(
    enable_llm_fallback=True,
    fuzzy_high_confidence=75.0,
    fuzzy_medium_confidence=60.0,
    fuzzy_low_confidence=40.0,
    cache_fuzzy_results=False,  # Disable caching for testing
    cache_llm_results=False,
    log_classification_decisions=True,
    log_fuzzy_scores=True,
    log_llm_prompts=True,
    log_timing=True
)


# ================================================================================
# ========================= CONFIGURATION HELPER =================================
# ================================================================================

def get_config(profile: str = "default") -> SemanticConfig:
    """
    Get configuration by profile name.

    Args:
        profile: Configuration profile name
            - "default": Recommended production settings
            - "conservative": No LLM, strict thresholds (initial rollout)
            - "aggressive": Full features, permissive thresholds
            - "dev": Development/testing settings

    Returns:
        SemanticConfig instance

    Raises:
        ValueError if profile name is invalid
    """
    profiles = {
        "default": DEFAULT_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "dev": DEV_CONFIG
    }

    if profile not in profiles:
        raise ValueError(
            f"Invalid profile: {profile}. "
            f"Must be one of {list(profiles.keys())}"
        )

    config = profiles[profile]
    config.validate()
    return config


# ================================================================================
# ========================= GROQ MODEL INFO ======================================
# ================================================================================

GROQ_MODELS = {
    "groq/compound-mini": {
        "name": "Groq Compound Mini",
        "description": "Fast reasoning model optimized for inference speed",
        "context_window": 8192,
        "recommended": True
    },
    "groq/compound": {
        "name": "Groq Compound",
        "description": "Full reasoning model with extended capabilities",
        "context_window": 8192
    },
    "llama-3.3-70b-versatile": {
        "name": "Llama 3.3 70B",
        "description": "Versatile large language model for complex tasks",
        "context_window": 128000
    },
    "qwen-3-32b": {
        "name": "Qwen 3 32B (Legacy)",
        "description": "Fast, efficient, good for classification and code generation",
        "context_window": 32768,
        "speed": "very_fast",
        "recommended": True
    },
    "llama-3-70b": {
        "name": "Llama 3 70B",
        "description": "Powerful, good reasoning, slower than Qwen",
        "context_window": 8192,
        "speed": "fast",
        "recommended": False
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B",
        "description": "Good balance of speed and capability",
        "context_window": 32768,
        "speed": "fast",
        "recommended": False
    },
    "llama-3.3-70b-versatile": {
        "name": "Llama 3.3 70B Versatile",
        "description": "Latest Llama model, excellent reasoning",
        "context_window": 128000,
        "speed": "medium",
        "recommended": True
    }
}


def get_recommended_model() -> str:
    """Get recommended Groq model for semantic understanding."""
    return "groq/llama-3.3-70b-versatile"  # Excellent reasoning and biological analysis


# ================================================================================
# ========================= EXAMPLE USAGE ========================================
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SEMANTIC UNDERSTANDING CONFIGURATION")
    print("=" * 80)

    print("\n--- Available Profiles ---")
    for profile_name in ["default", "conservative", "aggressive", "dev"]:
        config = get_config(profile_name)
        print(f"\n{profile_name.upper()}:")
        print(f"  LLM Enabled: {config.enable_llm_fallback}")
        print(f"  High Confidence Threshold: {config.fuzzy_high_confidence}")
        print(f"  Medium Confidence Threshold: {config.fuzzy_medium_confidence}")
        print(f"  Low Confidence Threshold: {config.fuzzy_low_confidence}")
        print(f"  Logging: Decisions={config.log_classification_decisions}, "
              f"Scores={config.log_fuzzy_scores}, Prompts={config.log_llm_prompts}")

    print("\n" + "=" * 80)
    print("GROQ MODELS")
    print("=" * 80)
    for model_id, info in GROQ_MODELS.items():
        recommended = " (RECOMMENDED)" if info["recommended"] else ""
        print(f"\n{model_id}{recommended}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Context Window: {info['context_window']:,} tokens")
        print(f"  Speed: {info['speed']}")

    print("\n" + "=" * 80)
    print("ENVIRONMENT VARIABLES")
    print("=" * 80)
    print("\nRequired:")
    print("  GROQ_API_KEY - Your Groq API key")
    print("\nOptional:")
    print("  SEMANTIC_PROFILE - Configuration profile (default, conservative, aggressive, dev)")
    print("  ENABLE_LLM_FALLBACK - Override LLM fallback setting (true/false)")

    # Check if API key is set
    if os.environ.get("GROQ_API_KEY"):
        print("\n✓ GROQ_API_KEY is set")
    else:
        print("\n✗ GROQ_API_KEY is NOT set - LLM fallback will be disabled")
