"""amplifier-module-util-context - Amplifier module for LLM context window management.

This module provides utilities for fitting domain state into LLM context windows.
Any application that needs to select and prioritize entities for LLM prompts can use it.

Example usage (direct import):
    from amplifier_module_util_context import (
        TokenEstimator,
        BudgetFitter,
        RelevanceScorer,
        EntityDeduplicator
    )

    estimator = TokenEstimator()
    fitter = BudgetFitter(estimator)
    scorer = RelevanceScorer()
    deduper = EntityDeduplicator()

Example usage (via Amplifier coordinator):
    # After mounting, access via coordinator
    context_utils = coordinator.get("context_utils")
    estimator = context_utils.create_estimator()
    fitter = context_utils.create_fitter(estimator)
"""

from dataclasses import dataclass

from amplifier_module_util_context.budget_fitter import BudgetFitter, FitResult
from amplifier_module_util_context.deduplicator import EntityDeduplicator
from amplifier_module_util_context.relevance_scorer import RelevanceScorer
from amplifier_module_util_context.token_estimator import TokenEstimator

__all__ = [
    "TokenEstimator",
    "BudgetFitter",
    "FitResult",
    "RelevanceScorer",
    "EntityDeduplicator",
    "ContextWindowUtils",
    "mount",
]

# Amplifier module type identifier
__amplifier_module_type__ = "util"


@dataclass
class ContextUtilsConfig:
    """Configuration for ContextWindowManager module.

    Attributes:
        tokenizer_model: Tokenizer to use for token counting.
            - "approximate": Fast approximation (~4 chars per token)
            - "cl100k_base": GPT-4/Claude tokenizer (requires tiktoken)
    """

    tokenizer_model: str = "approximate"


class ContextWindowUtils:
    """Factory for context window management utilities.

    This class is mounted by Amplifier and provides factory methods
    to create configured utility instances.
    """

    def __init__(self, config: ContextUtilsConfig):
        self._config = config

    def create_estimator(self) -> TokenEstimator:
        """Create a TokenEstimator with configured tokenizer."""
        return TokenEstimator(model=self._config.tokenizer_model)

    def create_fitter(self, estimator: TokenEstimator | None = None) -> BudgetFitter:
        """Create a BudgetFitter with optional custom estimator."""
        if estimator is None:
            estimator = self.create_estimator()
        return BudgetFitter(estimator)

    def create_scorer(self) -> RelevanceScorer:
        """Create a RelevanceScorer."""
        return RelevanceScorer()

    def create_deduplicator(self) -> EntityDeduplicator:
        """Create an EntityDeduplicator."""
        return EntityDeduplicator()


async def mount(coordinator, config: dict):
    """Amplifier module entry point.

    Mounts ContextWindowUtils at the 'context_utils' slot.

    Args:
        coordinator: Amplifier coordinator instance.
        config: Configuration dict with optional keys:
            - tokenizer_model: "approximate" (default) or "cl100k_base"
    """
    utils_config = ContextUtilsConfig(
        tokenizer_model=config.get("tokenizer_model", "approximate"),
    )

    utils = ContextWindowUtils(utils_config)

    # Mount at named slot
    await coordinator.mount("context_utils", utils)
