"""Budget fitting utilities for LLM context management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from amplifier_module_util_context.token_estimator import TokenEstimator


@dataclass
class FitResult:
    """Result of fitting entities to a token budget.

    Attributes:
        selected: Entities that fit within the budget.
        excluded: Entities that didn't fit.
        tokens_used: Number of tokens consumed by selected entities.
        tokens_remaining: Budget remaining after selection.
    """

    selected: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    tokens_used: int = 0
    tokens_remaining: int = 0


class BudgetFitter:
    """Select entities that fit within a token budget.

    This class helps you select the most important entities that will fit
    within your LLM's context window limit.

    Example:
        estimator = TokenEstimator()
        fitter = BudgetFitter(estimator)

        # Select entities that fit in 3000 tokens, prioritizing by score
        result = fitter.fit_to_budget(
            candidates=entities,
            max_tokens=3000,
            fields=["text", "status"],
            priority_key="relevance_score"
        )

        print(f"Selected {len(result.selected)} entities using {result.tokens_used} tokens")
    """

    def __init__(self, estimator: TokenEstimator):
        """Initialize with a token estimator.

        Args:
            estimator: TokenEstimator instance for counting tokens.
        """
        self.estimator = estimator

    def fit_to_budget(
        self,
        candidates: Sequence[dict],
        max_tokens: int,
        fields: Sequence[str] | None = None,
        priority_key: str | None = None,
        priority_desc: bool = True,
    ) -> FitResult:
        """Select entities that fit within token budget.

        Greedily selects entities in priority order until the budget is exhausted.

        Args:
            candidates: Entities to select from.
            max_tokens: Maximum tokens allowed.
            fields: Fields to include in token count. If None, counts all fields.
            priority_key: Field to sort by (higher priority first if desc=True).
                         If None, maintains original order.
            priority_desc: Sort descending (True) or ascending (False).

        Returns:
            FitResult with selected entities and metadata.
        """
        if not candidates:
            return FitResult(
                selected=[],
                excluded=[],
                tokens_used=0,
                tokens_remaining=max_tokens,
            )

        # Sort by priority if specified
        if priority_key:
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get(priority_key, 0) if x.get(priority_key) is not None else 0,
                reverse=priority_desc,
            )
        else:
            sorted_candidates = list(candidates)

        selected: list[dict] = []
        excluded: list[dict] = []
        tokens_used = 0

        for entity in sorted_candidates:
            entity_tokens = self.estimator.count_tokens_for_entity(entity, fields)

            if tokens_used + entity_tokens <= max_tokens:
                selected.append(entity)
                tokens_used += entity_tokens
            else:
                excluded.append(entity)

        return FitResult(
            selected=selected,
            excluded=excluded,
            tokens_used=tokens_used,
            tokens_remaining=max_tokens - tokens_used,
        )

    def fit_multiple_sources(
        self,
        sources: dict[str, Sequence[dict]],
        budgets: dict[str, int],
        total_budget: int,
        fields: dict[str, Sequence[str]] | None = None,
        priority_keys: dict[str, str] | None = None,
    ) -> dict[str, FitResult]:
        """Fit entities from multiple sources with individual and total budgets.

        This is useful when you want to include entities from different sources
        (e.g., similar items, recent items, project items) but want to limit
        how much of each type while also respecting a total budget.

        Args:
            sources: Dict mapping source name to list of entities.
                    Example: {"similar": [...], "recent": [...], "project": [...]}
            budgets: Dict mapping source name to max tokens for that source.
                    Example: {"similar": 1500, "recent": 1000, "project": 500}
            total_budget: Total tokens can't exceed this across all sources.
            fields: Optional dict mapping source name to fields to count.
                   If not specified for a source, counts all fields.
            priority_keys: Optional dict mapping source name to priority field.

        Returns:
            Dict mapping source name to FitResult for that source.

        Example:
            sources = {
                "similar": semantically_similar_todos,
                "recent": recent_todos,
                "project": project_todos
            }
            budgets = {
                "similar": 1500,
                "recent": 1000,
                "project": 500
            }
            results = fitter.fit_multiple_sources(
                sources=sources,
                budgets=budgets,
                total_budget=2500  # Total across all sources
            )
        """
        results: dict[str, FitResult] = {}
        total_used = 0

        fields = fields or {}
        priority_keys = priority_keys or {}

        for source_name, entities in sources.items():
            source_budget = budgets.get(source_name, total_budget)

            # Respect both source budget and remaining total budget
            available_budget = min(source_budget, total_budget - total_used)

            if available_budget <= 0:
                results[source_name] = FitResult(
                    selected=[],
                    excluded=list(entities),
                    tokens_used=0,
                    tokens_remaining=0,
                )
                continue

            result = self.fit_to_budget(
                candidates=entities,
                max_tokens=available_budget,
                fields=fields.get(source_name),
                priority_key=priority_keys.get(source_name),
            )

            results[source_name] = result
            total_used += result.tokens_used

        return results
