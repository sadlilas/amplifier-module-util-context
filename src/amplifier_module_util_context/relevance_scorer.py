"""Relevance scoring utilities for LLM context management."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class RelevanceScorer:
    """Score entities by various relevance criteria.

    This class provides utilities to assign relevance scores to entities
    based on recency, similarity, and other factors. Scores can then be
    combined and used for prioritization when fitting to token budgets.

    Example:
        scorer = RelevanceScorer()

        # Score by recency
        entities = scorer.score_by_recency(entities, "created_at", decay_days=30)

        # Combine with similarity scores from vector search
        entities = scorer.combine_scores(
            entities,
            weights={"recency_score": 0.3, "similarity_score": 0.7}
        )
    """

    def score_by_recency(
        self,
        entities: Sequence[dict],
        date_field: str,
        decay_days: int = 30,
        score_field: str = "recency_score",
        reference_time: datetime | None = None,
    ) -> list[dict]:
        """Add recency scores (0-1) based on age.

        Uses exponential decay so newer items score higher. An item that is
        `decay_days` old will have a score of ~0.37 (1/e).

        Args:
            entities: Entities to score.
            date_field: Field containing the datetime to measure from.
            decay_days: Days until score reaches ~0.37 (1/e).
                       Smaller values = faster decay.
            score_field: Field name for the output score.
            reference_time: Time to measure from. Defaults to now.

        Returns:
            New list of entities with score_field added.
        """
        now = reference_time or datetime.now()
        result = []

        for entity in entities:
            entity_copy = dict(entity)

            if date_field in entity and entity[date_field]:
                date_value = entity[date_field]

                # Parse string dates
                if isinstance(date_value, str):
                    try:
                        date_value = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                    except ValueError:
                        entity_copy[score_field] = 0.0
                        result.append(entity_copy)
                        continue

                # Handle timezone-aware vs naive datetimes
                if date_value.tzinfo is not None and now.tzinfo is None:
                    date_value = date_value.replace(tzinfo=None)
                elif date_value.tzinfo is None and now.tzinfo is not None:
                    now = now.replace(tzinfo=None)

                age_days = (now - date_value).total_seconds() / 86400

                # Exponential decay: score = e^(-age/decay)
                # Future dates get score of 1.0
                if age_days < 0:
                    score = 1.0
                else:
                    score = math.exp(-age_days / decay_days)

                entity_copy[score_field] = max(0.0, min(1.0, score))
            else:
                entity_copy[score_field] = 0.0

            result.append(entity_copy)

        return result

    def score_by_similarity(
        self,
        entities: Sequence[dict],
        similarity_field: str = "similarity",
        score_field: str = "similarity_score",
        invert: bool = False,
    ) -> list[dict]:
        """Normalize similarity scores to 0-1 range.

        Assumes entities already have similarity values from vector search.
        Some vector databases return distances (lower = more similar) while
        others return similarity scores (higher = more similar).

        Args:
            entities: Entities to score.
            similarity_field: Field containing raw similarity/distance value.
            score_field: Field name for the normalized score.
            invert: If True, treat input as distance (lower = more similar).
                   If False, treat as similarity (higher = more similar).

        Returns:
            New list of entities with normalized score_field added.
        """
        if not entities:
            return []

        # Extract raw values
        raw_values = []
        for entity in entities:
            val = entity.get(similarity_field)
            if val is not None:
                try:
                    raw_values.append(float(val))
                except (TypeError, ValueError):
                    pass

        if not raw_values:
            # No valid values, return entities with 0 scores
            return [{**e, score_field: 0.0} for e in entities]

        min_val = min(raw_values)
        max_val = max(raw_values)
        range_val = max_val - min_val

        result = []
        for entity in entities:
            entity_copy = dict(entity)
            raw = entity.get(similarity_field)

            if raw is not None:
                try:
                    raw_float = float(raw)
                    if range_val > 0:
                        normalized = (raw_float - min_val) / range_val
                    else:
                        normalized = 1.0  # All values are the same

                    # Invert if treating as distance
                    if invert:
                        normalized = 1.0 - normalized

                    entity_copy[score_field] = normalized
                except (TypeError, ValueError):
                    entity_copy[score_field] = 0.0
            else:
                entity_copy[score_field] = 0.0

            result.append(entity_copy)

        return result

    def combine_scores(
        self,
        entities: Sequence[dict],
        weights: dict[str, float],
        output_field: str = "combined_score",
    ) -> list[dict]:
        """Combine multiple scores with weights.

        Creates a weighted average of multiple score fields.

        Args:
            entities: Entities to score.
            weights: Dict mapping score field names to their weights.
                    Example: {"recency_score": 0.3, "similarity_score": 0.7}
            output_field: Field name for the combined score.

        Returns:
            New list of entities with combined score added.

        Example:
            scored = scorer.combine_scores(
                entities,
                weights={
                    "recency_score": 0.3,
                    "similarity_score": 0.5,
                    "priority": 0.2
                }
            )
        """
        if not weights:
            return [{**e, output_field: 0.0} for e in entities]

        total_weight = sum(weights.values())

        result = []
        for entity in entities:
            entity_copy = dict(entity)
            weighted_sum = 0.0
            actual_weight = 0.0

            for field, weight in weights.items():
                value = entity.get(field)
                if value is not None:
                    try:
                        weighted_sum += float(value) * weight
                        actual_weight += weight
                    except (TypeError, ValueError):
                        pass

            # Normalize by actual weight used (handles missing fields)
            if actual_weight > 0:
                entity_copy[output_field] = weighted_sum / actual_weight
            else:
                entity_copy[output_field] = 0.0

            result.append(entity_copy)

        return result

    def normalize_scores(
        self,
        entities: Sequence[dict],
        score_field: str,
        output_field: str | None = None,
    ) -> list[dict]:
        """Normalize scores to 0-1 range using min-max normalization.

        Args:
            entities: Entities to normalize.
            score_field: Field containing scores to normalize.
            output_field: Field for normalized scores. If None, overwrites score_field.

        Returns:
            New list of entities with normalized scores.
        """
        if output_field is None:
            output_field = score_field

        if not entities:
            return []

        # Extract values
        values = []
        for entity in entities:
            val = entity.get(score_field)
            if val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass

        if not values:
            return [{**e, output_field: 0.0} for e in entities]

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        result = []
        for entity in entities:
            entity_copy = dict(entity)
            val = entity.get(score_field)

            if val is not None:
                try:
                    val_float = float(val)
                    if range_val > 0:
                        entity_copy[output_field] = (val_float - min_val) / range_val
                    else:
                        entity_copy[output_field] = 1.0
                except (TypeError, ValueError):
                    entity_copy[output_field] = 0.0
            else:
                entity_copy[output_field] = 0.0

            result.append(entity_copy)

        return result

    def add_priority_boost(
        self,
        entities: Sequence[dict],
        priority_field: str,
        priority_values: dict[str, float],
        score_field: str,
        output_field: str | None = None,
        boost_mode: str = "multiply",
    ) -> list[dict]:
        """Boost scores based on a priority field.

        Useful for giving certain statuses or categories higher priority.

        Args:
            entities: Entities to boost.
            priority_field: Field containing priority category (e.g., "status").
            priority_values: Dict mapping field values to boost factors.
                           Example: {"high": 1.5, "normal": 1.0, "low": 0.5}
            score_field: Field containing the score to boost.
            output_field: Field for boosted score. If None, overwrites score_field.
            boost_mode: How to apply boost:
                       - "multiply": score * boost
                       - "add": score + boost

        Returns:
            New list of entities with boosted scores.
        """
        if output_field is None:
            output_field = score_field

        result = []
        for entity in entities:
            entity_copy = dict(entity)

            score = entity.get(score_field, 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            priority = entity.get(priority_field)
            boost = priority_values.get(priority, 1.0 if boost_mode == "multiply" else 0.0)

            if boost_mode == "multiply":
                entity_copy[output_field] = score * boost
            else:
                entity_copy[output_field] = score + boost

            result.append(entity_copy)

        return result
