"""Tests for ContextWindowManager utilities."""

import pytest
from datetime import datetime, timedelta

from amplifier_module_util_context import (
    TokenEstimator,
    BudgetFitter,
    FitResult,
    RelevanceScorer,
    EntityDeduplicator,
)


class TestTokenEstimator:
    """Tests for TokenEstimator class."""

    def test_count_tokens_empty_string(self):
        estimator = TokenEstimator()
        assert estimator.count_tokens("") == 0

    def test_count_tokens_short_text(self):
        estimator = TokenEstimator()
        # Approximate mode: ~4 chars per token
        count = estimator.count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be reasonable

    def test_count_tokens_longer_text(self):
        estimator = TokenEstimator()
        text = "This is a longer piece of text that should have more tokens."
        count = estimator.count_tokens(text)
        # ~15 tokens for this text
        assert count > 5
        assert count < 30

    def test_count_tokens_for_entity(self):
        estimator = TokenEstimator()
        entity = {
            "id": "123",
            "text": "Buy groceries for dinner tonight",
            "status": "pending",
        }
        count = estimator.count_tokens_for_entity(entity)
        assert count > 0

    def test_count_tokens_for_entity_specific_fields(self):
        estimator = TokenEstimator()
        entity = {
            "id": "123",
            "text": "Buy groceries for dinner tonight",
            "status": "pending",
            "long_field": "A" * 1000,  # Very long field
        }

        # Count all fields
        count_all = estimator.count_tokens_for_entity(entity)

        # Count only specific fields
        count_specific = estimator.count_tokens_for_entity(entity, fields=["text", "status"])

        assert count_specific < count_all

    def test_count_tokens_for_entity_with_json_field(self):
        estimator = TokenEstimator()
        entity = {
            "id": "123",
            "tags": ["work", "urgent", "meeting"],
            "metadata": {"priority": 1, "assignee": "John"},
        }
        count = estimator.count_tokens_for_entity(entity)
        assert count > 0

    def test_count_tokens_for_entities(self):
        estimator = TokenEstimator()
        entities = [
            {"text": "First todo"},
            {"text": "Second todo"},
            {"text": "Third todo"},
        ]
        count = estimator.count_tokens_for_entities(entities)
        single_count = estimator.count_tokens_for_entity(entities[0])
        assert count >= single_count * 3

    def test_estimate_formatted_tokens(self):
        estimator = TokenEstimator()
        entities = [
            {"text": "Buy groceries", "status": "pending"},
            {"text": "Call mom", "status": "active"},
        ]
        template = "- {text} ({status})"
        count = estimator.estimate_formatted_tokens(entities, template)
        assert count > 0


class TestBudgetFitter:
    """Tests for BudgetFitter class."""

    def setup_method(self):
        self.estimator = TokenEstimator()
        self.fitter = BudgetFitter(self.estimator)

    def test_fit_empty_candidates(self):
        result = self.fitter.fit_to_budget([], max_tokens=1000)
        assert result.selected == []
        assert result.excluded == []
        assert result.tokens_used == 0
        assert result.tokens_remaining == 1000

    def test_fit_all_candidates_fit(self):
        candidates = [
            {"text": "Short"},
            {"text": "Also short"},
        ]
        result = self.fitter.fit_to_budget(candidates, max_tokens=1000)
        assert len(result.selected) == 2
        assert len(result.excluded) == 0
        assert result.tokens_used > 0
        assert result.tokens_remaining < 1000

    def test_fit_exceeds_budget(self):
        candidates = [
            {"text": "A" * 100},  # ~25 tokens
            {"text": "B" * 100},  # ~25 tokens
            {"text": "C" * 100},  # ~25 tokens
        ]
        # Very small budget
        result = self.fitter.fit_to_budget(candidates, max_tokens=30)
        assert len(result.selected) < len(candidates)
        assert len(result.excluded) > 0
        assert result.tokens_used <= 30

    def test_fit_with_priority_key(self):
        candidates = [
            {"text": "Low priority", "score": 0.1},
            {"text": "High priority", "score": 0.9},
            {"text": "Medium priority", "score": 0.5},
        ]
        result = self.fitter.fit_to_budget(
            candidates,
            max_tokens=1000,
            priority_key="score",
            priority_desc=True,
        )
        # Should be ordered by score descending
        assert result.selected[0]["score"] == 0.9
        assert result.selected[1]["score"] == 0.5
        assert result.selected[2]["score"] == 0.1

    def test_fit_with_specific_fields(self):
        candidates = [
            {"text": "Short text", "metadata": "A" * 500},
        ]
        # With all fields - might not fit
        result_all = self.fitter.fit_to_budget(candidates, max_tokens=20)

        # With only text field - should fit
        result_text_only = self.fitter.fit_to_budget(
            candidates,
            max_tokens=20,
            fields=["text"],
        )

        # The text-only version should be more likely to fit
        assert len(result_text_only.selected) >= len(result_all.selected)

    def test_fit_multiple_sources(self):
        sources = {
            "similar": [{"text": "Similar 1"}, {"text": "Similar 2"}],
            "recent": [{"text": "Recent 1"}, {"text": "Recent 2"}],
            "project": [{"text": "Project 1"}],
        }
        budgets = {
            "similar": 50,
            "recent": 50,
            "project": 50,
        }

        results = self.fitter.fit_multiple_sources(
            sources=sources,
            budgets=budgets,
            total_budget=100,
        )

        assert "similar" in results
        assert "recent" in results
        assert "project" in results

        # Total shouldn't exceed total_budget
        total_used = sum(r.tokens_used for r in results.values())
        assert total_used <= 100


class TestRelevanceScorer:
    """Tests for RelevanceScorer class."""

    def setup_method(self):
        self.scorer = RelevanceScorer()

    def test_score_by_recency_new_item(self):
        now = datetime.now()
        entities = [{"created_at": now.isoformat()}]

        result = self.scorer.score_by_recency(entities, "created_at", decay_days=30)

        # New item should have high score
        assert result[0]["recency_score"] > 0.9

    def test_score_by_recency_old_item(self):
        old_date = datetime.now() - timedelta(days=60)
        entities = [{"created_at": old_date.isoformat()}]

        result = self.scorer.score_by_recency(entities, "created_at", decay_days=30)

        # Old item should have low score
        assert result[0]["recency_score"] < 0.5

    def test_score_by_recency_missing_field(self):
        entities = [{"text": "No date field"}]

        result = self.scorer.score_by_recency(entities, "created_at", decay_days=30)

        # Missing field should score 0
        assert result[0]["recency_score"] == 0.0

    def test_score_by_recency_preserves_other_fields(self):
        entities = [{"id": "123", "text": "Test", "created_at": datetime.now().isoformat()}]

        result = self.scorer.score_by_recency(entities, "created_at")

        # Original fields should be preserved
        assert result[0]["id"] == "123"
        assert result[0]["text"] == "Test"

    def test_score_by_similarity(self):
        entities = [
            {"similarity": 0.2},
            {"similarity": 0.8},
            {"similarity": 0.5},
        ]

        result = self.scorer.score_by_similarity(entities)

        # Scores should be normalized
        assert result[0]["similarity_score"] == 0.0  # Lowest
        assert result[1]["similarity_score"] == 1.0  # Highest
        assert 0 < result[2]["similarity_score"] < 1

    def test_score_by_similarity_invert_distance(self):
        entities = [
            {"distance": 0.1},  # Close = more similar
            {"distance": 0.9},  # Far = less similar
        ]

        result = self.scorer.score_by_similarity(
            entities,
            similarity_field="distance",
            invert=True,
        )

        # After inverting, low distance = high score
        assert result[0]["similarity_score"] > result[1]["similarity_score"]

    def test_combine_scores(self):
        entities = [
            {"recency_score": 1.0, "similarity_score": 0.0},
            {"recency_score": 0.0, "similarity_score": 1.0},
        ]

        result = self.scorer.combine_scores(
            entities,
            weights={"recency_score": 0.5, "similarity_score": 0.5},
        )

        # Both should have combined score of 0.5
        assert result[0]["combined_score"] == 0.5
        assert result[1]["combined_score"] == 0.5

    def test_combine_scores_missing_field(self):
        entities = [
            {"recency_score": 1.0},  # Missing similarity_score
        ]

        result = self.scorer.combine_scores(
            entities,
            weights={"recency_score": 0.5, "similarity_score": 0.5},
        )

        # Should only use available score
        assert result[0]["combined_score"] == 1.0

    def test_normalize_scores(self):
        entities = [
            {"raw_score": 10},
            {"raw_score": 20},
            {"raw_score": 30},
        ]

        result = self.scorer.normalize_scores(entities, "raw_score", "normalized")

        assert result[0]["normalized"] == 0.0
        assert result[1]["normalized"] == 0.5
        assert result[2]["normalized"] == 1.0

    def test_add_priority_boost_multiply(self):
        entities = [
            {"score": 0.5, "priority": "high"},
            {"score": 0.5, "priority": "low"},
        ]

        result = self.scorer.add_priority_boost(
            entities,
            priority_field="priority",
            priority_values={"high": 2.0, "low": 0.5},
            score_field="score",
            boost_mode="multiply",
        )

        assert result[0]["score"] == 1.0  # 0.5 * 2.0
        assert result[1]["score"] == 0.25  # 0.5 * 0.5


class TestEntityDeduplicator:
    """Tests for EntityDeduplicator class."""

    def setup_method(self):
        self.deduper = EntityDeduplicator()

    def test_dedupe_by_id_single_list(self):
        entities = [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
        ]

        result = self.deduper.dedupe_by_id(entities)

        assert len(result) == 2

    def test_dedupe_by_id_multiple_lists(self):
        list1 = [{"id": "1", "text": "First"}]
        list2 = [{"id": "1", "text": "Duplicate"}, {"id": "2", "text": "Second"}]

        result = self.deduper.dedupe_by_id(list1, list2)

        assert len(result) == 2
        # First occurrence wins
        assert result[0]["text"] == "First"
        assert result[1]["text"] == "Second"

    def test_dedupe_by_id_custom_field(self):
        entities = [
            {"uuid": "a", "text": "First"},
            {"uuid": "a", "text": "Duplicate"},
        ]

        result = self.deduper.dedupe_by_id(entities, id_field="uuid")

        assert len(result) == 1
        assert result[0]["text"] == "First"

    def test_dedupe_by_field_keep_first(self):
        entities = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "pending"},
            {"id": "3", "status": "active"},
        ]

        result = self.deduper.dedupe_by_field(entities, "status", keep="first")

        assert len(result) == 2
        assert result[0]["id"] == "1"

    def test_dedupe_by_field_keep_last(self):
        entities = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "pending"},
            {"id": "3", "status": "active"},
        ]

        result = self.deduper.dedupe_by_field(entities, "status", keep="last")

        assert len(result) == 2
        assert result[0]["id"] == "2"  # Last pending

    def test_dedupe_with_merge(self):
        list1 = [{"id": "1", "similarity_score": 0.9}]
        list2 = [{"id": "1", "recency_score": 0.8}]

        result = self.deduper.dedupe_with_merge(
            list1,
            list2,
            merge_fields=["similarity_score", "recency_score"],
            merge_strategy="first",
        )

        assert len(result) == 1
        assert result[0]["similarity_score"] == 0.9
        assert result[0]["recency_score"] == 0.8

    def test_partition_by_field(self):
        entities = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "active"},
            {"id": "3", "status": "pending"},
        ]

        result = self.deduper.partition_by_field(entities, "status")

        assert "pending" in result
        assert "active" in result
        assert len(result["pending"]) == 2
        assert len(result["active"]) == 1

    def test_filter_by_ids_keep(self):
        entities = [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
            {"id": "3", "text": "Third"},
        ]

        result = self.deduper.filter_by_ids(
            entities,
            ids_to_keep={"1", "3"},
        )

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "3"

    def test_filter_by_ids_exclude(self):
        entities = [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
            {"id": "3", "text": "Third"},
        ]

        result = self.deduper.filter_by_ids(
            entities,
            ids_to_exclude={"2"},
        )

        assert len(result) == 2
        assert all(e["id"] != "2" for e in result)
