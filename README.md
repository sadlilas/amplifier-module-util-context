# amplifier-module-util-context

Amplifier module for LLM context window management utilities.

## Overview

This module provides generic utilities for fitting domain state into LLM context windows. Any application that needs to select and prioritize entities for LLM prompts can use these utilities.

## Installation

```bash
pip install amplifier-module-util-context @ git+https://github.com/salil/amplifier-module-util-context
```

## Usage

### Direct Import

```python
from amplifier_module_util_context import (
    TokenEstimator,
    BudgetFitter,
    RelevanceScorer,
    EntityDeduplicator
)

# Estimate tokens
estimator = TokenEstimator()
count = estimator.count_tokens("Hello, world!")

# Fit entities to budget
fitter = BudgetFitter(estimator)
result = fitter.fit_to_budget(entities, max_tokens=3000)

# Score by recency
scorer = RelevanceScorer()
scored = scorer.score_by_recency(entities, "created_at", decay_days=30)

# Deduplicate from multiple sources
deduper = EntityDeduplicator()
unique = deduper.dedupe_by_id(list1, list2, list3)
```

### Via Amplifier Coordinator

```python
# In your Amplifier profile:
# modules:
#   context_utils:
#     source: "amplifier-module-util-context"
#     config:
#       tokenizer_model: "approximate"

# After mounting:
context_utils = coordinator.get("context_utils")
estimator = context_utils.create_estimator()
fitter = context_utils.create_fitter(estimator)
```

## Components

### TokenEstimator

Estimate token counts for text and entities.

- `count_tokens(text)` - Count tokens in a string
- `count_tokens_for_entity(entity, fields)` - Count tokens for an entity
- `count_tokens_for_entities(entities, fields)` - Count tokens for multiple entities

### BudgetFitter

Select entities that fit within a token budget.

- `fit_to_budget(candidates, max_tokens, ...)` - Select entities that fit
- `fit_multiple_sources(sources, budgets, total_budget)` - Fit from multiple sources

### RelevanceScorer

Score entities by various relevance criteria.

- `score_by_recency(entities, date_field, decay_days)` - Exponential decay scoring
- `score_by_similarity(entities, similarity_field)` - Normalize similarity scores
- `combine_scores(entities, weights)` - Weighted combination of scores

### EntityDeduplicator

Remove duplicate entities from multiple sources.

- `dedupe_by_id(*lists, id_field)` - Merge lists, keep first occurrence
- `dedupe_by_field(entities, field, keep)` - Dedupe by field value
- `dedupe_with_merge(*lists, merge_fields)` - Merge data from duplicates

## License

MIT
