"""Token estimation utilities for LLM context management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class TokenEstimator:
    """Estimate token counts for text and entities.

    This class provides methods to estimate how many tokens text or entities
    will consume in an LLM's context window. It supports both exact counting
    (when tiktoken is available) and fast approximation.

    Example:
        estimator = TokenEstimator()
        count = estimator.count_tokens("Hello, world!")

        # Count tokens for specific fields of an entity
        entity = {"text": "Buy groceries", "status": "pending"}
        count = estimator.count_tokens_for_entity(entity, fields=["text"])
    """

    def __init__(self, model: str = "approximate"):
        """Initialize with a tokenizer model.

        Args:
            model: Tokenizer to use. Options:
                - "cl100k_base": GPT-4, GPT-3.5-turbo, Claude (most accurate)
                - "p50k_base": text-davinci-003
                - "approximate": Fast approximation (~4 chars per token)
        """
        self.model = model
        self._encoder = None

        if model != "approximate":
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding(model)
            except ImportError:
                # tiktoken not installed, fall back to approximation
                self._encoder = None
            except Exception:
                # Unknown encoding, fall back to approximation
                self._encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        if self._encoder is not None:
            return len(self._encoder.encode(text))

        # Approximate: ~4 characters per token on average for English text
        # This is a reasonable approximation for most LLMs
        return max(1, len(text) // 4)

    def count_tokens_for_entity(
        self,
        entity: dict,
        fields: Sequence[str] | None = None,
    ) -> int:
        """Count tokens for an entity.

        Args:
            entity: Dictionary representing the entity.
            fields: If specified, only count tokens for these fields.
                   If None, counts all fields.

        Returns:
            Total token count for the entity.
        """
        total = 0
        fields_to_count = fields if fields is not None else entity.keys()

        for field in fields_to_count:
            if field in entity and entity[field] is not None:
                value = entity[field]
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, (list, dict)):
                    total += self.count_tokens(json.dumps(value))
                else:
                    total += self.count_tokens(str(value))

        return total

    def count_tokens_for_entities(
        self,
        entities: Sequence[dict],
        fields: Sequence[str] | None = None,
    ) -> int:
        """Count total tokens for a list of entities.

        Args:
            entities: List of entity dictionaries.
            fields: If specified, only count tokens for these fields.

        Returns:
            Total token count for all entities.
        """
        return sum(self.count_tokens_for_entity(e, fields) for e in entities)

    def estimate_formatted_tokens(
        self,
        entities: Sequence[dict],
        template: str,
    ) -> int:
        """Estimate tokens when entities will be formatted with a template.

        The template uses {field} placeholders that will be replaced with
        entity field values.

        Args:
            entities: List of entity dictionaries.
            template: Format template with {field} placeholders.
                     Example: "- {text} (status: {status})"

        Returns:
            Estimated token count after formatting.
        """
        total = 0

        for entity in entities:
            try:
                formatted = template.format(**entity)
                total += self.count_tokens(formatted)
            except KeyError:
                # If template has fields not in entity, estimate conservatively
                total += self.count_tokens(template)
                total += self.count_tokens_for_entity(entity)

        return total
