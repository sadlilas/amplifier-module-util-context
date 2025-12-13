"""Entity deduplication utilities for LLM context management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class EntityDeduplicator:
    """Remove duplicate entities from multiple sources.

    When gathering context from multiple sources (e.g., similar items,
    recent items, project items), the same entity might appear in multiple
    lists. This class helps merge them while preserving relevant metadata.

    Example:
        deduper = EntityDeduplicator()

        # Merge lists, keeping first occurrence of each ID
        unique = deduper.dedupe_by_id(similar_todos, recent_todos, project_todos)

        # Or dedupe within a single list by a field
        unique = deduper.dedupe_by_field(todos, field="text", keep="first")
    """

    def dedupe_by_id(
        self,
        *entity_lists: Sequence[dict],
        id_field: str = "id",
    ) -> list[dict]:
        """Merge multiple lists, keeping first occurrence of each ID.

        Entities from earlier lists take precedence over later lists.

        Args:
            *entity_lists: Variable number of entity lists to merge.
            id_field: Field containing the unique identifier.

        Returns:
            Merged list with duplicates removed.

        Example:
            # Items from similar will be kept if same ID appears in recent
            unique = deduper.dedupe_by_id(similar, recent, project)
        """
        seen_ids: set[str] = set()
        result: list[dict] = []

        for entity_list in entity_lists:
            for entity in entity_list:
                entity_id = entity.get(id_field)
                if entity_id is not None and entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    result.append(entity)

        return result

    def dedupe_by_field(
        self,
        entities: Sequence[dict],
        field: str,
        keep: str = "first",
    ) -> list[dict]:
        """Remove duplicates based on a field value.

        Args:
            entities: List of entities to deduplicate.
            field: Field to check for duplicates.
            keep: Which duplicate to keep:
                 - "first": Keep first occurrence
                 - "last": Keep last occurrence

        Returns:
            Deduplicated list.
        """
        if keep not in ("first", "last"):
            raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")

        if keep == "first":
            seen: set = set()
            result = []
            for entity in entities:
                value = entity.get(field)
                if value not in seen:
                    seen.add(value)
                    result.append(entity)
            return result
        else:  # keep == "last"
            seen_indices: dict = {}
            for i, entity in enumerate(entities):
                value = entity.get(field)
                seen_indices[value] = i

            # Return in original order, keeping only last occurrences
            result = []
            for i, entity in enumerate(entities):
                value = entity.get(field)
                if seen_indices.get(value) == i:
                    result.append(entity)
            return result

    def dedupe_with_merge(
        self,
        *entity_lists: Sequence[dict],
        id_field: str = "id",
        merge_fields: Sequence[str] | None = None,
        merge_strategy: str = "first",
    ) -> list[dict]:
        """Merge multiple lists, combining data from duplicates.

        When the same entity appears in multiple lists, this method can
        merge specific fields from all occurrences.

        Args:
            *entity_lists: Variable number of entity lists to merge.
            id_field: Field containing the unique identifier.
            merge_fields: Fields to merge from duplicates. If None, keeps first.
            merge_strategy: How to merge fields:
                          - "first": Use first non-None value
                          - "last": Use last non-None value
                          - "collect": Collect all values into a list

        Returns:
            Merged list with duplicate data combined.

        Example:
            # Merge scores from different sources
            merged = deduper.dedupe_with_merge(
                similar,  # has similarity_score
                recent,   # has recency_score
                merge_fields=["similarity_score", "recency_score"],
                merge_strategy="first"
            )
        """
        entities_by_id: dict[str, dict] = {}
        order: list[str] = []

        for entity_list in entity_lists:
            for entity in entity_list:
                entity_id = entity.get(id_field)
                if entity_id is None:
                    continue

                if entity_id not in entities_by_id:
                    entities_by_id[entity_id] = dict(entity)
                    order.append(entity_id)
                elif merge_fields:
                    existing = entities_by_id[entity_id]
                    for field in merge_fields:
                        if field in entity:
                            if merge_strategy == "first":
                                if existing.get(field) is None:
                                    existing[field] = entity[field]
                            elif merge_strategy == "last":
                                if entity[field] is not None:
                                    existing[field] = entity[field]
                            elif merge_strategy == "collect":
                                if field not in existing:
                                    existing[field] = []
                                elif not isinstance(existing[field], list):
                                    existing[field] = [existing[field]]
                                existing[field].append(entity[field])

        return [entities_by_id[eid] for eid in order]

    def partition_by_field(
        self,
        entities: Sequence[dict],
        field: str,
    ) -> dict[str, list[dict]]:
        """Partition entities into groups by a field value.

        Useful for grouping entities before applying different processing.

        Args:
            entities: List of entities to partition.
            field: Field to partition by.

        Returns:
            Dict mapping field values to lists of entities.

        Example:
            by_status = deduper.partition_by_field(todos, "status")
            # by_status = {"active": [...], "completed": [...], "pending": [...]}
        """
        result: dict[str, list[dict]] = {}

        for entity in entities:
            value = entity.get(field)
            key = str(value) if value is not None else "__none__"

            if key not in result:
                result[key] = []
            result[key].append(entity)

        return result

    def filter_by_ids(
        self,
        entities: Sequence[dict],
        ids_to_keep: set[str] | None = None,
        ids_to_exclude: set[str] | None = None,
        id_field: str = "id",
    ) -> list[dict]:
        """Filter entities by ID inclusion/exclusion.

        Args:
            entities: List of entities to filter.
            ids_to_keep: If provided, only keep entities with these IDs.
            ids_to_exclude: If provided, exclude entities with these IDs.
            id_field: Field containing the unique identifier.

        Returns:
            Filtered list of entities.
        """
        result = []

        for entity in entities:
            entity_id = entity.get(id_field)

            if ids_to_keep is not None and entity_id not in ids_to_keep:
                continue

            if ids_to_exclude is not None and entity_id in ids_to_exclude:
                continue

            result.append(entity)

        return result
