"""Tests for personal_knowledge_base.kb.registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from personal_knowledge_base.kb.registry import KBRegistry, KnowledgeBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path: Path) -> KBRegistry:
    """Return a fresh KBRegistry backed by a temporary SQLite file."""
    db_file = tmp_path / "test_pkb.db"
    return KBRegistry(db_path=str(db_file))


@pytest.fixture()
def populated_registry(registry: KBRegistry) -> KBRegistry:
    """Registry with a parent KB and a child KB pre-created."""
    registry.create_kb(id="parent-kb", name="Parent KB", description="Top-level KB")
    registry.create_kb(
        id="child-kb",
        name="Child KB",
        description="Sub KB",
        parent_id="parent-kb",
    )
    return registry


# ---------------------------------------------------------------------------
# create_kb
# ---------------------------------------------------------------------------


class TestCreateKb:
    def test_create_returns_knowledge_base(self, registry: KBRegistry) -> None:
        kb = registry.create_kb(id="alpha", name="Alpha", description="First KB")
        assert isinstance(kb, KnowledgeBase)
        assert kb.id == "alpha"
        assert kb.name == "Alpha"
        assert kb.description == "First KB"

    def test_create_persists_to_db(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="First KB")
        fetched = registry.get_kb("alpha")
        assert fetched is not None
        assert fetched.id == "alpha"
        assert fetched.name == "Alpha"

    def test_create_defaults(self, registry: KBRegistry) -> None:
        kb = registry.create_kb(id="alpha", name="Alpha", description="desc")
        assert kb.max_documents == 50
        assert kb.document_count == 0
        assert kb.auto_subcategory is False
        assert kb.centroid_vector is None
        assert kb.parent_id is None

    def test_create_with_custom_params(self, registry: KBRegistry) -> None:
        kb = registry.create_kb(
            id="custom",
            name="Custom",
            description="desc",
            max_documents=100,
            auto_subcategory=True,
        )
        assert kb.max_documents == 100
        assert kb.auto_subcategory is True

    def test_create_duplicate_id_raises(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        with pytest.raises(ValueError, match="already exists"):
            registry.create_kb(id="alpha", name="Alpha Again", description="desc2")

    def test_create_with_parent(self, registry: KBRegistry) -> None:
        registry.create_kb(id="parent", name="Parent", description="p")
        child = registry.create_kb(id="child", name="Child", description="c", parent_id="parent")
        assert child.parent_id == "parent"


# ---------------------------------------------------------------------------
# get_kb
# ---------------------------------------------------------------------------


class TestGetKb:
    def test_get_existing(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.id == "alpha"

    def test_get_nonexistent_returns_none(self, registry: KBRegistry) -> None:
        result = registry.get_kb("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# list_kbs
# ---------------------------------------------------------------------------


class TestListKbs:
    def test_list_empty(self, registry: KBRegistry) -> None:
        assert registry.list_kbs() == []

    def test_list_all_kbs(self, populated_registry: KBRegistry) -> None:
        kbs = populated_registry.list_kbs()
        assert len(kbs) == 2
        ids = {kb.id for kb in kbs}
        assert "parent-kb" in ids
        assert "child-kb" in ids

    def test_list_top_level_only(self, populated_registry: KBRegistry) -> None:
        kbs = populated_registry.list_kbs(include_sub=False)
        assert len(kbs) == 1
        assert kbs[0].id == "parent-kb"

    def test_list_include_sub_default_true(self, populated_registry: KBRegistry) -> None:
        kbs = populated_registry.list_kbs()
        assert len(kbs) == 2

    def test_list_multiple_top_level(self, registry: KBRegistry) -> None:
        registry.create_kb(id="kb1", name="KB1", description="d1")
        registry.create_kb(id="kb2", name="KB2", description="d2")
        kbs = registry.list_kbs(include_sub=False)
        assert len(kbs) == 2


# ---------------------------------------------------------------------------
# update_centroid
# ---------------------------------------------------------------------------


class TestUpdateCentroid:
    def test_update_persists_vector(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        centroid = [0.1, 0.2, 0.3]
        registry.update_centroid("alpha", centroid)
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.centroid_vector == centroid

    def test_update_returns_float_list(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.update_centroid("alpha", [1.0, 2.5, -0.75])
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert isinstance(kb.centroid_vector, list)
        assert all(isinstance(v, float) for v in kb.centroid_vector)

    def test_update_overwrites_previous(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.update_centroid("alpha", [0.1, 0.2])
        registry.update_centroid("alpha", [0.9, 0.8])
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.centroid_vector == [0.9, 0.8]

    def test_update_centroid_not_found_raises(self, registry: KBRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            registry.update_centroid("ghost", [0.1, 0.2])


# ---------------------------------------------------------------------------
# increment_doc_count
# ---------------------------------------------------------------------------


class TestIncrementDocCount:
    def test_increment_by_one(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.increment_doc_count("alpha")
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.document_count == 1

    def test_increment_by_delta(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.increment_doc_count("alpha", delta=5)
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.document_count == 5

    def test_increment_accumulates(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.increment_doc_count("alpha")
        registry.increment_doc_count("alpha")
        kb = registry.get_kb("alpha")
        assert kb is not None
        assert kb.document_count == 2

    def test_increment_not_found_raises(self, registry: KBRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            registry.increment_doc_count("ghost")


# ---------------------------------------------------------------------------
# delete_kb
# ---------------------------------------------------------------------------


class TestDeleteKb:
    def test_delete_removes_kb(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.delete_kb("alpha")
        assert registry.get_kb("alpha") is None

    def test_delete_not_in_list(self, registry: KBRegistry) -> None:
        registry.create_kb(id="alpha", name="Alpha", description="desc")
        registry.delete_kb("alpha")
        assert all(kb.id != "alpha" for kb in registry.list_kbs())

    def test_delete_not_found_raises(self, registry: KBRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            registry.delete_kb("ghost")


# ---------------------------------------------------------------------------
# seed_defaults
# ---------------------------------------------------------------------------


class TestSeedDefaults:
    def test_seed_creates_three_kbs(self, registry: KBRegistry) -> None:
        registry.seed_defaults()
        kbs = registry.list_kbs()
        assert len(kbs) == 3

    def test_seed_creates_expected_ids(self, registry: KBRegistry) -> None:
        registry.seed_defaults()
        ids = {kb.id for kb in registry.list_kbs()}
        assert ids == {"quant-trading", "ml-ai", "general"}

    def test_seed_is_idempotent(self, registry: KBRegistry) -> None:
        registry.seed_defaults()
        registry.seed_defaults()  # should not raise or duplicate
        kbs = registry.list_kbs()
        assert len(kbs) == 3

    def test_seed_does_not_overwrite_existing(self, registry: KBRegistry) -> None:
        registry.seed_defaults()
        registry.create_kb(id="extra", name="Extra", description="extra kb")
        registry.seed_defaults()
        kbs = registry.list_kbs()
        # Only default 3 + the one we added = 4
        assert len(kbs) == 4

    def test_seed_kb_descriptions_are_nonempty(self, registry: KBRegistry) -> None:
        registry.seed_defaults()
        for kb in registry.list_kbs():
            assert kb.description, f"KB '{kb.id}' has empty description"


# ---------------------------------------------------------------------------
# KBRegistry with tempfile (alternative path construction)
# ---------------------------------------------------------------------------


class TestTempfileIntegration:
    def test_registry_with_named_tempfile(self) -> None:
        """Verify KBRegistry works with tempfile.NamedTemporaryFile path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "integration.db"
            reg = KBRegistry(db_path=str(db_path))
            reg.seed_defaults()
            assert len(reg.list_kbs()) == 3
