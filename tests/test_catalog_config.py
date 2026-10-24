from __future__ import annotations

import textwrap

import pytest

from mnemo.catalog import CatalogError, load_catalog
from mnemo.config import ExperimentConfig, load_experiment_config


def test_default_catalog_is_consistent(catalog):
    assert len(catalog.names) >= 8
    assert catalog.is_multi("hobby") and not catalog.is_multi("location")
    assert catalog.canonical_value("location", " bangalore ") == "Bangalore"
    assert catalog.lookup_value("PYTHON") == ("preferred_language", "Python")


def test_config_fingerprint_changes_with_values(cfg):
    other = cfg.model_copy(deep=True)
    other.memory.budget_tokens += 1
    assert cfg.fingerprint() != other.fingerprint()
    assert cfg.fingerprint() == load_experiment_config().fingerprint()


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("memory: {budget_tokens: 10, nonsense: 1}\n")
    with pytest.raises(Exception):  # noqa: B017
        load_experiment_config(bad)


def test_split_ranges_must_be_disjoint(cfg):
    data = cfg.model_dump()
    data["benchmark"]["splits"]["val"] = (0, 10)
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)


def test_catalog_validation_errors(tmp_path):
    p = tmp_path / "cat.yaml"
    p.write_text(
        textwrap.dedent("""
        chitchat: [hi]
        attributes:
          a:
            cardinality: single
            importance: 1
            values: [x, y]
            templates: {state: ["I am {v}."]}
    """)
    )
    with pytest.raises(CatalogError, match="change"):
        load_catalog(p)


def test_duplicate_values_across_attributes_rejected(tmp_path):
    p = tmp_path / "cat.yaml"
    body = """
        chitchat: [hi]
        attributes:
          a: {cardinality: multi, importance: 1, values: [x, y], templates: {state: ["{v}"]}}
          b: {cardinality: multi, importance: 1, values: [y, z], templates: {state: ["{v}"]}}
    """
    p.write_text(textwrap.dedent(body))
    with pytest.raises(CatalogError, match="more than one"):
        load_catalog(p)
