"""Tests for FEVER wiki cache and leakage guard fixes."""

import json
import os
import sqlite3
import tempfile

import pytest


# ═══════════════════════════════════════════════════════════
#  Wiki Cache Tests
# ═══════════════════════════════════════════════════════════

def _make_test_cache(tmp_path, pages: dict[str, list[str]]) -> str:
    """Create a minimal SQLite wiki cache for testing."""
    db_path = os.path.join(str(tmp_path), "test_wiki.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE wiki_pages (
            title TEXT PRIMARY KEY,
            sentences_json TEXT NOT NULL
        )
    """)
    for title, sentences in pages.items():
        conn.execute(
            "INSERT INTO wiki_pages (title, sentences_json) VALUES (?, ?)",
            (title, json.dumps(sentences)),
        )
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON wiki_pages(title)")
    conn.commit()
    conn.close()
    return db_path


class TestWikiCache:
    """Test WikiCache read-only SQLite interface."""

    def test_lookup_existing_title(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        db = _make_test_cache(tmp_path, {
            "Albert_Einstein": ["Albert Einstein was a physicist.", "He was born in 1879."],
        })
        cache = WikiCache(db)
        result = cache.lookup("Albert_Einstein")
        assert result == ["Albert Einstein was a physicist.", "He was born in 1879."]
        cache.close()

    def test_lookup_missing_title(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        db = _make_test_cache(tmp_path, {"Page_A": ["text"]})
        cache = WikiCache(db)
        assert cache.lookup("Nonexistent") is None
        cache.close()

    def test_contains(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        db = _make_test_cache(tmp_path, {
            "Page_A": ["sent1"],
            "Page_B": ["sent2"],
        })
        cache = WikiCache(db)
        assert "Page_A" in cache
        assert "Page_B" in cache
        assert "Page_C" not in cache
        cache.close()

    def test_len(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        db = _make_test_cache(tmp_path, {
            "P1": ["s1"], "P2": ["s2"], "P3": ["s3"],
        })
        cache = WikiCache(db)
        assert len(cache) == 3
        cache.close()

    def test_titles(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        pages = {"Alpha": ["a"], "Beta": ["b"], "Gamma": ["g"]}
        db = _make_test_cache(tmp_path, pages)
        cache = WikiCache(db)
        assert set(cache.titles()) == {"Alpha", "Beta", "Gamma"}
        cache.close()

    def test_missing_db_raises(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        with pytest.raises(FileNotFoundError, match="Wiki cache not found"):
            WikiCache(os.path.join(str(tmp_path), "nonexistent.db"))

    def test_cache_stats_exists(self, tmp_path):
        from data.fever_wiki_cache import cache_stats
        db = _make_test_cache(tmp_path, {"X": ["x"], "Y": ["y"]})
        stats = cache_stats(db)
        assert stats["exists"] is True
        assert stats["n_pages"] == 2
        assert stats["size_mb"] > 0
        assert len(stats["titles_hash"]) == 16

    def test_cache_stats_missing(self, tmp_path):
        from data.fever_wiki_cache import cache_stats
        stats = cache_stats(os.path.join(str(tmp_path), "nope.db"))
        assert stats["exists"] is False


class TestWikiCacheAdapter:
    """Test _WikiCacheAdapter dict-like interface used by _concat_evidence_sentences."""

    def test_adapter_getitem(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter
        db = _make_test_cache(tmp_path, {"Title_A": ["sent0", "sent1"]})
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        assert adapter["Title_A"] == ["sent0", "sent1"]
        cache.close()

    def test_adapter_contains(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter
        db = _make_test_cache(tmp_path, {"Title_A": ["sent0"]})
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        assert "Title_A" in adapter
        assert "Missing" not in adapter
        cache.close()

    def test_adapter_get_default(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter
        db = _make_test_cache(tmp_path, {"Title_A": ["s"]})
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        assert adapter.get("Title_A") == ["s"]
        assert adapter.get("Missing", "fallback") == "fallback"
        cache.close()

    def test_adapter_bool(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter
        db = _make_test_cache(tmp_path, {"X": ["x"]})
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        assert bool(adapter) is True
        cache.close()

    def test_adapter_keyerror(self, tmp_path):
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter
        db = _make_test_cache(tmp_path, {"X": ["x"]})
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        with pytest.raises(KeyError):
            _ = adapter["Missing"]
        cache.close()

    def test_concat_evidence_uses_adapter(self, tmp_path):
        """_concat_evidence_sentences works with WikiCacheAdapter (not just dict)."""
        from data.fever_wiki_cache import WikiCache
        from data.fever_dataset import _WikiCacheAdapter, _concat_evidence_sentences
        db = _make_test_cache(tmp_path, {
            "Albert_Einstein": ["Albert Einstein was a physicist.", "He won the Nobel Prize."],
        })
        cache = WikiCache(db)
        adapter = _WikiCacheAdapter(cache)
        evidence_sets = [[[0, 0, "Albert_Einstein", 0], [0, 1, "Albert_Einstein", 1]]]
        result = _concat_evidence_sentences(evidence_sets, wiki_page_map=adapter)
        assert "Albert Einstein was a physicist" in result
        assert "Nobel Prize" in result
        cache.close()


# ═══════════════════════════════════════════════════════════
#  Leakage Guard Tests (data-dependency, not overlap)
# ═══════════════════════════════════════════════════════════

class TestLeakageGuard:
    """Test that FeverPipelineDataset uses data-dependency guard, not overlap."""

    def _make_items(self, n=10):
        return [
            {
                "id": i,
                "claim": f"Claim number {i}",
                "label": "SUPPORTS",
                "label_id": 0,
                "gold_evidence_text": f"Gold evidence for claim {i}",
            }
            for i in range(n)
        ]

    def test_pipeline_accepts_independent_evidence(self):
        from data.fever_dataset import FeverPipelineDataset
        items = self._make_items()
        retrieved = {it["id"]: "independent evidence" for it in items}
        ds = FeverPipelineDataset(items, retrieved)
        assert len(ds) == len(items)

    def test_pipeline_accepts_gold_overlap(self):
        """Overlap between retrieved and gold is NOT leakage — must not raise."""
        from data.fever_dataset import FeverPipelineDataset
        items = self._make_items()
        # Pass gold evidence AS retrieved — this should be ALLOWED
        retrieved = {it["id"]: it["gold_evidence_text"] for it in items}
        ds = FeverPipelineDataset(items, retrieved)
        assert len(ds) == len(items)

    def test_pipeline_getitem_excludes_gold(self):
        from data.fever_dataset import FeverPipelineDataset
        items = self._make_items()
        retrieved = {it["id"]: "retrieved text" for it in items}
        ds = FeverPipelineDataset(items, retrieved)
        sample = ds[0]
        assert "gold_evidence_text" not in sample
        assert sample["evidence"] == "retrieved text"

    def test_pipeline_uses_retrieved_not_gold(self):
        from data.fever_dataset import FeverPipelineDataset
        items = self._make_items()
        retrieved = {it["id"]: f"RETRIEVED for {it['id']}" for it in items}
        ds = FeverPipelineDataset(items, retrieved)
        for i in range(min(5, len(items))):
            sample = ds[i]
            assert sample["evidence"] == f"RETRIEVED for {i}"
            assert sample["evidence"] != items[i]["gold_evidence_text"]

    def test_pipeline_sample_keys(self):
        from data.fever_dataset import FeverPipelineDataset
        items = self._make_items()
        retrieved = {it["id"]: "ev" for it in items}
        ds = FeverPipelineDataset(items, retrieved)
        sample = ds[0]
        expected_keys = {"id", "claim", "evidence", "label_id", "label"}
        assert set(sample.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════
#  Parse wiki sentences
# ═══════════════════════════════════════════════════════════

class TestParseWikiSentences:
    def test_standard_format(self):
        from data.fever_wiki_cache import _parse_wiki_sentences
        raw = "0\tFirst sentence.\n1\tSecond sentence."
        result = _parse_wiki_sentences(raw)
        assert result == ["First sentence.", "Second sentence."]

    def test_empty_string(self):
        from data.fever_wiki_cache import _parse_wiki_sentences
        assert _parse_wiki_sentences("") == [""]

    def test_missing_tab(self):
        from data.fever_wiki_cache import _parse_wiki_sentences
        raw = "0\tHello\nbroken line"
        result = _parse_wiki_sentences(raw)
        assert result[0] == "Hello"
        assert result[1] == ""  # fallback for missing tab


# ═══════════════════════════════════════════════════════════
#  Line endings (.gitattributes)
# ═══════════════════════════════════════════════════════════

class TestGitattributes:
    def test_gitattributes_exists(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, ".gitattributes")
        assert os.path.exists(path), ".gitattributes must exist in repo root"

    def test_gitattributes_has_yaml_lf(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, ".gitattributes")
        content = open(path).read()
        assert "*.yaml" in content and "eol=lf" in content

    def test_gitattributes_has_py_lf(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, ".gitattributes")
        content = open(path).read()
        assert "*.py" in content and "eol=lf" in content


# ═══════════════════════════════════════════════════════════
#  Colab playbook structure
# ═══════════════════════════════════════════════════════════

class TestColabPlaybook:
    def test_cell_count_lte_12(self):
        from colab.fever_playbook import COLAB_CELLS
        assert len(COLAB_CELLS) <= 12, f"Playbook has {len(COLAB_CELLS)} cells, max is 12"

    def test_last_cell_uses_zip(self):
        from colab.fever_playbook import COLAB_CELLS
        last = COLAB_CELLS[-1]
        assert "make_archive" in last, "Last cell must use shutil.make_archive"
        # copytree is OK for collecting into staging dir; the FINAL save must zip
        assert "shutil.copy" in last, "Last cell must copy zip archive to Drive"

    def test_cache_build_cell_exists(self):
        from colab.fever_playbook import COLAB_CELLS
        has_cache = any("build-fever-wiki-cache" in cell for cell in COLAB_CELLS)
        assert has_cache, "Playbook must include wiki cache build step"

    def test_timestamp_in_zip(self):
        from colab.fever_playbook import COLAB_CELLS
        last = COLAB_CELLS[-1]
        assert "timestamp" in last or "strftime" in last, "Zip must include timestamp"


# ═══════════════════════════════════════════════════════════
#  Smoke config
# ═══════════════════════════════════════════════════════════

class TestSmokeConfig:
    def test_smoke_config_exists(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, "configs", "fever_gold_smoke.yaml")
        assert os.path.exists(path)

    def test_smoke_config_small(self):
        import yaml
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, "configs", "fever_gold_smoke.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["data"]["max_train"] <= 500
        assert cfg["data"]["max_dev"] <= 200
        assert cfg["train"]["epochs"] <= 2
