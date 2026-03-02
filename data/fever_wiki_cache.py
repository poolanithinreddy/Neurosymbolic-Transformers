"""SQLite-based minimal FEVER wiki page cache.

Problem: HuggingFace fever/v1.0 wiki_pages split has ~5.4M pages.
Loading all into memory causes OOM on Colab (12 GB RAM).

Solution: Two-pass approach:
  Pass 1 — Scan train + dev annotations to collect the ~25k unique
           page titles actually referenced by evidence annotations.
  Pass 2 — Stream wiki_pages, keep only needed pages, store in SQLite.

Result: ~15 MB SQLite file with O(1) title lookup, deterministic,
        and small enough for Colab.

Usage:
    # Build cache (once, ~5 min on Colab)
    python main.py build-fever-wiki-cache

    # Or programmatically:
    from data.fever_wiki_cache import build_wiki_cache, WikiCache
    build_wiki_cache(cache_path="data/fever_wiki.db")
    cache = WikiCache("data/fever_wiki.db")
    sentences = cache.lookup("Albert_Einstein")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any

logger = logging.getLogger("fever_wiki_cache")

_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "fever_wiki.db"
)


class WikiCache:
    """Read-only SQLite-backed wiki page cache with O(1) title lookup.

    Thread-safe: each call opens a fresh cursor (SQLite handles locking).
    """

    def __init__(self, db_path: str = _DEFAULT_CACHE_PATH):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Wiki cache not found at {db_path}. "
                f"Run: python main.py build-fever-wiki-cache"
            )
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")

    def lookup(self, title: str) -> list[str] | None:
        """Return list of sentence strings for a wiki page title, or None."""
        cur = self._conn.execute(
            "SELECT sentences_json FROM wiki_pages WHERE title = ?",
            (title,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def __contains__(self, title: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM wiki_pages WHERE title = ? LIMIT 1",
            (title,),
        )
        return cur.fetchone() is not None

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM wiki_pages")
        return cur.fetchone()[0]

    def titles(self) -> list[str]:
        """Return all cached titles (for diagnostics)."""
        cur = self._conn.execute("SELECT title FROM wiki_pages")
        return [row[0] for row in cur.fetchall()]

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass


def cache_stats(db_path: str = _DEFAULT_CACHE_PATH) -> dict[str, Any]:
    """Return cache statistics: title count, DB size, manifest hash."""
    if not os.path.exists(db_path):
        return {"exists": False}
    cache = WikiCache(db_path)
    n_pages = len(cache)
    titles = sorted(cache.titles())
    content_hash = hashlib.sha256(
        json.dumps(titles, sort_keys=True).encode()
    ).hexdigest()[:16]
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    cache.close()
    return {
        "exists": True,
        "path": db_path,
        "n_pages": n_pages,
        "size_mb": round(db_size_mb, 2),
        "titles_hash": content_hash,
    }


def _collect_needed_titles(ds, max_train: int | None = None,
                           max_dev: int | None = None) -> set[str]:
    """Pass 1: Scan train + dev annotations to find referenced page titles.

    Handles two HF FEVER formats:
      - Nested: row['evidence'] = [[[ann_id, ev_id, wiki_url, sent_id], ...], ...]
      - Flat:   row['evidence_wiki_url'] = 'Page_Title'  (one row per evidence piece)
    """
    needed = set()
    for split_name, hf_split in [("train", "train"), ("dev", "labelled_dev")]:
        if hf_split not in ds:
            # Try alternates
            for alt in (["paper_dev", "dev"] if hf_split == "labelled_dev" else []):
                if alt in ds:
                    hf_split = alt
                    break
            else:
                continue

        data = ds[hf_split]
        max_n = max_train if split_name == "train" else max_dev
        if max_n is not None and max_n < len(data):
            data = data.select(range(max_n))

        columns = data.column_names if hasattr(data, 'column_names') else []

        # Flat-row format: evidence_wiki_url is a column
        if "evidence_wiki_url" in columns:
            for row in data:
                url = row.get("evidence_wiki_url", "")
                if url:
                    needed.add(url)
        else:
            # Nested format
            for row in data:
                evidence_sets = row.get("evidence", [])
                if not evidence_sets:
                    continue
                for eset in evidence_sets:
                    if not eset:
                        continue
                    for ann in eset:
                        if ann and len(ann) >= 4 and ann[2] is not None:
                            needed.add(ann[2])

    return needed


def _parse_wiki_sentences(lines_raw: str) -> list[str]:
    """Parse FEVER wiki_pages line format: '0\\tsentence0\\n1\\tsentence1\\n...'"""
    sentences = []
    if not isinstance(lines_raw, str):
        return sentences
    for line in lines_raw.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            sent_text = parts[1].strip()
            sentences.append(sent_text if sent_text else "")
        else:
            sentences.append("")
    return sentences


def build_wiki_cache(
    cache_path: str = _DEFAULT_CACHE_PATH,
    hf_cache_dir: str | None = None,
    max_train: int | None = None,
    max_dev: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Build SQLite wiki cache from HuggingFace FEVER dataset.

    Two-pass approach:
      Pass 1: Scan train+dev annotations → needed titles (~25k).
      Pass 2: Stream wiki_pages split, keep only needed pages → SQLite.

    Args:
        cache_path: where to write the SQLite file.
        hf_cache_dir: HuggingFace dataset cache directory.
        max_train: limit train samples scanned (for smoke tests).
        max_dev: limit dev samples scanned (for smoke tests).
        smoke: if True, use max_train=2000, max_dev=500 for quick build.

    Returns:
        dict with build statistics (n_needed, n_found, n_missing, etc.).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  (required for wiki cache build)")

    if smoke:
        max_train = max_train or 2000
        max_dev = max_dev or 500

    t0 = time.time()
    logger.info("Building FEVER wiki cache...")

    # Load HF dataset
    logger.info("  Loading HF fever/v1.0 dataset...")
    ds = load_dataset(
        "fever", "v1.0",
        cache_dir=hf_cache_dir,
        trust_remote_code=True,
    )

    # ── Pass 1: Collect needed titles ──────────────────────────
    logger.info("  Pass 1: Scanning annotations for needed wiki page titles...")
    needed = _collect_needed_titles(ds, max_train=max_train, max_dev=max_dev)
    logger.info(f"  Found {len(needed)} unique page titles in annotations")

    if not needed:
        logger.warning("  No page titles found — nothing to cache")
        return {"n_needed": 0, "n_found": 0, "n_missing": 0, "elapsed_s": 0,
                "cache_path": cache_path, "cache_size_mb": 0.0, "n_scanned": 0}

    # ── Pass 2: Stream wiki_pages, keep only needed ────────────
    # wiki_pages may be a separate config or a split within v1.0
    wiki_data = None
    if "wiki_pages" in ds:
        wiki_data = ds["wiki_pages"]
    elif "wikipedia_pages" in ds:
        wiki_data = ds["wikipedia_pages"]
    else:
        # Load wiki_pages as a separate HF config
        logger.info("  wiki_pages not in v1.0 — loading fever/wiki_pages separately...")
        try:
            wp_ds = load_dataset(
                "fever", "wiki_pages",
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            )
            # The split name may be 'wikipedia_pages' or 'train'
            for sp in ["wikipedia_pages", "wiki_pages", "train"]:
                if sp in wp_ds:
                    wiki_data = wp_ds[sp]
                    break
        except Exception as e:
            raise RuntimeError(
                f"Cannot load wiki_pages: {e}. "
                "Try: datasets.load_dataset('fever', 'wiki_pages')"
            )

    if wiki_data is None:
        raise RuntimeError("Could not find wiki_pages data in any expected location.")

    logger.info(f"  Pass 2: Streaming {len(wiki_data)} wiki pages, filtering to {len(needed)} needed titles...")

    # Create/overwrite SQLite DB
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    conn = sqlite3.connect(cache_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE wiki_pages (
            title TEXT PRIMARY KEY,
            sentences_json TEXT NOT NULL
        )
    """)

    n_scanned = 0
    n_found = 0
    batch = []
    batch_size = 1000

    for page in wiki_data:
        n_scanned += 1
        title = page.get("id", page.get("title", page.get("wikipedia_id", "")))
        if not title or title not in needed:
            continue

        lines_raw = page.get("lines", page.get("text", page.get("wikipedia_text", "")))
        if not lines_raw:
            continue

        sentences = _parse_wiki_sentences(lines_raw)
        if sentences:
            batch.append((title, json.dumps(sentences)))
            n_found += 1

        # Batch insert for performance
        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO wiki_pages (title, sentences_json) VALUES (?, ?)",
                batch,
            )
            conn.commit()
            batch = []
            if n_found % 5000 == 0:
                logger.info(
                    f"    Scanned {n_scanned} pages, found {n_found}/{len(needed)}..."
                )

        # Early exit if we found everything
        if n_found >= len(needed):
            logger.info(f"  Found all {n_found} needed pages after scanning {n_scanned}")
            break

    # Final batch
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO wiki_pages (title, sentences_json) VALUES (?, ?)",
            batch,
        )
        conn.commit()

    # Create index for fast lookup
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON wiki_pages(title)")
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    n_missing = len(needed) - n_found
    stats = {
        "n_needed": len(needed),
        "n_found": n_found,
        "n_missing": n_missing,
        "n_scanned": n_scanned,
        "elapsed_s": round(elapsed, 1),
        "cache_path": cache_path,
        "cache_size_mb": round(os.path.getsize(cache_path) / (1024 * 1024), 2),
    }

    logger.info(
        f"  ✅ Wiki cache built: {n_found}/{len(needed)} pages "
        f"({n_missing} missing) in {elapsed:.1f}s → {cache_path} "
        f"({stats['cache_size_mb']:.1f} MB)"
    )
    if n_missing > 0:
        logger.warning(
            f"  ⚠️  {n_missing} pages not found in wiki_pages split. "
            f"Evidence for those will use title-only fallback."
        )

    # Write manifest
    manifest_path = cache_path.replace(".db", "_manifest.json")
    cache = WikiCache(cache_path)
    manifest = {
        **stats,
        "titles_hash": cache_stats(cache_path)["titles_hash"],
    }
    cache.close()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Manifest written to {manifest_path}")

    return stats
