"""BM25 retriever for FEVER full-pipeline evaluation.

Retrieves evidence sentences from a Wikipedia sentence store using BM25.
Supports deterministic caching to disk for reproducibility.

LEAKAGE GUARD: This retriever operates on a Wikipedia dump / sentence store.
It NEVER has access to FEVER gold evidence annotations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from typing import Any

logger = logging.getLogger("bm25_retriever")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """BM25-based evidence retriever for FEVER pipeline mode.

    Uses rank_bm25 for scoring. Supports caching to disk.

    Args:
        sentence_store: list of (doc_title, sent_idx, sentence_text) tuples.
        cache_dir: directory for caching retrieval results.
        top_k_docs: number of top documents to consider.
        top_k_sents: number of top sentences to return per claim.
    """

    def __init__(
        self,
        sentence_store: list[tuple[str, int, str]] | None = None,
        cache_dir: str = ".cache/bm25",
        top_k_docs: int = 5,
        top_k_sents: int = 5,
    ):
        self.cache_dir = cache_dir
        self.top_k_docs = top_k_docs
        self.top_k_sents = top_k_sents
        self.sentence_store = sentence_store or []
        self._bm25 = None
        self._tokenized_corpus: list[list[str]] = []

        if sentence_store:
            self._build_index(sentence_store)

    def _build_index(self, sentence_store: list[tuple[str, int, str]]) -> None:
        """Build BM25 index from sentence store."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25  (required for BM25 retrieval)")

        self.sentence_store = sentence_store
        texts = [sent for _, _, sent in sentence_store]
        self._tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(sentence_store)} sentences")

    def retrieve(
        self,
        claim: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve top-k evidence sentences for a claim.

        Args:
            claim: the claim text to find evidence for.
            top_k: override for top_k_sents.

        Returns:
            List of dicts with keys: title, sent_idx, text, score.
        """
        if self._bm25 is None:
            return []

        k = top_k or self.top_k_sents
        query_tokens = _tokenize(claim)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            idx = int(idx)
            title, sent_idx, text = self.sentence_store[idx]
            results.append({
                "title": title,
                "sent_idx": sent_idx,
                "text": text,
                "score": float(scores[idx]),
            })

        return results

    def retrieve_batch(
        self,
        claims: list[str],
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """Retrieve evidence for a batch of claims."""
        return [self.retrieve(claim, top_k) for claim in claims]

    def retrieve_as_text(
        self,
        claim: str,
        top_k: int | None = None,
        separator: str = " . ",
    ) -> str:
        """Retrieve evidence and concatenate as a single text string."""
        results = self.retrieve(claim, top_k)
        return separator.join(r["text"] for r in results)

    def retrieve_all_cached(
        self,
        items: list[dict],
        cache_name: str = "fever_retrieved",
    ) -> dict[int, str]:
        """Retrieve evidence for all items with disk caching.

        Args:
            items: list of dicts with 'id' and 'claim' keys.
            cache_name: name for the cache file.

        Returns:
            Dict mapping item id → retrieved evidence text.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.json")

        # Try loading from cache
        if os.path.exists(cache_path):
            logger.info(f"Loading cached retrievals from {cache_path}")
            with open(cache_path) as f:
                cached = json.load(f)
            # Convert string keys back to int
            return {int(k): v for k, v in cached.items()}

        # Retrieve
        logger.info(f"Retrieving evidence for {len(items)} claims...")
        retrieved = {}
        for i, item in enumerate(items):
            evidence_text = self.retrieve_as_text(item["claim"])
            retrieved[item["id"]] = evidence_text
            if (i + 1) % 1000 == 0:
                logger.info(f"  Retrieved {i+1}/{len(items)}")

        # Cache to disk
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in retrieved.items()}, f)
        logger.info(f"Cached retrievals to {cache_path}")

        return retrieved

    def save_index(self, path: str) -> None:
        """Save the BM25 index to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "sentence_store": self.sentence_store,
                "top_k_docs": self.top_k_docs,
                "top_k_sents": self.top_k_sents,
            }, f)

    @classmethod
    def load_index(cls, path: str) -> "BM25Retriever":
        """Load a saved BM25 index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls(
            sentence_store=data["sentence_store"],
            top_k_docs=data.get("top_k_docs", 5),
            top_k_sents=data.get("top_k_sents", 5),
        )
        return retriever


def build_sentence_store_from_jsonl(
    wiki_pages_path: str,
    max_pages: int | None = None,
) -> list[tuple[str, int, str]]:
    """Build sentence store from FEVER wiki-pages JSONL dump.

    Each line in the JSONL has: {"id": title, "lines": "0\\tsentence0\\n1\\tsentence1\\n..."}

    Args:
        wiki_pages_path: path to wiki-pages.jsonl
        max_pages: limit number of pages (for testing)

    Returns:
        List of (title, sent_idx, sentence_text) tuples.
    """
    store = []
    count = 0
    with open(wiki_pages_path) as f:
        for line in f:
            if max_pages and count >= max_pages:
                break
            page = json.loads(line)
            title = page.get("id", "")
            lines = page.get("lines", "")
            if not lines:
                continue
            for sent_line in lines.split("\n"):
                parts = sent_line.split("\t")
                if len(parts) >= 2:
                    try:
                        sent_idx = int(parts[0])
                    except ValueError:
                        continue
                    sent_text = parts[1].strip()
                    if sent_text:
                        store.append((title, sent_idx, sent_text))
            count += 1

    logger.info(f"Built sentence store: {len(store)} sentences from {count} pages")
    return store


def build_synthetic_sentence_store(
    items: list[dict],
    noise_sentences: int = 50,
) -> list[tuple[str, int, str]]:
    """Build a minimal sentence store from gold evidence for testing.

    WARNING: This is ONLY for smoke-testing the retrieval pipeline.
    For real evaluation, use build_sentence_store_from_jsonl with the
    full Wikipedia dump. Results using this store are NOT legitimate.

    The store includes:
    1. Gold evidence sentences (split by '. ')
    2. Random noise sentences for distraction
    """
    import random

    store = []
    seen = set()

    # Add evidence from items
    for item in items:
        evidence = item.get("gold_evidence_text", "")
        claim = item.get("claim", "")
        if evidence:
            for i, sent in enumerate(evidence.split(" . ")):
                sent = sent.strip()
                if sent and sent not in seen:
                    seen.add(sent)
                    store.append(("evidence_doc", i, sent))

        # Also add the claim itself as a potential sentence (for entity overlap)
        if claim and claim not in seen:
            seen.add(claim)
            store.append(("claim_doc", 0, claim))

    # Add some generic noise
    noise = [
        "The sun is a star at the center of the solar system.",
        "Water covers approximately 71 percent of the Earth's surface.",
        "The speed of light is approximately 299792458 meters per second.",
        "DNA stands for deoxyribonucleic acid.",
        "The Great Wall of China is over 13000 miles long.",
        "Python is a high-level programming language.",
        "The Amazon River is the longest river in South America.",
        "Mount Everest is the tallest mountain above sea level.",
        "The human body contains approximately 206 bones.",
        "Shakespeare wrote Romeo and Juliet in the 1590s.",
    ]
    for i, sent in enumerate(noise[:noise_sentences]):
        if sent not in seen:
            store.append(("noise_doc", i, sent))

    random.shuffle(store)
    return store
