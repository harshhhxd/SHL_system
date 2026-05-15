import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set
from sentence_transformers import SentenceTransformer
import faiss

ALWAYS_INCLUDE_PATTERNS: List[str] = [
    "OPQ32r",
    "Motivation Questionnaire",
    "Verify G+",
    "Verify - Numerical Ability",
    "Verify - Verbal Ability",
    "Verify Interactive - Numerical",
    "Verify Interactive - Verbal",
]

# keywords that should trigger pulling ALL matching catalog items by name
TECH_KEYWORDS = [
    "java", "python", "javascript", "sql", "c++", "c#", ".net",
    "react", "angular", "node", "php", "ruby", "swift", "kotlin",
    "excel", "powerpoint", "word", "sap", "salesforce",
]


def _name_matches_patterns(name: str, patterns: List[str]) -> bool:
    name_lower = name.lower()
    return any(p.lower() in name_lower for p in patterns)


def build_search_text(item: dict) -> str:
    test_types = " ".join(item.get("test_types", []))
    job_levels = " ".join(item.get("job_levels", []))
    languages  = " ".join(item.get("languages", []))
    fields = [
        item.get("name", ""),
        item.get("description", ""),
        test_types,
        job_levels,
        languages,
    ]
    return " ".join(str(f) for f in fields if f)


class CatalogRetriever:
    def __init__(self, catalog_path: str = "shl_catalog.json"):
        with open(Path(catalog_path), encoding="utf-8") as f:
            raw = json.load(f)

        self.catalog: List[Dict] = raw["individual_test_solutions"]
        self.valid_urls: Set[str] = {item["url"] for item in self.catalog}

        self._broad: List[Dict] = [
            item for item in self.catalog
            if _name_matches_patterns(item["name"], ALWAYS_INCLUDE_PATTERNS)
        ]

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_index()

    def _build_index(self):
        texts = [build_search_text(item) for item in self.catalog]
        embeddings = self.encoder.encode(
            texts, show_progress_bar=False
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"[retriever] indexed {len(self.catalog)} assessments")

    def _extract_tech_keywords(self, query: str) -> List[str]:
        # return tech words found in query that have dedicated catalog tests
        q_lower = query.lower()
        return [kw for kw in TECH_KEYWORDS if kw in q_lower]

    def _tech_matches(self, query: str) -> List[Dict]:
        # pull every catalog item whose name contains the detected tech keyword
        hits = []
        seen = set()
        for kw in self._extract_tech_keywords(query):
            for item in self.catalog:
                if kw in item["name"].lower() and item["url"] not in seen:
                    hits.append(item)
                    seen.add(item["url"])
        return hits

    def _keyword_score(self, query: str, item: dict) -> int:
        # count how many query words (len>=3) appear in item name
        q_words = {w.lower() for w in query.split() if len(w) >= 3}
        name_lower = item["name"].lower()
        return sum(1 for w in q_words if w in name_lower)

    def search(self, query: str, top_k: int = 12) -> List[Dict]:
        # FAISS semantic search with a bigger fetch window
        fetch_k = min(top_k * 3, len(self.catalog))
        q = self.encoder.encode([query]).astype(np.float32)
        faiss.normalize_L2(q)
        _, indices = self.index.search(q, fetch_k)
        candidates = [self.catalog[i] for i in indices[0]]

        # score by keyword overlap and sort descending
        candidates.sort(key=lambda item: self._keyword_score(query, item), reverse=True)

        # inject tech-specific items at the front, deduplicated
        tech_hits = self._tech_matches(query)
        seen_urls = {item["url"] for item in tech_hits}
        non_tech = [c for c in candidates if c["url"] not in seen_urls]

        merged = tech_hits + non_tech
        return merged[:top_k]

    def ensure_broad_assessments(self, items: List[Dict]) -> List[Dict]:
        seen_urls = {item["url"] for item in items}
        extras = [b for b in self._broad if b["url"] not in seen_urls]
        return items + extras

    def is_valid_url(self, url: str) -> bool:
        return url in self.valid_urls