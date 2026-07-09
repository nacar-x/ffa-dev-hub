import json
import os
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path(os.getenv("RESOURCE_DATA_DIR", "data"))
DATA_FILE = DATA_DIR / "resources.json"


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class ResourceStore:
    """Stores posted resources (skripts/plugins/messages) and lets the bot
    search them with keyword search (BM25) so the AI can answer questions
    grounded in what people actually posted in the server."""

    def __init__(self, path: Path = DATA_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.resources = self._load()
        self._build_index()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.resources, f, indent=2, ensure_ascii=False)

    def _build_index(self):
        if self.resources:
            tokenized = [_tokenize(r["content"]) for r in self.resources]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def add(self, entry: dict) -> bool:
        if any(r["id"] == entry["id"] for r in self.resources):
            return False
        self.resources.append(entry)
        self._save()
        self._build_index()
        return True

    def count(self) -> int:
        return len(self.resources)

    def search(self, query: str, top_k: int = 4):
        if not self.bm25 or not self.resources:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.resources, scores), key=lambda x: x[1], reverse=True)
        return [r for r, s in ranked[:top_k] if s > 0]
