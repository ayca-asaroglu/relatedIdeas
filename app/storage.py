from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class StoredIssue(BaseModel):
    """
    Dosya sisteminde tutulan embedding kaydı.

    Not: Buradaki `id` bizim oluşturduğumuz teknik bir kimlik.
    Asıl iş kimliği olarak `jira_key` kullanılabilir.
    """

    id: str
    jira_key: Optional[str]
    summary: str
    description: Optional[str]
    embedding: List[float]


class FileEmbeddingStore:
    """
    Embedding'leri veritabanı yerine klasörde JSON dosyaları olarak saklar.
    """

    def __init__(self, base_dir: str | Path = "data/embeddings") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, issue_id: str) -> Path:
        return self.base_dir / f"{issue_id}.json"

    def save(
        self,
        *,
        jira_key: Optional[str],
        summary: str,
        description: Optional[str],
        embedding: List[float],
    ) -> StoredIssue:
        issue_id = str(uuid.uuid4())
        record = StoredIssue(
            id=issue_id,
            jira_key=jira_key,
            summary=summary,
            description=description,
            embedding=embedding,
        )
        path = self._file_path(issue_id)
        path.write_text(record.model_dump_json(), encoding="utf-8")
        return record

    def all(self) -> List[StoredIssue]:
        issues: List[StoredIssue] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                issues.append(StoredIssue.model_validate(data))
            except Exception:
                # Bozuk dosya varsa atla, sistemi bozmasın.
                continue
        return issues

