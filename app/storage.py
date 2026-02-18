from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class StoredIssue(BaseModel):
    """
    Dosya sisteminde tutulan embedding kaydı.

    Not: Buradaki `id` bizim oluşturduğumuz teknik bir kimlik.
    Asıl iş kimliği olarak `jira_key` kullanılabilir.
    
    Embedding NPY formatında saklanır (daha verimli).
    """

    id: str
    jira_key: Optional[str]
    summary: str
    description: Optional[str]
    embedding: List[float]


class FileEmbeddingStore:
    """
    Embedding'leri veritabanı yerine klasörde hibrit format olarak saklar:
    - Metadata (jira_key, summary, description) -> JSON
    - Embedding vektörü -> NPY (NumPy binary format, çok daha verimli)
    
    Avantajlar:
    - NPY formatı JSON'dan 3-5x daha küçük dosya boyutu
    - 10-50x daha hızlı okuma/yazma
    - NumPy ile doğrudan uyumlu
    - Metadata hala okunabilir (JSON)
    """

    def __init__(self, base_dir: str | Path = "data/embeddings") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, issue_id: str) -> Path:
        """Metadata için JSON dosyası yolu."""
        return self.base_dir / f"{issue_id}.json"

    def _embedding_path(self, issue_id: str) -> Path:
        """Embedding için NPY dosyası yolu."""
        return self.base_dir / f"{issue_id}.npy"

    def save(
        self,
        *,
        jira_key: Optional[str],
        summary: str,
        description: Optional[str],
        embedding: List[float],
    ) -> StoredIssue:
        issue_id = str(uuid.uuid4())
        
        # Metadata'yı JSON olarak kaydet
        metadata = {
            "id": issue_id,
            "jira_key": jira_key,
            "summary": summary,
            "description": description,
        }
        metadata_path = self._metadata_path(issue_id)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        
        # Embedding'i NPY olarak kaydet (çok daha verimli)
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_path = self._embedding_path(issue_id)
        np.save(embedding_path, embedding_array)
        
        record = StoredIssue(
            id=issue_id,
            jira_key=jira_key,
            summary=summary,
            description=description,
            embedding=embedding,  # Memory'de List[float] olarak tutuyoruz
        )
        return record

    def all(self) -> List[StoredIssue]:
        """
        Tüm kayıtları yükler. Metadata JSON'dan, embedding NPY'den okunur.
        Geriye dönük uyumluluk: Eski JSON formatındaki dosyalar atlanır.
        """
        issues: List[StoredIssue] = []
        
        # Tüm metadata JSON dosyalarını bul
        for metadata_path in sorted(self.base_dir.glob("*.json")):
            try:
                # Metadata'yı oku
                raw = metadata_path.read_text(encoding="utf-8")
                metadata = json.loads(raw)
                issue_id = metadata.get("id")
                
                # Eski format kontrolü: Eğer JSON içinde embedding varsa, yeni formata geçiş yapılmamış
                if "embedding" in metadata:
                    # Eski format - skip et (geriye dönük uyumluluk)
                    continue
                
                if not issue_id:
                    continue
                
                # Embedding'i NPY'den oku
                embedding_path = self._embedding_path(issue_id)
                if not embedding_path.exists():
                    # NPY dosyası yoksa skip et
                    continue
                
                embedding_array = np.load(embedding_path)
                embedding = embedding_array.tolist()
                
                issues.append(
                    StoredIssue(
                        id=issue_id,
                        jira_key=metadata.get("jira_key"),
                        summary=metadata["summary"],
                        description=metadata.get("description"),
                        embedding=embedding,
                    )
                )
            except Exception:
                # Bozuk dosya varsa atla, sistemi bozmasın.
                continue
        
        return issues

