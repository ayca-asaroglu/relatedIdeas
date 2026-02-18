from __future__ import annotations

import os
from typing import BinaryIO, List, Optional
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session


class JiraDbRecord(BaseModel):
    """
    Jira kaydını temsil eder (MSSQL veya Excel kaynağından gelebilir).

    Not: Kolon adları her iki kaynakta da şu isimlerle eşleştirilir:
    - jira_key
    - summary
    - description
    """

    jira_key: Optional[str]
    summary: str
    description: Optional[str] = None


def fetch_jira_issues_from_db(db: Session) -> List[JiraDbRecord]:
    """
    MSSQL Jira veritabanından kayıtları çeker.

    - SQL sorgusu JIRA_BACKFILL_QUERY environment değişkeninden okunur.
    - Sorgu çıktısında en azından şu kolonlar olmalı:
        - jira_key
        - summary
        - description
      (gerekirse SELECT içinde alias vererek uyarlayabilirsin)
    """
    raw_query = os.getenv("JIRA_BACKFILL_QUERY")
    if not raw_query:
        raise RuntimeError(
            "JIRA_BACKFILL_QUERY environment değişkeni tanımlı değil. "
            "Örnek: export JIRA_BACKFILL_QUERY='SELECT JiraKey AS jira_key, "
            "Summary AS summary, Description AS description FROM JiraIssues'"
        )

    result = db.execute(text(raw_query))
    records: List[JiraDbRecord] = []

    for row in result.mappings():
        records.append(
            JiraDbRecord(
                jira_key=row.get("jira_key"),
                summary=row.get("summary"),
                description=row.get("description"),
            )
        )

    return records


def fetch_jira_issues_from_excel(f: BinaryIO) -> List[JiraDbRecord]:
    """
    Excel dosyasından Jira kayıtlarını okur.

    Beklenen kolon adları:
    - jira_key (opsiyonel)
    - summary (zorunlu)
    - description (opsiyonel)

    Excel'de farklı kolon isimleri varsa, dosyayı buna göre uyarlaman gerekir
    (örn. JiraKey -> jira_key, Summary -> summary).
    """
    df = pd.read_excel(f)

    missing = [col for col in ("summary",) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Excel dosyasında zorunlu kolon(lar) eksik: {', '.join(missing)}. "
            "En azından 'summary' kolonu olmalı."
        )

    records: List[JiraDbRecord] = []

    for _, row in df.iterrows():
        # summary boşsa kaydı atla
        if pd.isna(row.get("summary")):
            continue

        jira_key_val = row.get("jira_key")
        desc_val = row.get("description")

        records.append(
            JiraDbRecord(
                jira_key=None if pd.isna(jira_key_val) else str(jira_key_val),
                summary=str(row.get("summary")),
                description=None if pd.isna(desc_val) else str(desc_val),
            )
        )

    return records


def fetch_jira_issues_from_excel_path(path: str | Path) -> List[JiraDbRecord]:
    """
    Dosya sistemindeki bir Excel dosyasından Jira kayıtlarını okur.

    Örnek kullanım:
    - Excel dosyasını proje altında `data/jira_issues.xlsx` olarak koy
    - Sonra bu fonksiyona o path'i ver.
    """
    p = Path(path)
    with p.open("rb") as f:
        return fetch_jira_issues_from_excel(f)

