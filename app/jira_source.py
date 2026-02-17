from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session


class JiraDbRecord(BaseModel):
    """
    MSSQL'den çekilen Jira kaydını temsil eder.

    Not: Kolon adları sorgu ile eşleştirilir; JIRA_BACKFILL_QUERY içinde
    `AS jira_key`, `AS summary`, `AS description` alias'larını vermek gerekir.
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

