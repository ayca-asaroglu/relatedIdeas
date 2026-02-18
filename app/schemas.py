from typing import List, Optional

from pydantic import BaseModel, Field


class JiraIssueBase(BaseModel):
    jira_key: Optional[str] = Field(None, description="Jira issue key, örn. PROJ-123")
    summary: str
    description: Optional[str] = None


class JiraIssueCreate(JiraIssueBase):
    pass


class JiraIssueOut(JiraIssueBase):
    # Dosya tabanlı saklama için id'yi string yapıyoruz (UUID).
    id: str


class SimilarIssue(JiraIssueOut):
    score: float = Field(..., description="0-1 arası benzerlik skoru (1 en benzer)")


class SimilarityRequest(JiraIssueBase):
    top_k: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum benzerlik skoru (varsayılan 0.3, basit embedding'ler için düşük tutulmalı)")
    store_issue: bool = Field(
        True, description="Bu istekle gelen talebi de embedding hesaplayıp klasöre kaydet"
    )


class SimilarityResponse(BaseModel):
    query_issue: JiraIssueOut
    similar_issues: List[SimilarIssue]
    debug_info: dict = Field(
        default_factory=dict,
        description="Debug bilgileri: toplam kayıt sayısı, kontrol edilen kayıt sayısı vb."
    )

