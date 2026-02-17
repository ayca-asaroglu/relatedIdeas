from typing import List

from fastapi import Depends, FastAPI

from .embeddings import SimpleHashEmbeddingProvider
from .retrieval import JiraRetriever
from .schemas import (
    JiraIssueCreate,
    JiraIssueOut,
    SimilarIssue,
    SimilarityRequest,
    SimilarityResponse,
)
from .storage import FileEmbeddingStore

app = FastAPI(
    title="Jira Benzerlik Servisi",
    description=(
        "Jira taleplerinin summary/description alanlarından embedding üreten ve "
        "benzer talepleri RAG mantığı ile getiren servis.\n\n"
        "Not: Jira kayıtlarının asıl kaynağı MSSQL olabilir; bu servis şu anda "
        "sadece embedding'leri dosya sisteminde saklar."
    ),
    version="0.2.0",
)


def get_retriever() -> JiraRetriever:
    """
    DB bağımlılığını kaldırıp embedding'leri dosya sisteminde tutan retriever.
    """
    embedder = SimpleHashEmbeddingProvider(dim=128)
    store = FileEmbeddingStore(base_dir="data/embeddings")
    return JiraRetriever(embedder=embedder, store=store)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@app.post("/issues", response_model=JiraIssueOut, tags=["issues"])
def create_issue(
    issue_in: JiraIssueCreate,
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Tek bir Jira talebini embedding ile birlikte kaydet.

    Embedding'ler veritabanına değil, `data/embeddings` klasörüne yazılır.
    Jira'nın asıl kayıtları MSSQL'de tutulabilir; bu endpoint sadece
    benzerlik araması için gerekli vektör indeksini oluşturur.
    """
    issue = retriever.index_issue(issue_in)
    return issue


@app.post("/issues/bulk", response_model=List[JiraIssueOut], tags=["issues"])
def bulk_create_issues(
    items: List[JiraIssueCreate],
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Çoklu Jira talebini embedding ile indekse ekle.

    Örn. MSSQL Jira veritabanınızdan okuyup bu servise POST ederek
    başlangıç embedding indeksini doldurabilirsiniz.
    """
    issues = retriever.bulk_index(items)
    return issues


@app.post("/similar-issues", response_model=SimilarityResponse, tags=["similarity"])
def find_similar_issues(
    payload: SimilarityRequest,
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Yeni bir Jira talebine göre mevcut embedding kayıtlarında benzer talepler var mı
    kontrol eder.

    - summary + description alanlarından embedding üretir
    - klasördeki embedding'lerle cosine similarity hesaplar
    - `min_score` üzerindekilerden en iyi `top_k` sonucu döner
    - `store_issue=True` ise bu talebi de embedding ile birlikte klasöre kaydeder
    """
    query_issue, similar = retriever.find_similar(
        JiraIssueCreate(
            jira_key=payload.jira_key,
            summary=payload.summary,
            description=payload.description,
        ),
        top_k=payload.top_k,
        min_score=payload.min_score,
        store_query=payload.store_issue,
    )
    return SimilarityResponse(
        query_issue=query_issue,
        similar_issues=similar,
    )


@app.get("/issues", response_model=List[JiraIssueOut], tags=["issues"])
def list_issues(
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Kayıtlı tüm embedding index kayıtlarını listele (debug / kontrol amaçlı).
    """
    return retriever.list_indexed()

