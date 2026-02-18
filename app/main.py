from typing import List
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

# .env dosyasını yükle
load_dotenv()

from .db import get_db
from .embeddings import SimpleHashEmbeddingProvider
from .jira_source import (
    JiraDbRecord,
    fetch_jira_issues_from_db,
    fetch_jira_issues_from_excel,
    fetch_jira_issues_from_excel_path,
)
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
        "Not: Jira kayıtlarının asıl kaynağı MSSQL veya Excel olabilir; bu servis "
        "Jira kayıtlarını bu kaynaklardan okuyup embedding'leri dosya sisteminde saklar."
    ),
    version="0.3.0",
)


def get_retriever() -> JiraRetriever:
    """
    Embedding'leri dosya sisteminde tutan retriever.
    
    Embedding provider seçimi:
    - EMBEDDING_PROVIDER env değişkeni ile kontrol edilir:
        - "sentence-transformers" veya "st" -> SentenceTransformerProvider (önerilen)
        - "openai" -> OpenAIEmbeddingProvider
        - "azure" -> AzureOpenAIEmbeddingProvider
        - Yoksa veya "hash" -> SimpleHashEmbeddingProvider (demo)
    
    Örnek:
        export EMBEDDING_PROVIDER="sentence-transformers"
        export EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
    """
    from .embeddings import (
        SimpleHashEmbeddingProvider,
        SentenceTransformerProvider,
        OpenAIEmbeddingProvider,
        AzureOpenAIEmbeddingProvider,
    )
    
    provider_type = os.getenv("EMBEDDING_PROVIDER", "hash").lower()
    
    if provider_type in ("sentence-transformers", "st", "sentence_transformer"):
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        embedder = SentenceTransformerProvider(model_name=model_name)
    elif provider_type == "openai":
        model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
        embedder = OpenAIEmbeddingProvider(model=model)
    elif provider_type == "azure":
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")
        embedder = AzureOpenAIEmbeddingProvider(deployment_name=deployment_name)
    else:
        # Varsayılan: basit hash (demo için)
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


@app.post(
    "/issues/backfill-from-db",
    response_model=List[JiraIssueOut],
    tags=["issues"],
)
def backfill_from_mssql(
    retriever: JiraRetriever = Depends(get_retriever),
    db: Session = Depends(get_db),
):
    """
    MSSQL Jira veritabanından kayıtları okuyup embedding indeksine ekler.

    - MSSQL bağlantı bilgisi `MSSQL_URL` environment değişkeninden gelir.
    - Hangi kayıtların çekileceği `JIRA_BACKFILL_QUERY` ile belirlenir.
      Örnek:

        export JIRA_BACKFILL_QUERY='
          SELECT
            JiraKey AS jira_key,
            Summary AS summary,
            Description AS description
          FROM JiraIssues
        '
    """
    records: List[JiraDbRecord] = fetch_jira_issues_from_db(db)
    items = [
        JiraIssueCreate(
            jira_key=rec.jira_key,
            summary=rec.summary,
            description=rec.description,
        )
        for rec in records
    ]
    issues = retriever.bulk_index(items)
    return issues


@app.post(
    "/issues/backfill-from-excel",
    response_model=List[JiraIssueOut],
    tags=["issues"],
)
async def backfill_from_excel(
    file: UploadFile = File(...),
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Excel dosyasından Jira kayıtlarını okuyup embedding indeksine ekler.

    Beklenen Excel kolonları:
    - jira_key (opsiyonel)
    - summary (zorunlu)
    - description (opsiyonel)

    Örnek akış:
    - Jira'dan export aldığın Excel dosyasını bu endpoint'e upload et
    - Servis her satır için embedding üretip `data/embeddings` klasörüne yazar
    """
    records = fetch_jira_issues_from_excel(file.file)
    items = [
        JiraIssueCreate(
            jira_key=rec.jira_key,
            summary=rec.summary,
            description=rec.description,
        )
        for rec in records
    ]
    issues = retriever.bulk_index(items)
    return issues


@app.post(
    "/issues/backfill-from-excel-file",
    response_model=List[JiraIssueOut],
    tags=["issues"],
)
def backfill_from_excel_file(
    retriever: JiraRetriever = Depends(get_retriever),
    path: str | None = None,
):
    """
    Sunucu tarafındaki bir Excel dosyasından Jira kayıtlarını okuyup
    embedding indeksine ekler (upload gerekmez).

    - Excel dosyasını `app` klasörü altına `jira_issues.xlsx` ismiyle koyarsan,
      herhangi bir parametre veya environment değişkeni vermeden bu endpoint'i
      çağırman yeterli olur.

    - İsteğe bağlı olarak:
        1) `path` parametresi ile tam/relatif path verebilirsin:
           `/issues/backfill-from-excel-file?path=app/jira_issues.xlsx`
        2) `JIRA_EXCEL_PATH` environment değişkenini tanımlayabilirsin:

               export JIRA_EXCEL_PATH="app/jira_issues.xlsx"

    Beklenen kolonlar:
    - jira_key (opsiyonel)
    - summary (zorunlu)
    - description (opsiyonel)
    """
    default_path = Path(__file__).resolve().parent / "jira_issues.xlsx"
    env_path = os.getenv("JIRA_EXCEL_PATH")
    excel_path_str = path or env_path or str(default_path)
    excel_path = Path(excel_path_str)

    if not excel_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Excel dosyası bulunamadı: {excel_path}",
        )

    records = fetch_jira_issues_from_excel_path(excel_path)
    items = [
        JiraIssueCreate(
            jira_key=rec.jira_key,
            summary=rec.summary,
            description=rec.description,
        )
        for rec in records
    ]
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
    try:
        query_issue, similar, debug_info = retriever.find_similar(
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
            query_issue=JiraIssueOut(
                id=query_issue.id,
                jira_key=query_issue.jira_key,
                summary=query_issue.summary,
                description=query_issue.description,
            ),
            similar_issues=similar,
            debug_info=debug_info,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benzerlik araması sırasında hata: {str(e)}",
        )


@app.get("/issues", response_model=List[JiraIssueOut], tags=["issues"])
def list_issues(
    retriever: JiraRetriever = Depends(get_retriever),
):
    """
    Kayıtlı tüm embedding index kayıtlarını listele (debug / kontrol amaçlı).
    """
    return retriever.list_indexed()

