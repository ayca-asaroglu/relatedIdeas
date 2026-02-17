from typing import List, Tuple

from .embeddings import EmbeddingProvider, cosine_similarity
from .schemas import JiraIssueCreate, SimilarIssue
from .storage import FileEmbeddingStore, StoredIssue


class JiraRetriever:
    """
    Jira taleplerini embedding'ler üzerinden benzerlik ile arayan sınıf.

    Not: Jira kayıtlarının asıl kaynağı MSSQL olabilir; bu sınıf yalnızca
    embedding tarafını yönetir ve embedding'leri dosya sisteminde saklar.
    """

    def __init__(self, embedder: EmbeddingProvider, store: FileEmbeddingStore):
        self.embedder = embedder
        self.store = store

    def index_issue(self, data: JiraIssueCreate) -> StoredIssue:
        """
        Tek bir Jira talebini embedding ile birlikte indekse ekle.
        Embedding'ler veritabanına değil, klasöre yazılır.
        """
        text = f"{data.summary}\n{data.description or ''}"
        vec = self.embedder.embed(text)
        return self.store.save(
            jira_key=data.jira_key,
            summary=data.summary,
            description=data.description,
            embedding=vec,
        )

    def bulk_index(self, items: List[JiraIssueCreate]) -> List[StoredIssue]:
        """
        Çoklu Jira talebini embedding ile indekse ekler.
        """
        indexed: List[StoredIssue] = []
        for item in items:
            indexed.append(self.index_issue(item))
        return indexed

    def list_indexed(self) -> List[StoredIssue]:
        """
        Klasördeki tüm embedding kayıtlarını getirir.
        """
        return self.store.all()

    def find_similar(
        self,
        query: JiraIssueCreate,
        top_k: int = 5,
        min_score: float = 0.6,
        store_query: bool = True,
    ) -> Tuple[StoredIssue, List[SimilarIssue]]:
        """
        Verilen talebe göre benzer talepleri bulur.

        - query talebi için embedding hesaplanır
        - store_query=True ise embedding klasöre kaydedilir
        - klasördeki diğer kayıtlarla cosine similarity hesaplanır
        """
        text = f"{query.summary}\n{query.description or ''}"
        query_vec = self.embedder.embed(text)

        if store_query:
            query_issue = self.store.save(
                jira_key=query.jira_key,
                summary=query.summary,
                description=query.description,
                embedding=query_vec,
            )
        else:
            query_issue = StoredIssue(
                id="memory",
                jira_key=query.jira_key,
                summary=query.summary,
                description=query.description,
                embedding=query_vec,
            )

        all_issues = self.store.all()

        candidates: List[SimilarIssue] = []
        for issue in all_issues:
            if issue.id == query_issue.id:
                # Sorgu talebinin kendisini listeye eklemeyelim
                continue

            score = cosine_similarity(query_vec, issue.embedding)
            if score < min_score:
                continue
            candidates.append(
                SimilarIssue(
                    id=issue.id,
                    jira_key=issue.jira_key,
                    summary=issue.summary,
                    description=issue.description,
                    score=score,
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return query_issue, candidates[:top_k]

