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
    ) -> Tuple[StoredIssue, List[SimilarIssue], dict]:
        """
        Verilen talebe göre benzer talepleri bulur.

        - query talebi için embedding hesaplanır
        - ÖNCE mevcut kayıtlarla karşılaştırma yapılır
        - store_query=True ise embedding klasöre kaydedilir (karşılaştırmadan SONRA)

        Returns:
            (query_issue, similar_issues, debug_info)
        """
        text = f"{query.summary}\n{query.description or ''}"
        query_vec = self.embedder.embed(text)

        # ÖNCE mevcut kayıtları al (sorgu kaydı henüz kaydedilmedi)
        all_issues = self.store.all()
        
        # Şimdi sorgu kaydını oluştur (henüz kaydetmedik)
        query_issue_temp = StoredIssue(
            id="temp_query",
            jira_key=query.jira_key,
            summary=query.summary,
            description=query.description,
            embedding=query_vec,
        )
        total_count = len(all_issues)

        candidates: List[SimilarIssue] = []
        checked_count = 0
        passed_min_score_count = 0
        all_scores = []

        # Mevcut tüm kayıtlarla karşılaştır (sorgu kaydı henüz kaydedilmedi, bu yüzden filtreleme gerekmez)
        for issue in all_issues:
            checked_count += 1
            score = cosine_similarity(query_vec, issue.embedding)
            all_scores.append(score)

            if score < min_score:
                continue

            passed_min_score_count += 1
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
        
        # ŞİMDİ sorgu kaydını kaydet (eğer isteniyorsa) - karşılaştırmadan SONRA
        if store_query:
            query_issue = self.store.save(
                jira_key=query.jira_key,
                summary=query.summary,
                description=query.description,
                embedding=query_vec,
            )
        else:
            query_issue = query_issue_temp
        
        debug_info = {
            "total_issues_in_index": total_count,
            "checked_issues": checked_count,
            "passed_min_score": passed_min_score_count,
            "min_score_threshold": min_score,
            "max_score_found": max(all_scores) if all_scores else 0.0,
            "min_score_found": min(all_scores) if all_scores else 0.0,
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        }

        return query_issue, candidates[:top_k], debug_info

