from typing import List, Optional

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class JiraIssue(Base):
    __tablename__ = "jira_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    jira_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True, nullable=True)
    summary: Mapped[str] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Embedding'i virgülle ayrılmış string olarak saklayacağız (küçük hacimler için basit çözüm).
    embedding: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    def embedding_as_vector(self) -> Optional[List[float]]:
        if self.embedding is None:
            return None
        try:
            return [float(v) for v in self.embedding.split(",") if v]
        except ValueError:
            return None

    @staticmethod
    def vector_to_string(vec: List[float]) -> str:
        # Embedding'i compact bir string olarak sakla
        return ",".join(f"{v:.8f}" for v in vec)

