"""
Embedding sağlayıcı arabirimi.

Burada basit ve deterministik bir pseudo-embedding kullanıyoruz ki
harici bir LLM/embedding servisine ihtiyaç olmasın.

Gerçek sistemde bu sınıfı OpenAI, Azure, Google vb. ile değiştirebilirsiniz.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class SimpleHashEmbeddingProvider(EmbeddingProvider):
    """
    Küçük demo/prototip için, kelime hash’lerinden sabit boyutlu bir vektör üretir.
    Anlamsal kalite gerçek LLM embedding'lerine göre zayıftır ama akış aynıdır.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec.tolist()

        for token in text.lower().split():
            h = hash(token)
            idx = h % self.dim
            vec[idx] += 1.0

        # L2 normalizasyon
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

