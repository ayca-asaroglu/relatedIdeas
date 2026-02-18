"""
Embedding sağlayıcı arabirimi.

Farklı embedding yöntemleri:
1. SimpleHashEmbeddingProvider - Basit hash-based (demo için)
2. SentenceTransformerProvider - Sentence Transformers (önerilen, ücretsiz, lokal)
3. OpenAIEmbeddingProvider - OpenAI API (ücretli, yüksek kalite)
4. AzureOpenAIEmbeddingProvider - Azure OpenAI (kurumsal)

Kullanım:
    # Sentence Transformers (önerilen)
    embedder = SentenceTransformerProvider(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # OpenAI
    embedder = OpenAIEmbeddingProvider(api_key="...", model="text-embedding-3-small")
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os

import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class SimpleHashEmbeddingProvider(EmbeddingProvider):
    """
    Küçük demo/prototip için, kelime hash'lerinden sabit boyutlu bir vektör üretir.
    Anlamsal kalite gerçek LLM embedding'lerine göre zayıftır ama akış aynıdır.
    
    NOT: Gerçek kullanım için SentenceTransformerProvider kullanılmalı!
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


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Sentence Transformers kullanarak embedding üretir.
    
    Ücretsiz, lokal çalışır, Türkçe desteği var.
    
    Popüler modeller:
    - paraphrase-multilingual-MiniLM-L12-v2 (hızlı, küçük)
    - paraphrase-multilingual-mpnet-base-v2 (daha kaliteli, daha yavaş)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    
    Kurulum:
        pip install sentence-transformers
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except ImportError:
            raise ImportError(
                "sentence-transformers kurulu değil. Kurmak için: pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        if not text:
            # Boş metin için model'in embedding boyutuna göre sıfır vektör
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Sentence Transformers otomatik olarak normalize edilmiş embedding döner
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI Embeddings API kullanarak embedding üretir.
    
    Modeller:
    - text-embedding-3-small (384 dim, ucuz)
    - text-embedding-3-large (3072 dim, daha kaliteli)
    - text-embedding-ada-002 (1536 dim, eski)
    
    Kurulum:
        pip install openai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        organization: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                organization=organization,
            )
            self.model = model
        except ImportError:
            raise ImportError(
                "openai kurulu değil. Kurmak için: pip install openai"
            )
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key gerekli. OPENAI_API_KEY env değişkenini ayarla veya api_key parametresini ver."
            )

    def embed(self, text: str) -> List[float]:
        if not text:
            # Model'e göre varsayılan boyut
            return [0.0] * 384  # text-embedding-3-small için
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Azure OpenAI Embeddings kullanarak embedding üretir.
    
    Environment değişkenleri:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_API_VERSION (opsiyonel, varsayılan: 2024-02-15-preview)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "text-embedding-3-small",
    ):
        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
            self.deployment_name = deployment_name
        except ImportError:
            raise ImportError(
                "openai kurulu değil. Kurmak için: pip install openai"
            )

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384
        
        response = self.client.embeddings.create(
            model=self.deployment_name,
            input=text,
        )
        return response.data[0].embedding


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
