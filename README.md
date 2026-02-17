# Jira Benzerlik Servisi (RAG tabanlı)

Bu servis, Jira taleplerinin `summary` ve `description` alanlarından vektör (embedding) üretip
mevcut talepler arasında **anlamsal benzerlik** araması yapar. 

Basit bir demo için harici LLM/embedding servisi yerine deterministik bir pseudo-embedding kullanır,
ancak mimari olarak OpenAI, Azure, Google vb. gerçek embedding modelleri ile rahatça değiştirilebilir.

## Teknolojiler

- FastAPI (REST API)
- Dosya tabanlı embedding store (`data/embeddings/*.json`)
- Basit embedding sağlayıcı (ileride gerçek LLM embedding ile değiştirilebilir)

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn app.main:app --reload
```

Ardından Swagger UI üzerinden endpoint’leri inceleyebilirsiniz:

- `http://localhost:8000/docs`

## Temel Akış

1. Var olan Jira taleplerini (summary + description) bu servise gönderip embedding’leri dosya sistemine (`data/embeddings`) indekslersiniz.
2. Yeni bir Jira talebi geldiğinde, bu servisi çağırıp benzer talepleri sorgularsınız.
3. Servis hem benzer talepleri döner, hem de isterseniz gelen talebi de embedding ile birlikte klasöre kaydeder.

## Önemli Endpoint'ler

### Health Check

- **GET** `/health`

### Tekil Jira talebi kaydetme (embedding oluşturma)

- **POST** `/issues`

Body örneği:

```json
{
  "jira_key": "PROJ-1",
  "summary": "Login sayfasında şifre reset çalışmıyor",
  "description": "Kullanıcı şifremi unuttum linkine tıkladığında 500 hatası alıyor."
}
```

### Çoklu Jira talebi indeksleme (backfill)

- **POST** `/issues/bulk`

Body örneği:

```json
[
  {
    "jira_key": "PROJ-1",
    "summary": "Login sayfasında şifre reset çalışmıyor",
    "description": "Kullanıcı şifremi unuttum linkine tıkladığında 500 hatası alıyor."
  },
  {
    "jira_key": "PROJ-2",
    "summary": "Profil sayfası yavaş yükleniyor",
    "description": "Profil sayfası ilk açılışta 10 saniyeden uzun sürüyor."
  }
]
```

### Benzer Jira taleplerini sorgulama (semantic search)

- **POST** `/similar-issues`

Body örneği:

```json
{
  "jira_key": "PROJ-NEW",
  "summary": "Şifre sıfırlama linki hata veriyor",
  "description": "Şifre yenilemek isteyince sistem hata döndürüyor.",
  "top_k": 5,
  "min_score": 0.6,
  "store_issue": true
}
```

Dönen cevap örneği:

```json
{
  "query_issue": {
    "jira_key": "PROJ-NEW",
    "summary": "Şifre sıfırlama linki hata veriyor",
    "description": "Şifre yenilemek isteyince sistem hata döndürüyor."
  },
  "similar_issues": [
    {
      "id": 1,
      "jira_key": "PROJ-1",
      "summary": "Login sayfasında şifre reset çalışmıyor",
      "description": "Kullanıcı şifremi unuttum linkine tıkladığında 500 hatası alıyor.",
      "score": 0.87
    }
  ]
}
```

## RAG ile Genişletme

Bu proje RAG mimarisi için **retrieval** kısmını hazırlar:

- Jira taleplerinin embedding’leri dosya sisteminde (`data/embeddings`) tutulur.
- Benzer talepler cosine similarity ile bulunur.

Bir üst aşamada, bu benzer talepler ve Jira alanları bir LLM’e context olarak verilip:

- Otomatik çözüm önerisi,
- İlgili eski ticket’ların özetlenmesi,
- Yeni ticket için otomatik açıklama/öneri oluşturma

gibi yetenekler eklenebilir.

Bu durumda `app/embeddings.py` ve benzer bir `llm_client.py` içinde OpenAI/Azure/Google vb.
entegrasyonları yapmanız yeterlidir.

