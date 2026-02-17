"""
Bu modül, ileride Jira kayıtlarının tutulduğu MSSQL veritabanına bağlanmak
için kullanılmak üzere bırakıldı.

Şu anda servis:
- Jira kayıtlarını dışarıdan (senin uygulamandan veya MSSQL'den) alıyor,
- sadece embedding'leri dosya sisteminde (`data/embeddings` klasörü) saklıyor.

MSSQL entegrasyonunu eklemek istediğinde, örnek bir SQLAlchemy bağlantısı:

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, DeclarativeBase

    class Base(DeclarativeBase):
        pass

    MSSQL_URL = (
        "mssql+pyodbc://USER:PASSWORD@HOST:1433/DBNAME"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )

    engine = create_engine(MSSQL_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def get_db():
        from sqlalchemy.orm import Session

        db: Session = SessionLocal()
        try:
            yield db
        finally:
            db.close()

Bu kod şu an kullanılmıyor; sadece MSSQL tarafını bağlamak istediğinde referans
olsun diye bırakıldı.
"""

