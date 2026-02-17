"""
MSSQL tarafındaki Jira veritabanına bağlantı.

Bu modül:
- SQLAlchemy ile MSSQL'e bağlanır (pyodbc sürücüsü ile),
- FastAPI endpoint'lerinde dependency injection ile kullanılmak üzere `get_db`
  fonksiyonunu sağlar.

ÖNEMLİ:
- Gerçek bağlantı string'ini `MSSQL_URL` environment değişkeni ile vermelisin.
- Örnek:

    export MSSQL_URL="mssql+pyodbc://USER:PASSWORD@HOST:1433/DBNAME?driver=ODBC+Driver+17+for+SQL+Server"

"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """İleride ORM modelleri tanımlamak istersen kullanılabilir."""


MSSQL_URL = os.getenv(
    "MSSQL_URL",
    # Kullanıcı kendi ortamında bu değeri env ile override etmeli.
    "mssql+pyodbc://USER:PASSWORD@HOST:1433/DBNAME"
    "?driver=ODBC+Driver+17+for+SQL+Server",
)

engine = create_engine(
    MSSQL_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    FastAPI dependency: MSSQL Session nesnesi üretir.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

