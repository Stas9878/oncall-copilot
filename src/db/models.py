"""Таблицы Postgres."""

import uuid
from enum import StrEnum
from datetime import date, datetime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func
)

# Предсказуемые имена ограничений, иначе Alembic не сможет их удалить в миграции.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Role(StrEnum):
    READER = "reader"
    EDITOR = "editor"


class DocumentKind(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    IMAGE = "image"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


def _str_enum(enum_cls: type[StrEnum]) -> Enum:
    # Значения храним строкой: новое значение не потребует ALTER TYPE в миграции.
    return Enum(
        enum_cls,
        native_enum=False,
        length=16,
        values_callable=lambda members: [member.value for member in members]
    )


class Tenant(Base):
    """Команда или компания-клиент. Все данные изолированы по tenant."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    monthly_token_budget: Mapped[int] = mapped_column(BigInteger, default=2_000_000)
    # Домены, на которые ответ может ссылаться. Остальные ссылки вырезаются.
    link_domains: Mapped[list[str]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    api_keys: Mapped[list[ApiKey]] = relationship(back_populates="tenant")


class ApiKey(Base):
    """Ключ доступа. Сам ключ не хранится, только его SHA-256."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[Role] = mapped_column(_str_enum(Role))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


class Document(Base):
    """Загруженный файл и состояние его индексации."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "sha256"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    kind: Mapped[DocumentKind] = mapped_column(_str_enum(DocumentKind))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(_str_enum(DocumentStatus), default=DocumentStatus.QUEUED)
    error: Mapped[str | None] = mapped_column(Text)
    # Версия разбиения и модель эмбеддингов, с которыми документ проиндексирован.
    index_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    sections: Mapped[list[Section]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Section(Base):
    """Раздел документа: его текст уходит в промпт, а в Qdrant лежат его фрагменты."""

    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(16))
    page_number: Mapped[int | None] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="sections")


class TokenUsage(Base):
    """Расход токенов облачных моделей по месяцам: из него считают квоту команды."""

    __tablename__ = "token_usage"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)
    month: Mapped[date] = mapped_column(Date, primary_key=True)
    model: Mapped[str] = mapped_column(String(100), primary_key=True)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
