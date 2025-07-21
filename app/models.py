from sqlalchemy import Column, Integer, String, LargeBinary, TIMESTAMP, ForeignKey, Enum, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from app.database import Base
import enum
import uuid

class UploadStatus(enum.Enum):
    """Status states for uploaded document processing."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"

class Upload(Base):
    __tablename__ = "uploads"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    status = Column(Enum(UploadStatus, name="upload_status"),
                    default=UploadStatus.PENDING,
                    nullable=False)

    pages = relationship(
        "Page",
        back_populates="upload",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class Page(Base):
    __tablename__ = "pages"
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(PG_UUID(as_uuid=True),
                       ForeignKey("uploads.id", ondelete="CASCADE"),
                       nullable=False,
                       index=True)
    page_number = Column(Integer, nullable=False)
    img_bytes = Column(LargeBinary, nullable=False)

    upload = relationship("Upload", back_populates="pages")
    redacted_pages = relationship(
        "RedactedPage",
        back_populates="page",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class RedactedPage(Base):
    __tablename__ = "redacted_pages"
    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer,
                     ForeignKey("pages.id", ondelete="CASCADE"),
                     nullable=False,
                     index=True)
    redacted_bytes = Column(LargeBinary, nullable=False)

    page = relationship("Page", back_populates="redacted_pages")
