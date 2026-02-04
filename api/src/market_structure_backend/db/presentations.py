"""
SQLAlchemy database models for presentations.

Presentations allow analysts to combine figures and insights from multiple
model runs into a single client-ready HTML document.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    String, Text, DateTime, Integer, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .models import Base


class SlideType(str, Enum):
    """Type of presentation slide."""
    FIGURE = "figure"
    TEXT = "text"
    COMPARISON = "comparison"
    SUMMARY = "summary"


class Presentation(Base):
    """
    A multi-run presentation composed of slides.

    Presentations allow analysts to combine figures from multiple model runs
    with custom commentary and branding for client-ready deliverables.
    """
    __tablename__ = "presentations"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4())
    )

    # Presentation metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Client/project information
    client_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Branding options (colors, logo URL, theme)
    branding_options: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationship to slides
    slides: Mapped[List["PresentationSlide"]] = relationship(
        "PresentationSlide",
        back_populates="presentation",
        order_by="PresentationSlide.order",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Indexes
    __table_args__ = (
        Index("ix_presentations_created_at", "created_at"),
        Index("ix_presentations_name", "name"),
    )

    def to_dict(self) -> dict:
        """Convert presentation to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "client_name": self.client_name,
            "project_name": self.project_name,
            "branding_options": self.branding_options,
            "slides": [slide.to_dict() for slide in self.slides] if self.slides else [],
            "slide_count": len(self.slides) if self.slides else 0,
        }


class PresentationSlide(Base):
    """
    Individual slide within a presentation.

    Slides can contain figures from model runs, text content with markdown,
    or comparison layouts showing multiple figures side by side.
    """
    __tablename__ = "presentation_slides"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4())
    )

    # Foreign key to presentation
    presentation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("presentations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Slide ordering (0-indexed)
    order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Slide content
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    slide_type: Mapped[str] = mapped_column(String(50), nullable=False, default="figure")

    # Figure reference (for figure slides)
    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("model_runs.id", ondelete="SET NULL"),
        nullable=True
    )
    figure_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    figure_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Text content (for text slides or markdown commentary)
    text_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Layout options (width, alignment, etc.)
    layout: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationship back to presentation
    presentation: Mapped["Presentation"] = relationship(
        "Presentation",
        back_populates="slides"
    )

    # Indexes
    __table_args__ = (
        Index("ix_presentation_slides_presentation_id", "presentation_id"),
        Index("ix_presentation_slides_order", "order"),
        Index("ix_presentation_slides_model_run_id", "model_run_id"),
    )

    def to_dict(self) -> dict:
        """Convert slide to dictionary for API responses."""
        return {
            "id": self.id,
            "presentation_id": self.presentation_id,
            "order": self.order,
            "title": self.title,
            "description": self.description,
            "slide_type": self.slide_type,
            "model_run_id": self.model_run_id,
            "figure_type": self.figure_type,
            "figure_config": self.figure_config,
            "text_content": self.text_content,
            "layout": self.layout,
        }
