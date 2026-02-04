"""
API routes for presentation management.

Presentations allow analysts to combine figures from multiple model runs
into client-ready HTML documents with custom commentary and branding.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...db import get_session, Presentation, PresentationSlide, ModelRun, ModelRunStatus
from ...schemas.api import (
    PresentationCreate,
    PresentationUpdate,
    PresentationResponse,
    PresentationListResponse,
    PresentationSlideCreate,
    PresentationSlideUpdate,
    PresentationSlideResponse,
    SlideReorderRequest,
    RunFiguresResponse,
    FigureDataResponse,
    FigureInfo,
    FigureTypeEnum,
    SlideTypeEnum,
    ExportFormatEnum,
    RevealThemeEnum,
    SlidePreviewResponse,
)
from ...services.figures import (
    get_available_figures,
    generate_figure,
    FigureType,
)

# Import the data extraction function from runs.py
from .runs import _extract_report_data

# Import clustering utilities
from market_structure.utils import (
    find_optimal_clusters,
    perform_kmeans_clustering,
    compute_hierarchical_clustering,
    get_hierarchical_labels,
)


router = APIRouter(prefix="/presentations", tags=["Presentations"])


def _perform_clustering(extracted_data: dict, n_clusters: int, method: str = "kmeans") -> dict:
    """Perform clustering on product embeddings.

    Args:
        extracted_data: Data extracted from model results (must contain product_embeddings)
        n_clusters: Number of clusters
        method: Clustering method ("kmeans" or "hierarchical")

    Returns:
        Dict with clustering results (labels, n_clusters, silhouette_scores, etc.)
    """
    embeddings = extracted_data.get("product_embeddings")
    if embeddings is None:
        raise ValueError("Product embeddings not available for clustering")

    n_products = embeddings.shape[0]
    max_k = min(10, n_products - 1)

    # Get optimal k analysis for silhouette plot
    optimal_result = find_optimal_clusters(embeddings, max_k=max_k)
    silhouette_scores = optimal_result["scores"]
    k_range = list(optimal_result["range"])
    optimal_k = optimal_result["optimal_k"]

    if method == "kmeans":
        cluster_result = perform_kmeans_clustering(embeddings, n_clusters)
        labels = cluster_result["labels"].tolist()
        silhouette_score = cluster_result.get("silhouette_score")
        linkage_matrix = None
    else:
        # Hierarchical clustering
        # Compute similarity matrix from embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings_norm = embeddings / norms
        similarity_matrix = embeddings_norm @ embeddings_norm.T

        hier_result = compute_hierarchical_clustering(similarity_matrix, method="ward")
        linkage_matrix = hier_result["linkage_matrix"].tolist()
        labels = get_hierarchical_labels(hier_result["linkage_matrix"], n_clusters).tolist()
        silhouette_score = None

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette_score,
        "silhouette_scores": silhouette_scores,
        "k_range": k_range,
        "optimal_k": optimal_k,
        "linkage_matrix": linkage_matrix,
    }
runs_router = APIRouter(prefix="/runs", tags=["Presentations"])


# =============================================================================
# PRESENTATION CRUD
# =============================================================================

@router.post("", response_model=PresentationResponse, status_code=201)
async def create_presentation(
    request: PresentationCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new presentation."""
    presentation = Presentation(
        id=str(uuid4()),
        name=request.name,
        description=request.description,
        client_name=request.client_name,
        project_name=request.project_name,
        branding_options=request.branding_options.model_dump() if request.branding_options else None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    session.add(presentation)
    await session.commit()
    await session.refresh(presentation)

    return _presentation_to_response(presentation)


@router.get("", response_model=PresentationListResponse)
async def list_presentations(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List all presentations."""
    # Get total count
    count_stmt = select(func.count(Presentation.id))
    total = await session.scalar(count_stmt)

    # Get presentations with slides
    stmt = (
        select(Presentation)
        .options(selectinload(Presentation.slides))
        .order_by(Presentation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    presentations = result.scalars().all()

    return PresentationListResponse(
        presentations=[_presentation_to_response(p) for p in presentations],
        total=total or 0
    )


@router.get("/{presentation_id}", response_model=PresentationResponse)
async def get_presentation(
    presentation_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a presentation by ID."""
    stmt = (
        select(Presentation)
        .options(selectinload(Presentation.slides))
        .where(Presentation.id == presentation_id)
    )
    result = await session.execute(stmt)
    presentation = result.scalar_one_or_none()

    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    return _presentation_to_response(presentation)


@router.put("/{presentation_id}", response_model=PresentationResponse)
async def update_presentation(
    presentation_id: str,
    request: PresentationUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update presentation metadata."""
    stmt = (
        select(Presentation)
        .options(selectinload(Presentation.slides))
        .where(Presentation.id == presentation_id)
    )
    result = await session.execute(stmt)
    presentation = result.scalar_one_or_none()

    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    # Update fields
    if request.name is not None:
        presentation.name = request.name
    if request.description is not None:
        presentation.description = request.description
    if request.client_name is not None:
        presentation.client_name = request.client_name
    if request.project_name is not None:
        presentation.project_name = request.project_name
    if request.branding_options is not None:
        presentation.branding_options = request.branding_options.model_dump()

    presentation.updated_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(presentation)

    return _presentation_to_response(presentation)


@router.delete("/{presentation_id}", status_code=204)
async def delete_presentation(
    presentation_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a presentation and all its slides."""
    presentation = await session.get(Presentation, presentation_id)

    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    await session.delete(presentation)
    await session.commit()

    return Response(status_code=204)


# =============================================================================
# SLIDE MANAGEMENT
# =============================================================================

@router.post("/{presentation_id}/slides", response_model=PresentationSlideResponse, status_code=201)
async def add_slide(
    presentation_id: str,
    request: PresentationSlideCreate,
    session: AsyncSession = Depends(get_session),
):
    """Add a new slide to a presentation."""
    presentation = await session.get(Presentation, presentation_id)
    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    # Validate model run if figure slide
    if request.slide_type == SlideTypeEnum.FIGURE and request.model_run_id:
        run = await session.get(ModelRun, request.model_run_id)
        if not run:
            raise HTTPException(404, f"Model run {request.model_run_id} not found")
        if run.status != ModelRunStatus.COMPLETED:
            raise HTTPException(400, f"Model run is not completed (status: {run.status})")

    # Determine order
    if request.order is not None:
        order = request.order
    else:
        # Get max order and add 1
        stmt = select(func.max(PresentationSlide.order)).where(
            PresentationSlide.presentation_id == presentation_id
        )
        max_order = await session.scalar(stmt)
        order = (max_order or -1) + 1

    slide = PresentationSlide(
        id=str(uuid4()),
        presentation_id=presentation_id,
        order=order,
        title=request.title,
        description=request.description,
        slide_type=request.slide_type.value,
        model_run_id=request.model_run_id,
        figure_type=request.figure_type.value if request.figure_type else None,
        figure_config=request.figure_config,
        text_content=request.text_content,
        layout=request.layout,
    )

    session.add(slide)

    # Update presentation timestamp
    presentation.updated_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(slide)

    return _slide_to_response(slide)


@router.put("/{presentation_id}/slides/{slide_id}", response_model=PresentationSlideResponse)
async def update_slide(
    presentation_id: str,
    slide_id: str,
    request: PresentationSlideUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update an existing slide."""
    slide = await session.get(PresentationSlide, slide_id)

    if not slide or slide.presentation_id != presentation_id:
        raise HTTPException(404, f"Slide {slide_id} not found in presentation {presentation_id}")

    # Validate model run if changing to figure slide
    if request.model_run_id is not None:
        run = await session.get(ModelRun, request.model_run_id)
        if not run:
            raise HTTPException(404, f"Model run {request.model_run_id} not found")
        if run.status != ModelRunStatus.COMPLETED:
            raise HTTPException(400, f"Model run is not completed (status: {run.status})")

    # Update fields
    if request.title is not None:
        slide.title = request.title
    if request.description is not None:
        slide.description = request.description
    if request.slide_type is not None:
        slide.slide_type = request.slide_type.value
    if request.model_run_id is not None:
        slide.model_run_id = request.model_run_id
    if request.figure_type is not None:
        slide.figure_type = request.figure_type.value
    if request.figure_config is not None:
        slide.figure_config = request.figure_config
    if request.text_content is not None:
        slide.text_content = request.text_content
    if request.layout is not None:
        slide.layout = request.layout
    if request.order is not None:
        slide.order = request.order

    # Update presentation timestamp
    presentation = await session.get(Presentation, presentation_id)
    if presentation:
        presentation.updated_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(slide)

    return _slide_to_response(slide)


@router.delete("/{presentation_id}/slides/{slide_id}", status_code=204)
async def delete_slide(
    presentation_id: str,
    slide_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a slide from a presentation."""
    slide = await session.get(PresentationSlide, slide_id)

    if not slide or slide.presentation_id != presentation_id:
        raise HTTPException(404, f"Slide {slide_id} not found in presentation {presentation_id}")

    await session.delete(slide)

    # Update presentation timestamp
    presentation = await session.get(Presentation, presentation_id)
    if presentation:
        presentation.updated_at = datetime.now(timezone.utc)

    await session.commit()

    return Response(status_code=204)


@router.put("/{presentation_id}/slides/reorder", response_model=PresentationResponse)
async def reorder_slides(
    presentation_id: str,
    request: SlideReorderRequest,
    session: AsyncSession = Depends(get_session),
):
    """Reorder slides in a presentation."""
    stmt = (
        select(Presentation)
        .options(selectinload(Presentation.slides))
        .where(Presentation.id == presentation_id)
    )
    result = await session.execute(stmt)
    presentation = result.scalar_one_or_none()

    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    # Validate all slide IDs belong to this presentation
    slide_ids_set = {s.id for s in presentation.slides}
    for slide_id in request.slide_ids:
        if slide_id not in slide_ids_set:
            raise HTTPException(400, f"Slide {slide_id} does not belong to this presentation")

    # Update order
    for new_order, slide_id in enumerate(request.slide_ids):
        for slide in presentation.slides:
            if slide.id == slide_id:
                slide.order = new_order
                break

    presentation.updated_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(presentation)

    return _presentation_to_response(presentation)


# =============================================================================
# FIGURE ENDPOINTS (on runs router)
# =============================================================================

@runs_router.get("/{run_id}/figures", response_model=RunFiguresResponse)
async def get_run_figures(
    run_id: str,
    include_clustering: bool = Query(default=False, description="Include clustering figures if clustering has been run"),
    n_clusters: Optional[int] = Query(default=None, ge=2, le=20, description="Number of clusters (required if include_clustering=True)"),
    clustering_method: str = Query(default="kmeans", description="Clustering method: kmeans or hierarchical"),
    session: AsyncSession = Depends(get_session),
):
    """Get available figure types for a model run.

    If include_clustering is True and n_clusters is provided, clustering will be
    performed and clustering figures will be included in the available figures.
    """
    run = await session.get(ModelRun, run_id)
    if not run:
        raise HTTPException(404, f"Model run {run_id} not found")

    if run.status != ModelRunStatus.COMPLETED:
        raise HTTPException(400, f"Model run is not completed (status: {run.status})")

    if not run.results_path:
        raise HTTPException(404, "Results not available")

    # Load results
    results_path = Path(run.results_path)
    if not results_path.exists():
        raise HTTPException(404, "Results file not found")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
    product_columns = run.product_columns or []

    # Extract data for figure availability check
    extracted_data = _extract_report_data(results, model_type, product_columns)

    # Get clustering results if requested
    clustering_result = None
    if include_clustering and n_clusters is not None:
        clustering_result = _perform_clustering(extracted_data, n_clusters, clustering_method)

    # Get available figures
    available = get_available_figures(model_type, extracted_data, clustering_result)

    return RunFiguresResponse(
        model_run_id=run_id,
        model_type=run.model_type,
        product_columns=product_columns,
        available_figures=[
            FigureInfo(
                type=FigureTypeEnum(fig["type"]),
                name=fig["name"],
                description=fig["description"],
                available=fig["available"]
            )
            for fig in available
        ]
    )


@runs_router.get("/{run_id}/figures/{figure_type}", response_model=FigureDataResponse)
async def get_figure_data(
    run_id: str,
    figure_type: FigureTypeEnum,
    n_clusters: Optional[int] = Query(default=None, ge=2, le=20, description="Number of clusters for clustering figures"),
    clustering_method: str = Query(default="kmeans", description="Clustering method: kmeans or hierarchical"),
    session: AsyncSession = Depends(get_session),
):
    """Get a specific figure as Plotly JSON.

    For clustering figures (clustered_biplot, silhouette_analysis, cluster_sizes, dendrogram),
    you must provide n_clusters parameter.
    """
    run = await session.get(ModelRun, run_id)
    if not run:
        raise HTTPException(404, f"Model run {run_id} not found")

    if run.status != ModelRunStatus.COMPLETED:
        raise HTTPException(400, f"Model run is not completed (status: {run.status})")

    if not run.results_path:
        raise HTTPException(404, "Results not available")

    # Load results
    results_path = Path(run.results_path)
    if not results_path.exists():
        raise HTTPException(404, "Results file not found")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
    product_columns = run.product_columns or []

    # Extract data
    extracted_data = _extract_report_data(results, model_type, product_columns)

    # Check if clustering is required for this figure type
    clustering_figure_types = ["clustered_biplot", "silhouette_analysis", "cluster_sizes", "dendrogram"]
    clustering_result = None
    if figure_type.value in clustering_figure_types:
        if n_clusters is None:
            raise HTTPException(400, f"n_clusters parameter is required for {figure_type.value} figure")
        clustering_result = _perform_clustering(extracted_data, n_clusters, clustering_method)

    # Generate figure
    try:
        fig = generate_figure(
            FigureType(figure_type.value),
            extracted_data,
            config=None,
            clustering_result=clustering_result
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to generate figure: {str(e)}")

    return FigureDataResponse(
        model_run_id=run_id,
        figure_type=figure_type,
        figure_json=fig.to_dict()
    )


# =============================================================================
# EXPORT ENDPOINT
# =============================================================================

@router.get("/{presentation_id}/export")
async def export_presentation(
    presentation_id: str,
    format: ExportFormatEnum = Query(default=ExportFormatEnum.REVEALJS, description="Export format: revealjs (modern slide-based) or html (legacy scroll-based)"),
    session: AsyncSession = Depends(get_session),
):
    """Export presentation as HTML.

    Supports two formats:
    - revealjs: Modern slide-based presentation with keyboard navigation (default)
    - html: Legacy scroll-based HTML document
    """
    stmt = (
        select(Presentation)
        .options(selectinload(Presentation.slides))
        .where(Presentation.id == presentation_id)
    )
    result = await session.execute(stmt)
    presentation = result.scalar_one_or_none()

    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    # Generate HTML based on format
    try:
        if format == ExportFormatEnum.REVEALJS:
            from ...services.presentation_generator_revealjs import generate_revealjs_presentation
            html_content = await generate_revealjs_presentation(presentation, session)
            suffix = "_slides"
        else:
            from ...services.presentation_generator import generate_presentation_html
            html_content = await generate_presentation_html(presentation, session)
            suffix = "_document"
    except Exception as e:
        raise HTTPException(500, f"Failed to generate presentation: {str(e)}")

    # Return as HTML file
    filename = f"{presentation.name.replace(' ', '_')}{suffix}.html"

    return Response(
        content=html_content,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# =============================================================================
# SLIDE PREVIEW ENDPOINT
# =============================================================================

@router.get("/{presentation_id}/slides/{slide_id}/preview", response_model=SlidePreviewResponse)
async def get_slide_preview(
    presentation_id: str,
    slide_id: str,
    theme: RevealThemeEnum = Query(default=RevealThemeEnum.WHITE, description="Reveal.js theme for preview"),
    primary_color: str = Query(default="#667eea", description="Primary brand color"),
    secondary_color: str = Query(default="#764ba2", description="Secondary brand color"),
    session: AsyncSession = Depends(get_session),
):
    """Get a standalone HTML preview for a single slide.

    Returns HTML that can be embedded in an iframe for live preview during editing.
    """
    # Verify presentation exists
    presentation = await session.get(Presentation, presentation_id)
    if not presentation:
        raise HTTPException(404, f"Presentation {presentation_id} not found")

    # Get the slide
    slide = await session.get(PresentationSlide, slide_id)
    if not slide or slide.presentation_id != presentation_id:
        raise HTTPException(404, f"Slide {slide_id} not found in presentation {presentation_id}")

    # Generate single-slide preview
    try:
        from ...services.presentation_generator_revealjs import generate_single_slide_preview
        html_content = await generate_single_slide_preview(
            slide=slide,
            session=session,
            theme=theme.value,
            primary_color=primary_color,
            secondary_color=secondary_color,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate slide preview: {str(e)}")

    return SlidePreviewResponse(
        slide_id=slide_id,
        html=html_content,
        theme=theme,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _presentation_to_response(presentation: Presentation) -> PresentationResponse:
    """Convert Presentation model to response schema."""
    slides = sorted(presentation.slides, key=lambda s: s.order) if presentation.slides else []

    return PresentationResponse(
        id=presentation.id,
        name=presentation.name,
        description=presentation.description,
        created_at=presentation.created_at,
        updated_at=presentation.updated_at,
        client_name=presentation.client_name,
        project_name=presentation.project_name,
        branding_options=presentation.branding_options,
        slides=[_slide_to_response(s) for s in slides],
        slide_count=len(slides)
    )


def _slide_to_response(slide: PresentationSlide) -> PresentationSlideResponse:
    """Convert PresentationSlide model to response schema."""
    return PresentationSlideResponse(
        id=slide.id,
        presentation_id=slide.presentation_id,
        order=slide.order,
        title=slide.title,
        description=slide.description,
        slide_type=SlideTypeEnum(slide.slide_type),
        model_run_id=slide.model_run_id,
        figure_type=FigureTypeEnum(slide.figure_type) if slide.figure_type else None,
        figure_config=slide.figure_config,
        text_content=slide.text_content,
        layout=slide.layout,
    )
