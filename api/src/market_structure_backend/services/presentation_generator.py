"""
HTML presentation generator.

Generates standalone HTML presentations with:
- Navigation sidebar with table of contents
- Slide sections with figures and text
- Prev/Next navigation
- Custom branding options
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional
import markdown
import numpy as np

import plotly.io as pio
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Presentation, PresentationSlide, ModelRun
from .figures import generate_figure, FigureType

# Import clustering utilities
from market_structure.utils import (
    find_optimal_clusters,
    perform_kmeans_clustering,
    compute_hierarchical_clustering,
    get_hierarchical_labels,
)


async def generate_presentation_html(
    presentation: Presentation,
    session: AsyncSession
) -> str:
    """
    Generate standalone HTML for a presentation.

    Args:
        presentation: Presentation model with slides loaded
        session: Database session for loading model run data

    Returns:
        Complete HTML string
    """
    # Extract branding options
    branding = presentation.branding_options or {}
    primary_color = branding.get("primary_color", "#667eea")
    secondary_color = branding.get("secondary_color", "#764ba2")
    logo_url = branding.get("logo_url")

    # Sort slides by order
    slides = sorted(presentation.slides, key=lambda s: s.order)

    # Generate slide content
    slide_html_parts = []
    toc_items = []

    for idx, slide in enumerate(slides):
        slide_num = idx + 1
        slide_id = f"slide-{slide_num}"

        # Add to TOC
        toc_items.append(f'''
            <li>
                <a href="#{slide_id}" class="toc-link" data-slide="{slide_num}">
                    <span class="toc-number">{slide_num}</span>
                    <span class="toc-title">{_escape_html(slide.title)}</span>
                </a>
            </li>
        ''')

        # Generate slide content
        slide_content = await _generate_slide_content(slide, session)

        slide_html = f'''
        <section id="{slide_id}" class="slide slide-{slide.slide_type}">
            <div class="slide-header">
                <h2 class="slide-title">{_escape_html(slide.title)}</h2>
                {f'<p class="slide-description">{_escape_html(slide.description)}</p>' if slide.description else ''}
            </div>
            <div class="slide-content">
                {slide_content}
            </div>
            <div class="slide-nav">
                <button onclick="navigateSlide(-1)" class="nav-btn" {' disabled' if idx == 0 else ''}>
                    ← Previous
                </button>
                <span class="slide-counter">{slide_num} / {len(slides)}</span>
                <button onclick="navigateSlide(1)" class="nav-btn" {' disabled' if idx == len(slides) - 1 else ''}>
                    Next →
                </button>
            </div>
        </section>
        '''
        slide_html_parts.append(slide_html)

    # Build full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(presentation.name)}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {_get_css(primary_color, secondary_color)}
    </style>
</head>
<body>
    <nav class="toc-sidebar">
        <div class="toc-header">
            {f'<img src="{logo_url}" alt="Logo" class="toc-logo">' if logo_url else ''}
            <div class="toc-branding">
                {f'<span class="client-name">{_escape_html(presentation.client_name)}</span>' if presentation.client_name else ''}
                {f'<span class="project-name">{_escape_html(presentation.project_name)}</span>' if presentation.project_name else ''}
            </div>
        </div>
        <ul class="toc-list">
            {''.join(toc_items)}
        </ul>
        <div class="toc-footer">
            <span class="generated-date">Generated {datetime.now().strftime('%B %d, %Y')}</span>
        </div>
    </nav>

    <main class="presentation-content">
        <header class="presentation-header">
            <h1>{_escape_html(presentation.name)}</h1>
            {f'<p class="presentation-description">{_escape_html(presentation.description)}</p>' if presentation.description else ''}
            <div class="header-meta">
                {f'<span class="meta-client">{_escape_html(presentation.client_name)}</span>' if presentation.client_name else ''}
                {f'<span class="meta-divider">|</span>' if presentation.client_name and presentation.project_name else ''}
                {f'<span class="meta-project">{_escape_html(presentation.project_name)}</span>' if presentation.project_name else ''}
            </div>
        </header>

        {''.join(slide_html_parts)}

        <footer class="presentation-footer">
            <p>Generated with Market Structure Analysis Tool</p>
        </footer>
    </main>

    <script>
        {_get_javascript()}
    </script>
</body>
</html>
'''
    return html


async def _generate_slide_content(
    slide: PresentationSlide,
    session: AsyncSession
) -> str:
    """Generate HTML content for a single slide."""
    if slide.slide_type == "text":
        # Render markdown content
        if slide.text_content:
            md_html = markdown.markdown(
                slide.text_content,
                extensions=['tables', 'fenced_code', 'nl2br']
            )
            return f'<div class="text-slide-content">{md_html}</div>'
        return '<div class="text-slide-content"><p>No content</p></div>'

    if slide.slide_type == "figure":
        if not slide.model_run_id or not slide.figure_type:
            return '<div class="figure-error">Figure configuration incomplete</div>'

        # Load model run and results
        run = await session.get(ModelRun, slide.model_run_id)
        if not run or not run.results_path:
            return '<div class="figure-error">Model run not found or results unavailable</div>'

        results_path = Path(run.results_path)
        if not results_path.exists():
            return '<div class="figure-error">Results file not found</div>'

        try:
            with open(results_path, "rb") as f:
                results = pickle.load(f)

            # Import here to avoid circular import
            from ..api.routes.runs import _extract_report_data

            model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
            product_columns = run.product_columns or []

            extracted_data = _extract_report_data(results, model_type, product_columns)

            # Check if clustering is needed for this figure type
            clustering_result = None
            clustering_figure_types = ["clustered_biplot", "silhouette_analysis", "cluster_sizes", "dendrogram"]
            if slide.figure_type in clustering_figure_types:
                # Get clustering parameters from figure_config
                config = slide.figure_config or {}
                n_clusters = config.get("n_clusters")
                clustering_method = config.get("clustering_method", "kmeans")

                if n_clusters:
                    clustering_result = _perform_clustering_for_presentation(
                        extracted_data, n_clusters, clustering_method
                    )

            # Generate figure
            fig = generate_figure(
                FigureType(slide.figure_type),
                extracted_data,
                config=slide.figure_config,
                clustering_result=clustering_result
            )

            # Convert to HTML
            fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

            # Add run info
            run_info = f'''
            <div class="figure-source">
                <span class="source-label">Source:</span>
                <span class="source-run">{_escape_html(run.name or f"Run {run.id[:8]}")}</span>
                <span class="source-type">({model_type})</span>
            </div>
            '''

            return f'''
            <div class="figure-container">
                {fig_html}
                {run_info}
            </div>
            '''
        except Exception as e:
            return f'<div class="figure-error">Error generating figure: {_escape_html(str(e))}</div>'

    if slide.slide_type == "summary":
        # Summary slides can have both text and minimal figures
        content_parts = []
        if slide.text_content:
            md_html = markdown.markdown(slide.text_content, extensions=['tables', 'nl2br'])
            content_parts.append(f'<div class="summary-text">{md_html}</div>')
        return ''.join(content_parts) or '<div class="summary-content"><p>Summary slide</p></div>'

    if slide.slide_type == "comparison":
        # Comparison slides - for now just render as text
        if slide.text_content:
            md_html = markdown.markdown(slide.text_content, extensions=['tables', 'nl2br'])
            return f'<div class="comparison-content">{md_html}</div>'
        return '<div class="comparison-content"><p>Comparison slide</p></div>'

    return '<div class="unknown-slide">Unknown slide type</div>'


def _perform_clustering_for_presentation(extracted_data: dict, n_clusters: int, method: str = "kmeans") -> dict:
    """Perform clustering on product embeddings for presentation figures.

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


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _get_css(primary_color: str, secondary_color: str) -> str:
    """Get CSS styles for the presentation."""
    return f'''
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        /* TOC Sidebar */
        .toc-sidebar {{
            position: fixed;
            left: 0;
            top: 0;
            bottom: 0;
            width: 280px;
            background: linear-gradient(180deg, {primary_color} 0%, {secondary_color} 100%);
            color: white;
            padding: 20px;
            overflow-y: auto;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }}

        .toc-header {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}

        .toc-logo {{
            max-width: 150px;
            max-height: 60px;
            margin-bottom: 10px;
        }}

        .toc-branding {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .client-name {{
            font-weight: 600;
            font-size: 14px;
        }}

        .project-name {{
            font-size: 12px;
            opacity: 0.8;
        }}

        .toc-list {{
            list-style: none;
            flex: 1;
        }}

        .toc-list li {{
            margin-bottom: 8px;
        }}

        .toc-link {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
            text-decoration: none;
            padding: 8px 12px;
            border-radius: 6px;
            transition: background 0.2s;
            font-size: 14px;
        }}

        .toc-link:hover,
        .toc-link.active {{
            background: rgba(255,255,255,0.2);
        }}

        .toc-number {{
            background: rgba(255,255,255,0.2);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
        }}

        .toc-title {{
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .toc-footer {{
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
            font-size: 11px;
            opacity: 0.7;
        }}

        /* Main Content */
        .presentation-content {{
            margin-left: 280px;
            min-height: 100vh;
            padding: 40px;
        }}

        .presentation-header {{
            background: linear-gradient(135deg, {primary_color} 0%, {secondary_color} 100%);
            color: white;
            padding: 60px 40px;
            border-radius: 16px;
            margin-bottom: 40px;
            text-align: center;
        }}

        .presentation-header h1 {{
            font-size: 2.5rem;
            margin-bottom: 16px;
        }}

        .presentation-description {{
            font-size: 1.1rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto 20px;
        }}

        .header-meta {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}

        .meta-divider {{
            margin: 0 10px;
        }}

        /* Slides */
        .slide {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .slide-header {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid {primary_color}20;
        }}

        .slide-title {{
            font-size: 1.8rem;
            color: {primary_color};
            margin-bottom: 10px;
        }}

        .slide-description {{
            color: #666;
            font-size: 1rem;
        }}

        .slide-content {{
            min-height: 400px;
        }}

        .slide-nav {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}

        .nav-btn {{
            background: {primary_color};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: opacity 0.2s;
        }}

        .nav-btn:hover {{
            opacity: 0.9;
        }}

        .nav-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}

        .slide-counter {{
            color: #666;
            font-size: 14px;
        }}

        /* Figure slides */
        .figure-container {{
            width: 100%;
        }}

        .figure-source {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #888;
        }}

        .source-label {{
            margin-right: 5px;
        }}

        .source-run {{
            font-weight: 600;
            color: {primary_color};
        }}

        .source-type {{
            margin-left: 5px;
        }}

        .figure-error {{
            background: #fff5f5;
            border: 1px solid #feb2b2;
            color: #c53030;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        /* Text slides */
        .text-slide-content {{
            font-size: 1.1rem;
            line-height: 1.8;
        }}

        .text-slide-content h1,
        .text-slide-content h2,
        .text-slide-content h3 {{
            color: {primary_color};
            margin: 20px 0 10px;
        }}

        .text-slide-content ul,
        .text-slide-content ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}

        .text-slide-content li {{
            margin-bottom: 8px;
        }}

        .text-slide-content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .text-slide-content th,
        .text-slide-content td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}

        .text-slide-content th {{
            background: {primary_color}10;
            font-weight: 600;
        }}

        /* Footer */
        .presentation-footer {{
            text-align: center;
            padding: 40px;
            color: #888;
            font-size: 14px;
        }}

        /* Print styles */
        @media print {{
            .toc-sidebar {{
                display: none;
            }}

            .presentation-content {{
                margin-left: 0;
            }}

            .slide {{
                page-break-inside: avoid;
                page-break-after: always;
            }}

            .slide-nav {{
                display: none;
            }}
        }}

        /* Responsive */
        @media (max-width: 1024px) {{
            .toc-sidebar {{
                width: 60px;
                padding: 10px;
            }}

            .toc-header,
            .toc-title,
            .toc-footer {{
                display: none;
            }}

            .toc-link {{
                justify-content: center;
                padding: 10px;
            }}

            .presentation-content {{
                margin-left: 60px;
                padding: 20px;
            }}
        }}
    '''


def _get_javascript() -> str:
    """Get JavaScript for presentation interactivity."""
    return '''
        // Track current slide
        let currentSlide = 1;
        const totalSlides = document.querySelectorAll('.slide').length;

        // Navigate to a specific slide
        function navigateTo(slideNum) {
            if (slideNum < 1 || slideNum > totalSlides) return;

            const slideElement = document.getElementById('slide-' + slideNum);
            if (slideElement) {
                slideElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                currentSlide = slideNum;
                updateActiveTocItem();
            }
        }

        // Navigate relative to current slide
        function navigateSlide(delta) {
            navigateTo(currentSlide + delta);
        }

        // Update active TOC item
        function updateActiveTocItem() {
            document.querySelectorAll('.toc-link').forEach((link, idx) => {
                link.classList.toggle('active', idx + 1 === currentSlide);
            });
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === ' ') {
                e.preventDefault();
                navigateSlide(1);
            } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                e.preventDefault();
                navigateSlide(-1);
            } else if (e.key === 'Home') {
                e.preventDefault();
                navigateTo(1);
            } else if (e.key === 'End') {
                e.preventDefault();
                navigateTo(totalSlides);
            }
        });

        // Track scroll position to update current slide
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const slideId = entry.target.id;
                    const slideNum = parseInt(slideId.split('-')[1]);
                    if (!isNaN(slideNum)) {
                        currentSlide = slideNum;
                        updateActiveTocItem();
                    }
                }
            });
        }, { threshold: 0.5 });

        document.querySelectorAll('.slide').forEach(slide => {
            observer.observe(slide);
        });

        // Initialize
        updateActiveTocItem();
    '''
