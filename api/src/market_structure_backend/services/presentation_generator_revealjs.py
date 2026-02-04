"""
Reveal.js presentation generator.

Generates professional slide-based presentations using reveal.js framework with:
- Theme support (black, white, league, sky, beige, etc.)
- Transition effects (slide, fade, convex, concave, zoom)
- Interactive Plotly figures
- Keyboard navigation
- Speaker notes support
"""

import json
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


# Reveal.js CDN URLs
REVEALJS_VERSION = "4.6.1"
REVEALJS_CSS = f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/reveal.min.css"
REVEALJS_JS = f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/reveal.min.js"
PLOTLY_JS = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Available themes
REVEAL_THEMES = {
    "black": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/black.min.css",
    "white": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/white.min.css",
    "league": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/league.min.css",
    "beige": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/beige.min.css",
    "sky": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/sky.min.css",
    "night": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/night.min.css",
    "serif": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/serif.min.css",
    "simple": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/simple.min.css",
    "solarized": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/solarized.min.css",
    "moon": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/moon.min.css",
    "dracula": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/dracula.min.css",
    "blood": f"https://cdnjs.cloudflare.com/ajax/libs/reveal.js/{REVEALJS_VERSION}/theme/blood.min.css",
}


async def generate_revealjs_presentation(
    presentation: Presentation,
    session: AsyncSession
) -> str:
    """
    Generate a reveal.js presentation as standalone HTML.

    Args:
        presentation: Presentation model with slides loaded
        session: Database session for loading model run data

    Returns:
        Complete HTML string for reveal.js presentation
    """
    # Extract branding/config options
    branding = presentation.branding_options or {}
    theme = branding.get("theme", "white")
    transition = branding.get("transition", "slide")
    primary_color = branding.get("primary_color", "#667eea")
    secondary_color = branding.get("secondary_color", "#764ba2")
    logo_url = branding.get("logo_url")  # Legacy, for title slide
    client_logo_url = branding.get("client_logo_url")
    agency_logo_url = branding.get("agency_logo_url")
    slide_numbers = branding.get("slide_numbers", True)
    progress_bar = branding.get("progress_bar", True)
    controls = branding.get("controls", True)

    # Get theme CSS URL
    theme_css = REVEAL_THEMES.get(theme, REVEAL_THEMES["white"])

    # Sort slides by order
    slides = sorted(presentation.slides, key=lambda s: s.order)

    # Generate slide sections
    slide_sections = []
    figure_data_scripts = []  # JavaScript to render Plotly figures
    figure_counter = 0

    # Title slide
    title_slide = _generate_title_slide(
        presentation.name,
        presentation.description,
        presentation.client_name,
        presentation.project_name,
        primary_color,
        secondary_color,
        logo_url
    )
    slide_sections.append(title_slide)

    # Content slides
    for slide in slides:
        slide_html, fig_scripts = await _generate_slide_section(
            slide, session, figure_counter, primary_color,
            client_logo_url=client_logo_url,
            agency_logo_url=agency_logo_url
        )
        slide_sections.append(slide_html)
        figure_data_scripts.extend(fig_scripts)
        figure_counter += len(fig_scripts)

    # Build final HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(presentation.name)}</title>

    <!-- Reveal.js CSS -->
    <link rel="stylesheet" href="{REVEALJS_CSS}">
    <link rel="stylesheet" href="{theme_css}" id="theme">

    <!-- Plotly.js -->
    <script src="{PLOTLY_JS}"></script>

    <!-- Custom styles -->
    <style>
        {_get_custom_css(primary_color, secondary_color)}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {''.join(slide_sections)}
        </div>
    </div>

    <!-- Reveal.js -->
    <script src="{REVEALJS_JS}"></script>

    <script>
        // Initialize Reveal.js
        Reveal.initialize({{
            hash: true,
            transition: '{transition}',
            slideNumber: {str(slide_numbers).lower()},
            progress: {str(progress_bar).lower()},
            controls: {str(controls).lower()},
            center: true,
            width: 1920,
            height: 1080,
            margin: 0.04,
            minScale: 0.2,
            maxScale: 2.0
        }});

        // Render Plotly figures after reveal is ready
        Reveal.on('ready', function() {{
            {chr(10).join(figure_data_scripts)}
        }});

        // Re-render figures on slide change to ensure proper sizing
        Reveal.on('slidechanged', function(event) {{
            const currentSlide = event.currentSlide;
            const plotlyDivs = currentSlide.querySelectorAll('.plotly-figure');
            plotlyDivs.forEach(function(div) {{
                if (div.data) {{
                    Plotly.relayout(div, {{}});
                }}
            }});
        }});
    </script>
</body>
</html>'''

    return html


def _generate_title_slide(
    name: str,
    description: Optional[str],
    client_name: Optional[str],
    project_name: Optional[str],
    primary_color: str,
    secondary_color: str,
    logo_url: Optional[str]
) -> str:
    """Generate the title slide section."""
    logo_html = f'<img src="{logo_url}" alt="Logo" class="title-logo">' if logo_url else ''

    meta_parts = []
    if client_name:
        meta_parts.append(_escape_html(client_name))
    if project_name:
        meta_parts.append(_escape_html(project_name))
    meta_html = ' | '.join(meta_parts) if meta_parts else ''

    date_str = datetime.now().strftime('%B %d, %Y')

    return f'''
            <section data-background-gradient="linear-gradient(135deg, {primary_color} 0%, {secondary_color} 100%)" class="title-slide">
                {logo_html}
                <h1>{_escape_html(name)}</h1>
                {f'<p class="subtitle">{_escape_html(description)}</p>' if description else ''}
                {f'<p class="meta">{meta_html}</p>' if meta_html else ''}
                <p class="date">{date_str}</p>
            </section>
'''


async def _generate_slide_section(
    slide: PresentationSlide,
    session: AsyncSession,
    figure_start_idx: int,
    primary_color: str,
    client_logo_url: Optional[str] = None,
    agency_logo_url: Optional[str] = None
) -> tuple[str, list[str]]:
    """
    Generate a single slide section.

    Returns:
        Tuple of (slide HTML, list of JavaScript statements for figures)
    """
    figure_scripts = []

    # Get layout options
    layout = slide.layout or {}
    bg_color = layout.get("background_color")
    bg_image = layout.get("background_image")
    transition_override = layout.get("transition_override")
    slide_layout = layout.get("layout", "full")

    # Build data attributes
    data_attrs = []
    if bg_color:
        data_attrs.append(f'data-background-color="{bg_color}"')
    if bg_image:
        data_attrs.append(f'data-background-image="{bg_image}"')
    if transition_override:
        data_attrs.append(f'data-transition="{transition_override}"')

    data_attrs_str = ' '.join(data_attrs)

    # Generate content based on slide type
    if slide.slide_type == "text":
        content = _generate_text_content(slide)
    elif slide.slide_type == "figure":
        content, figure_scripts = await _generate_figure_content(
            slide, session, figure_start_idx, slide_layout
        )
    elif slide.slide_type == "title":
        content = _generate_section_title_content(slide, primary_color)
    else:
        content = _generate_text_content(slide)

    # Speaker notes
    notes_html = ""
    if layout.get("speaker_notes"):
        notes_html = f'<aside class="notes">{_escape_html(layout["speaker_notes"])}</aside>'

    # Logo footer (bottom right)
    logo_footer_html = _generate_logo_footer(client_logo_url, agency_logo_url)

    return f'''
            <section {data_attrs_str} class="content-slide layout-{slide_layout}">
                <h2 class="slide-title">{_escape_html(slide.title)}</h2>
                {f'<p class="slide-description">{_escape_html(slide.description)}</p>' if slide.description else ''}
                <div class="slide-content">
                    {content}
                </div>
                {logo_footer_html}
                {notes_html}
            </section>
''', figure_scripts


def _generate_logo_footer(client_logo_url: Optional[str], agency_logo_url: Optional[str]) -> str:
    """Generate the logo footer HTML for bottom-right of slides."""
    if not client_logo_url and not agency_logo_url:
        return ""

    logos = []
    if client_logo_url:
        logos.append(f'<img src="{client_logo_url}" alt="Client Logo" class="footer-logo client-logo">')
    if agency_logo_url:
        logos.append(f'<img src="{agency_logo_url}" alt="Agency Logo" class="footer-logo agency-logo">')

    return f'''
        <div class="slide-logo-footer">
            {''.join(logos)}
        </div>
    '''


def _generate_text_content(slide: PresentationSlide) -> str:
    """Generate HTML for a text slide."""
    if slide.text_content:
        md_html = markdown.markdown(
            slide.text_content,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        return f'<div class="text-content">{md_html}</div>'
    return '<div class="text-content"><p>No content</p></div>'


def _generate_section_title_content(slide: PresentationSlide, primary_color: str) -> str:
    """Generate HTML for a section title slide."""
    return f'''
        <div class="section-title-content">
            {f'<div class="section-description">{_escape_html(slide.description)}</div>' if slide.description else ''}
        </div>
    '''


async def _generate_figure_content(
    slide: PresentationSlide,
    session: AsyncSession,
    figure_start_idx: int,
    layout: str
) -> tuple[str, list[str]]:
    """
    Generate HTML and JavaScript for a figure slide.

    Returns:
        Tuple of (HTML content, list of JavaScript statements)
    """
    if not slide.model_run_id or not slide.figure_type:
        return '<div class="figure-error">Figure configuration incomplete</div>', []

    # Load model run and results
    run = await session.get(ModelRun, slide.model_run_id)
    if not run or not run.results_path:
        return '<div class="figure-error">Model run not found or results unavailable</div>', []

    results_path = Path(run.results_path)
    if not results_path.exists():
        return '<div class="figure-error">Results file not found</div>', []

    try:
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        # Import here to avoid circular import
        from ..api.routes.runs import _extract_report_data

        model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
        product_columns = run.product_columns or []

        extracted_data = _extract_report_data(results, model_type, product_columns)

        # Check if clustering is needed
        clustering_result = None
        clustering_figure_types = ["clustered_biplot", "silhouette_analysis", "cluster_sizes", "dendrogram"]
        if slide.figure_type in clustering_figure_types:
            config = slide.figure_config or {}
            n_clusters = config.get("n_clusters")
            clustering_method = config.get("clustering_method", "kmeans")

            if n_clusters:
                clustering_result = _perform_clustering(
                    extracted_data, n_clusters, clustering_method
                )

        # Generate figure
        fig = generate_figure(
            FigureType(slide.figure_type),
            extracted_data,
            config=slide.figure_config,
            clustering_result=clustering_result
        )

        # Convert to JSON for JavaScript
        fig_dict = fig.to_dict()
        fig_json = json.dumps(fig_dict)

        # Create unique div ID
        div_id = f"plotly-fig-{figure_start_idx}"

        # JavaScript to render the figure
        js_statement = f'''
            (function() {{
                var figData = {fig_json};
                var layout = figData.layout || {{}};
                layout.autosize = true;
                layout.height = null;
                layout.width = null;
                var config = {{responsive: true, displayModeBar: true}};
                Plotly.newPlot('{div_id}', figData.data, layout, config);
            }})();
        '''

        # HTML content
        if layout == "split_left":
            html = f'''
                <div class="split-layout">
                    <div class="figure-panel">
                        <div id="{div_id}" class="plotly-figure"></div>
                    </div>
                    <div class="text-panel">
                        {_generate_text_content(slide) if slide.text_content else '<p>Add description in text content</p>'}
                    </div>
                </div>
            '''
        elif layout == "split_right":
            html = f'''
                <div class="split-layout">
                    <div class="text-panel">
                        {_generate_text_content(slide) if slide.text_content else '<p>Add description in text content</p>'}
                    </div>
                    <div class="figure-panel">
                        <div id="{div_id}" class="plotly-figure"></div>
                    </div>
                </div>
            '''
        else:  # full layout
            html = f'''
                <div class="full-figure">
                    <div id="{div_id}" class="plotly-figure"></div>
                </div>
            '''

        # Add source info
        run_name = run.name or f"Run {run.id[:8]}"
        html += f'''
            <div class="figure-source">
                Source: {_escape_html(run_name)} ({model_type})
            </div>
        '''

        return html, [js_statement]

    except Exception as e:
        return f'<div class="figure-error">Error generating figure: {_escape_html(str(e))}</div>', []


def _perform_clustering(extracted_data: dict, n_clusters: int, method: str = "kmeans") -> dict:
    """Perform clustering on product embeddings."""
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


def _get_custom_css(primary_color: str, secondary_color: str) -> str:
    """Get custom CSS for the presentation."""
    return f'''
        /* Title slide styles */
        .title-slide {{
            text-align: center;
        }}

        .title-slide h1 {{
            color: white;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .title-slide .subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 1.3em;
            margin-bottom: 1em;
        }}

        .title-slide .meta {{
            color: rgba(255,255,255,0.8);
            font-size: 1em;
        }}

        .title-slide .date {{
            color: rgba(255,255,255,0.7);
            font-size: 0.9em;
            margin-top: 2em;
        }}

        .title-slide .title-logo {{
            max-width: 200px;
            max-height: 80px;
            margin-bottom: 1em;
        }}

        /* Content slide styles */
        .content-slide {{
            text-align: left;
        }}

        .content-slide .slide-title {{
            color: {primary_color};
            font-size: 1.8em;
            margin-bottom: 0.3em;
            border-bottom: 3px solid {primary_color};
            padding-bottom: 0.2em;
        }}

        .content-slide .slide-description {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 1em;
            font-style: italic;
        }}

        .content-slide .slide-content {{
            width: 100%;
            height: calc(100% - 120px);
        }}

        /* Figure styles */
        .plotly-figure {{
            width: 100%;
            height: 65vh;
            min-height: 400px;
        }}

        .full-figure {{
            width: 100%;
            height: 100%;
        }}

        .figure-source {{
            font-size: 0.6em;
            color: #888;
            text-align: right;
            margin-top: 0.5em;
        }}

        .figure-error {{
            background: #fff5f5;
            border: 2px solid #fc8181;
            color: #c53030;
            padding: 2em;
            border-radius: 8px;
            text-align: center;
        }}

        /* Split layout styles */
        .split-layout {{
            display: flex;
            gap: 2em;
            height: 100%;
            align-items: stretch;
        }}

        .split-layout .figure-panel {{
            flex: 1.2;
            min-width: 0;
        }}

        .split-layout .text-panel {{
            flex: 0.8;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;  /* Align text to top */
            padding-top: 0.5em;
        }}

        .split-layout .plotly-figure {{
            height: 55vh;
        }}

        /* Logo footer styles */
        .slide-logo-footer {{
            position: absolute;
            bottom: 20px;
            right: 40px;
            display: flex;
            align-items: center;
            gap: 15px;
            z-index: 10;
        }}

        .footer-logo {{
            max-height: 40px;
            max-width: 120px;
            object-fit: contain;
            opacity: 0.85;
        }}

        .footer-logo:hover {{
            opacity: 1;
        }}

        /* Text content styles */
        .text-content {{
            font-size: 0.85em;
            line-height: 1.6;
        }}

        .text-content ul, .text-content ol {{
            text-align: left;
            margin-left: 1em;
        }}

        .text-content li {{
            margin-bottom: 0.5em;
        }}

        .text-content table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8em;
        }}

        .text-content th, .text-content td {{
            border: 1px solid #ddd;
            padding: 0.5em;
            text-align: left;
        }}

        .text-content th {{
            background: {primary_color}15;
        }}

        /* Section title styles */
        .section-title-content {{
            text-align: center;
            padding: 2em;
        }}

        /* Override reveal.js defaults for better figure display */
        .reveal .slides section {{
            height: 100%;
            padding: 20px 40px;
        }}

        .reveal .slides > section {{
            padding: 20px 40px;
        }}

        /* Progress bar color */
        .reveal .progress {{
            background: rgba(0, 0, 0, 0.2);
        }}

        .reveal .progress span {{
            background: {primary_color};
        }}

        /* Slide number styling */
        .reveal .slide-number {{
            background: {primary_color};
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
        }}
    '''


async def generate_single_slide_preview(
    slide: PresentationSlide,
    session: AsyncSession,
    theme: str = "white",
    primary_color: str = "#667eea",
    secondary_color: str = "#764ba2"
) -> str:
    """
    Generate a standalone HTML preview for a single slide.

    Used for real-time preview in the editor.
    """
    theme_css = REVEAL_THEMES.get(theme, REVEAL_THEMES["white"])

    slide_html, figure_scripts = await _generate_slide_section(
        slide, session, 0, primary_color
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slide Preview</title>

    <link rel="stylesheet" href="{REVEALJS_CSS}">
    <link rel="stylesheet" href="{theme_css}">
    <script src="{PLOTLY_JS}"></script>

    <style>
        {_get_custom_css(primary_color, secondary_color)}

        /* Preview-specific styles */
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}

        .reveal {{
            height: 100vh;
        }}

        .reveal .slides {{
            height: 100%;
        }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {slide_html}
        </div>
    </div>

    <script src="{REVEALJS_JS}"></script>
    <script>
        Reveal.initialize({{
            hash: false,
            controls: false,
            progress: false,
            slideNumber: false,
            embedded: true,
            center: true,
            width: 1920,
            height: 1080,
            margin: 0.04
        }});

        Reveal.on('ready', function() {{
            {chr(10).join(figure_scripts)}
        }});
    </script>
</body>
</html>'''
