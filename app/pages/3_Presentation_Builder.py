"""
Presentation Builder page for creating multi-run presentations.

Features:
- Two-panel layout with live preview
- Reveal.js-powered slides with themes and transitions
- Per-slide customization (layout, colors, transitions)
- Real-time preview of individual slides
- Export to reveal.js or legacy HTML format
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
import urllib.parse


# API configuration
API_BASE_URL = "http://localhost:8000"


# =============================================================================
# CONSTANTS
# =============================================================================

REVEAL_THEMES = [
    ("white", "White (Light)"),
    ("black", "Black (Dark)"),
    ("league", "League"),
    ("beige", "Beige"),
    ("sky", "Sky"),
    ("night", "Night"),
    ("serif", "Serif"),
    ("simple", "Simple"),
    ("solarized", "Solarized"),
    ("moon", "Moon"),
    ("dracula", "Dracula"),
    ("blood", "Blood"),
]

REVEAL_TRANSITIONS = [
    ("slide", "Slide"),
    ("fade", "Fade"),
    ("convex", "Convex"),
    ("concave", "Concave"),
    ("zoom", "Zoom"),
    ("none", "None"),
]

SLIDE_LAYOUTS = [
    ("full", "Full Width Figure"),
    ("split_left", "Figure Left, Text Right"),
    ("split_right", "Text Left, Figure Right"),
]


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_completed_runs() -> List[Dict]:
    """Get all completed model runs."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/runs",
            params={"status": "completed", "limit": 500},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("runs", [])
        return []
    except Exception:
        return []


def get_run_figures(run_id: str, n_clusters: Optional[int] = None, clustering_method: str = "kmeans") -> List[Dict]:
    """Get available figures for a model run."""
    try:
        params = {}
        if n_clusters is not None:
            params["include_clustering"] = "true"
            params["n_clusters"] = n_clusters
            params["clustering_method"] = clustering_method

        response = requests.get(
            f"{API_BASE_URL}/api/v1/runs/{run_id}/figures",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("available_figures", [])
        return []
    except Exception:
        return []


def run_clustering(run_id: str, n_clusters: Optional[int] = None, max_k: int = 10, method: str = "kmeans") -> Optional[Dict]:
    """Run clustering on a model run."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/runs/{run_id}/clustering",
            json={
                "method": method,
                "n_clusters": n_clusters,
                "max_k": max_k,
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_presentations() -> List[Dict]:
    """Get all presentations."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/presentations",
            params={"limit": 100},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("presentations", [])
        return []
    except Exception:
        return []


def get_presentation(presentation_id: str) -> Optional[Dict]:
    """Get a presentation by ID."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def create_presentation(name: str, description: str = "", client_name: str = "", project_name: str = "", branding_options: Optional[Dict] = None) -> Optional[Dict]:
    """Create a new presentation."""
    try:
        data = {
            "name": name,
            "description": description or None,
            "client_name": client_name or None,
            "project_name": project_name or None,
        }
        if branding_options:
            data["branding_options"] = branding_options

        response = requests.post(
            f"{API_BASE_URL}/api/v1/presentations",
            json=data,
            timeout=10
        )
        if response.status_code == 201:
            return response.json()
        return None
    except Exception:
        return None


def update_presentation(presentation_id: str, **kwargs) -> Optional[Dict]:
    """Update presentation metadata."""
    try:
        data = {k: v for k, v in kwargs.items() if v is not None}
        response = requests.put(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def delete_presentation(presentation_id: str) -> bool:
    """Delete a presentation."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}",
            timeout=10
        )
        return response.status_code == 204
    except Exception:
        return False


def add_slide(presentation_id: str, slide_data: Dict) -> Optional[Dict]:
    """Add a slide to a presentation."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/slides",
            json=slide_data,
            timeout=10
        )
        if response.status_code == 201:
            return response.json()
        return None
    except Exception:
        return None


def update_slide(presentation_id: str, slide_id: str, slide_data: Dict) -> Optional[Dict]:
    """Update a slide."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/slides/{slide_id}",
            json=slide_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def delete_slide(presentation_id: str, slide_id: str) -> bool:
    """Delete a slide."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/slides/{slide_id}",
            timeout=10
        )
        return response.status_code == 204
    except Exception:
        return False


def reorder_slides(presentation_id: str, slide_ids: List[str]) -> Optional[Dict]:
    """Reorder slides in a presentation."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/slides/reorder",
            json={"slide_ids": slide_ids},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def export_presentation(presentation_id: str, format: str = "revealjs") -> Optional[bytes]:
    """Export presentation as HTML."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/export",
            params={"format": format},
            timeout=60
        )
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None


def get_slide_preview(presentation_id: str, slide_id: str, theme: str = "white", primary_color: str = "#667eea", secondary_color: str = "#764ba2") -> Optional[str]:
    """Get single slide preview HTML."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/slides/{slide_id}/preview",
            params={
                "theme": theme,
                "primary_color": primary_color,
                "secondary_color": secondary_color,
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("html")
        return None
    except Exception:
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_display_name(model_type: str) -> str:
    """Get display name for model type."""
    names = {
        "lca": "Latent Class Analysis",
        "lca_covariates": "LCA with Covariates",
        "factor_tetrachoric": "Factor Analysis (Tetrachoric)",
        "bayesian_factor_vi": "Bayesian Factor (VI)",
        "bayesian_factor_pymc": "Bayesian Factor (PyMC)",
        "nmf": "Non-negative Matrix Factorization",
        "mca": "Multiple Correspondence Analysis",
        "dcm": "Discrete Choice Model",
        "lda": "Latent Dirichlet Allocation",
        "network": "Network Analysis",
    }
    return names.get(model_type, model_type)


def init_session_state():
    """Initialize session state for presentation builder."""
    if "pb_mode" not in st.session_state:
        st.session_state.pb_mode = "select"  # select, edit
    if "pb_presentation_id" not in st.session_state:
        st.session_state.pb_presentation_id = None
    if "pb_selected_slide_idx" not in st.session_state:
        st.session_state.pb_selected_slide_idx = 0
    if "pb_slides" not in st.session_state:
        st.session_state.pb_slides = []
    if "pb_clustering_settings" not in st.session_state:
        st.session_state.pb_clustering_settings = {}
    # Presentation settings
    if "pb_name" not in st.session_state:
        st.session_state.pb_name = "Market Structure Analysis"
    if "pb_desc" not in st.session_state:
        st.session_state.pb_desc = ""
    if "pb_client" not in st.session_state:
        st.session_state.pb_client = ""
    if "pb_project" not in st.session_state:
        st.session_state.pb_project = ""
    # Reveal.js settings
    if "pb_theme" not in st.session_state:
        st.session_state.pb_theme = "white"
    if "pb_transition" not in st.session_state:
        st.session_state.pb_transition = "slide"
    if "pb_primary_color" not in st.session_state:
        st.session_state.pb_primary_color = "#667eea"
    if "pb_secondary_color" not in st.session_state:
        st.session_state.pb_secondary_color = "#764ba2"
    if "pb_client_logo_url" not in st.session_state:
        st.session_state.pb_client_logo_url = ""
    if "pb_agency_logo_url" not in st.session_state:
        st.session_state.pb_agency_logo_url = ""
    if "pb_slide_numbers" not in st.session_state:
        st.session_state.pb_slide_numbers = True
    if "pb_progress_bar" not in st.session_state:
        st.session_state.pb_progress_bar = True
    if "pb_controls" not in st.session_state:
        st.session_state.pb_controls = True
    if "pb_export_triggered" not in st.session_state:
        st.session_state.pb_export_triggered = False


def get_slide_icon(slide_type: str) -> str:
    """Get icon for slide type."""
    icons = {
        "figure": "📊",
        "text": "📝",
        "title": "🏷️",
        "comparison": "⚖️",
        "summary": "📋",
    }
    return icons.get(slide_type, "📄")


# =============================================================================
# RENDER SELECT/CREATE MODE
# =============================================================================

def render_select_mode():
    """Render presentation selection or creation view."""
    st.header("📑 Presentation Builder")
    st.write("Create professional reveal.js presentations from your model analyses.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Create New Presentation")

        name = st.text_input("Presentation Name", value="Market Structure Analysis")
        desc = st.text_area("Description (optional)", height=68)

        col_a, col_b = st.columns(2)
        with col_a:
            client = st.text_input("Client Name (optional)")
        with col_b:
            project = st.text_input("Project Name (optional)")

        if st.button("Create Presentation", type="primary"):
            if name:
                st.session_state.pb_name = name
                st.session_state.pb_desc = desc
                st.session_state.pb_client = client
                st.session_state.pb_project = project
                st.session_state.pb_slides = []
                st.session_state.pb_mode = "edit"
                st.rerun()
            else:
                st.error("Please enter a presentation name.")

    with col2:
        st.subheader("Recent Presentations")
        presentations = get_presentations()

        if presentations:
            for pres in presentations[:5]:
                pres_name = pres.get("name", "Untitled")
                slide_count = pres.get("slide_count", 0)
                created_at = pres.get("created_at", "")
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        date_str = dt.strftime("%Y-%m-%d")
                    except Exception:
                        date_str = ""
                else:
                    date_str = ""

                with st.container():
                    st.write(f"**{pres_name}**")
                    st.caption(f"{slide_count} slides • {date_str}")
                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("Load", key=f"load_{pres['id']}"):
                            load_presentation(pres["id"])
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{pres['id']}"):
                            if delete_presentation(pres["id"]):
                                st.rerun()
                    st.divider()
        else:
            st.info("No presentations yet.")


def load_presentation(presentation_id: str):
    """Load an existing presentation into session state."""
    presentation = get_presentation(presentation_id)
    if presentation:
        st.session_state.pb_presentation_id = presentation_id
        st.session_state.pb_name = presentation.get("name", "Untitled")
        st.session_state.pb_desc = presentation.get("description", "")
        st.session_state.pb_client = presentation.get("client_name", "")
        st.session_state.pb_project = presentation.get("project_name", "")

        # Load branding options
        branding = presentation.get("branding_options") or {}
        st.session_state.pb_theme = branding.get("theme", "white")
        st.session_state.pb_transition = branding.get("transition", "slide")
        st.session_state.pb_primary_color = branding.get("primary_color", "#667eea")
        st.session_state.pb_secondary_color = branding.get("secondary_color", "#764ba2")
        st.session_state.pb_client_logo_url = branding.get("client_logo_url", "")
        st.session_state.pb_agency_logo_url = branding.get("agency_logo_url", "")
        st.session_state.pb_slide_numbers = branding.get("slide_numbers", True)
        st.session_state.pb_progress_bar = branding.get("progress_bar", True)
        st.session_state.pb_controls = branding.get("controls", True)

        # Load slides
        slides = presentation.get("slides", [])
        st.session_state.pb_slides = [
            {
                "id": s.get("id"),
                "title": s.get("title", "Untitled"),
                "description": s.get("description"),
                "slide_type": s.get("slide_type", "figure"),
                "model_run_id": s.get("model_run_id"),
                "figure_type": s.get("figure_type"),
                "figure_config": s.get("figure_config"),
                "text_content": s.get("text_content"),
                "layout": s.get("layout") or {},
            }
            for s in slides
        ]

        st.session_state.pb_selected_slide_idx = 0
        st.session_state.pb_mode = "edit"


# =============================================================================
# RENDER EDIT MODE - TWO PANEL LAYOUT
# =============================================================================

def render_edit_mode():
    """Render the two-panel editor layout."""
    # Check if export was triggered - handle it outside all column contexts FIRST
    if st.session_state.get("pb_export_triggered"):
        st.session_state.pb_export_triggered = False
        render_export_view()
        return  # Don't render the rest of the editor when showing export

    # Top bar with presentation info and export
    render_top_bar()

    # Main content area: Slide list | Preview panel
    col_sidebar, col_preview = st.columns([1, 2])

    with col_sidebar:
        render_slide_sidebar()

    with col_preview:
        render_preview_panel()

    # Slide editor below
    render_slide_editor()


def render_top_bar():
    """Render the top bar with presentation settings and export."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"### 📑 {st.session_state.pb_name}")
        if st.session_state.pb_client or st.session_state.pb_project:
            meta = " | ".join(filter(None, [st.session_state.pb_client, st.session_state.pb_project]))
            st.caption(meta)

    with col2:
        # Theme selector
        theme_names = [t[1] for t in REVEAL_THEMES]
        theme_values = [t[0] for t in REVEAL_THEMES]
        current_idx = theme_values.index(st.session_state.pb_theme) if st.session_state.pb_theme in theme_values else 0
        theme_display = st.selectbox(
            "Theme",
            options=theme_names,
            index=current_idx,
            key="theme_select"
        )
        st.session_state.pb_theme = theme_values[theme_names.index(theme_display)]

    with col3:
        # Transition selector
        trans_names = [t[1] for t in REVEAL_TRANSITIONS]
        trans_values = [t[0] for t in REVEAL_TRANSITIONS]
        current_trans_idx = trans_values.index(st.session_state.pb_transition) if st.session_state.pb_transition in trans_values else 0
        trans_display = st.selectbox(
            "Transition",
            options=trans_names,
            index=current_trans_idx,
            key="trans_select"
        )
        st.session_state.pb_transition = trans_values[trans_names.index(trans_display)]

    with col4:
        st.write("")  # Spacer
        if st.button("🚀 Export", type="primary", use_container_width=True):
            # Store export result in session state - will be rendered outside columns
            st.session_state.pb_export_triggered = True

    # Settings expander
    with st.expander("⚙️ Presentation Settings", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.session_state.pb_name = st.text_input("Name", value=st.session_state.pb_name)
            st.session_state.pb_client = st.text_input("Client", value=st.session_state.pb_client)

        with col_b:
            st.session_state.pb_desc = st.text_area("Description", value=st.session_state.pb_desc, height=68)
            st.session_state.pb_project = st.text_input("Project", value=st.session_state.pb_project)

        with col_c:
            st.session_state.pb_primary_color = st.color_picker("Primary Color", value=st.session_state.pb_primary_color)
            st.session_state.pb_secondary_color = st.color_picker("Secondary Color", value=st.session_state.pb_secondary_color)

        with col_d:
            st.session_state.pb_slide_numbers = st.checkbox("Slide Numbers", value=st.session_state.pb_slide_numbers)
            st.session_state.pb_progress_bar = st.checkbox("Progress Bar", value=st.session_state.pb_progress_bar)
            st.session_state.pb_controls = st.checkbox("Navigation Controls", value=st.session_state.pb_controls)

        # Logo settings row
        st.markdown("**Slide Footer Logos** (displayed bottom-right on each slide)")
        col_logo1, col_logo2 = st.columns(2)
        with col_logo1:
            st.session_state.pb_client_logo_url = st.text_input(
                "Client Logo URL",
                value=st.session_state.pb_client_logo_url,
                placeholder="https://example.com/client-logo.png"
            )
        with col_logo2:
            st.session_state.pb_agency_logo_url = st.text_input(
                "Agency Logo URL",
                value=st.session_state.pb_agency_logo_url,
                placeholder="https://example.com/agency-logo.png"
            )

        col_back, col_spacer = st.columns([1, 3])
        with col_back:
            if st.button("← Back to Start"):
                st.session_state.pb_mode = "select"
                st.rerun()

    st.divider()


def render_slide_sidebar():
    """Render the slide list sidebar."""
    st.subheader("Slides")

    # Add slide button
    if st.button("➕ Add Slide", use_container_width=True):
        new_slide = {
            "id": None,  # Will be assigned when saved
            "title": f"Slide {len(st.session_state.pb_slides) + 1}",
            "description": "",
            "slide_type": "figure",
            "model_run_id": None,
            "figure_type": None,
            "figure_config": {},
            "text_content": "",
            "layout": {"layout": "full"},
        }
        st.session_state.pb_slides.append(new_slide)
        st.session_state.pb_selected_slide_idx = len(st.session_state.pb_slides) - 1
        st.rerun()

    st.divider()

    # Slide list
    slides = st.session_state.pb_slides
    if slides:
        for i, slide in enumerate(slides):
            is_selected = i == st.session_state.pb_selected_slide_idx
            slide_icon = get_slide_icon(slide.get("slide_type", "figure"))
            slide_title = slide.get("title", "Untitled")[:25]
            if len(slide.get("title", "")) > 25:
                slide_title += "..."

            # Slide card
            container_class = "selected-slide" if is_selected else ""
            with st.container():
                col_num, col_info, col_actions = st.columns([0.3, 1.5, 0.5])

                with col_num:
                    st.markdown(f"**{i + 1}**")

                with col_info:
                    if st.button(f"{slide_icon} {slide_title}", key=f"slide_btn_{i}", use_container_width=True):
                        st.session_state.pb_selected_slide_idx = i
                        st.rerun()

                with col_actions:
                    # Action buttons
                    act_col1, act_col2 = st.columns(2)
                    with act_col1:
                        if i > 0:
                            if st.button("↑", key=f"up_{i}"):
                                slides[i], slides[i-1] = slides[i-1], slides[i]
                                st.session_state.pb_selected_slide_idx = i - 1
                                st.rerun()
                    with act_col2:
                        if i < len(slides) - 1:
                            if st.button("↓", key=f"down_{i}"):
                                slides[i], slides[i+1] = slides[i+1], slides[i]
                                st.session_state.pb_selected_slide_idx = i + 1
                                st.rerun()

                # Show selected indicator
                if is_selected:
                    st.markdown("---")
    else:
        st.info("No slides yet. Click 'Add Slide' to begin.")


def render_preview_panel():
    """Render the slide preview panel."""
    st.subheader("Preview")

    slides = st.session_state.pb_slides
    if not slides:
        st.info("Add a slide to see preview")
        return

    idx = st.session_state.pb_selected_slide_idx
    if idx >= len(slides):
        idx = 0
        st.session_state.pb_selected_slide_idx = 0

    slide = slides[idx]

    # Check if slide is complete enough for preview
    if slide.get("slide_type") == "figure":
        if not slide.get("model_run_id") or not slide.get("figure_type"):
            st.warning("Configure the slide's model run and figure type to see preview.")
            # Show placeholder
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {st.session_state.pb_primary_color}20, {st.session_state.pb_secondary_color}20);
                border: 2px dashed {st.session_state.pb_primary_color};
                border-radius: 8px;
                padding: 60px 20px;
                text-align: center;
                color: #666;
            ">
                <h3>{slide.get('title', 'Untitled Slide')}</h3>
                <p>📊 Select a model run and figure type</p>
            </div>
            """, unsafe_allow_html=True)
            return

    # For text slides or complete figure slides, generate preview
    # Create a mock preview using the slide data
    render_slide_mock_preview(slide)


def render_slide_mock_preview(slide: Dict):
    """Render a mock preview of the slide (client-side)."""
    slide_type = slide.get("slide_type", "figure")
    title = slide.get("title", "Untitled")
    description = slide.get("description", "")
    layout_opts = slide.get("layout") or {}
    layout = layout_opts.get("layout", "full")

    # Preview container
    st.markdown(f"""
    <style>
    .preview-slide {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        min-height: 350px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .preview-title {{
        color: {st.session_state.pb_primary_color};
        font-size: 1.5em;
        border-bottom: 3px solid {st.session_state.pb_primary_color};
        padding-bottom: 10px;
        margin-bottom: 15px;
    }}
    .preview-desc {{
        color: #666;
        font-style: italic;
        margin-bottom: 20px;
    }}
    .preview-content {{
        background: #f8f9fa;
        border-radius: 4px;
        padding: 20px;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f'<div class="preview-slide">', unsafe_allow_html=True)
        st.markdown(f'<div class="preview-title">{title}</div>', unsafe_allow_html=True)

        if description:
            st.markdown(f'<div class="preview-desc">{description}</div>', unsafe_allow_html=True)

        if slide_type == "figure":
            figure_type = slide.get("figure_type", "")
            model_run_id = slide.get("model_run_id", "")
            run_label = model_run_id[:8] if model_run_id else "None"

            content_html = f"""
            <div class="preview-content">
                <div style="text-align: center;">
                    <div style="font-size: 3em; margin-bottom: 10px;">📊</div>
                    <div><strong>{figure_type or 'No figure selected'}</strong></div>
                    <div style="color: #888; font-size: 0.9em;">Run: {run_label}</div>
                    <div style="color: #aaa; font-size: 0.8em; margin-top: 10px;">Layout: {layout}</div>
                </div>
            </div>
            """
        else:
            text_content = slide.get("text_content", "No content")
            content_html = f"""
            <div class="preview-content" style="text-align: left; align-items: flex-start;">
                <div style="width: 100%;">
                    {text_content[:500]}{'...' if len(text_content) > 500 else ''}
                </div>
            </div>
            """

        st.markdown(content_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Theme preview indicator
    st.caption(f"Theme: {st.session_state.pb_theme} | Transition: {st.session_state.pb_transition}")


def render_slide_editor():
    """Render the slide editor panel."""
    slides = st.session_state.pb_slides
    if not slides:
        return

    idx = st.session_state.pb_selected_slide_idx
    if idx >= len(slides):
        return

    slide = slides[idx]

    st.divider()
    st.subheader(f"Edit Slide {idx + 1}")

    # Main editor columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Basic info
        slide["title"] = st.text_input(
            "Slide Title",
            value=slide.get("title", ""),
            key=f"edit_title_{idx}"
        )

        slide["description"] = st.text_area(
            "Description (shown on slide)",
            value=slide.get("description", "") or "",
            key=f"edit_desc_{idx}",
            height=68
        )

        # Slide type
        type_options = ["figure", "text"]
        current_type = slide.get("slide_type", "figure")
        type_idx = type_options.index(current_type) if current_type in type_options else 0

        new_type = st.radio(
            "Slide Type",
            options=["Figure", "Text"],
            index=type_idx,
            horizontal=True,
            key=f"edit_type_{idx}"
        )
        slide["slide_type"] = "figure" if new_type == "Figure" else "text"

    with col2:
        # Layout options
        layout_opts = slide.get("layout") or {}

        layout_names = [l[1] for l in SLIDE_LAYOUTS]
        layout_values = [l[0] for l in SLIDE_LAYOUTS]
        current_layout = layout_opts.get("layout", "full")
        layout_idx = layout_values.index(current_layout) if current_layout in layout_values else 0

        layout_display = st.selectbox(
            "Layout",
            options=layout_names,
            index=layout_idx,
            key=f"edit_layout_{idx}"
        )
        layout_opts["layout"] = layout_values[layout_names.index(layout_display)]

        # Per-slide appearance
        col_bg, col_trans = st.columns(2)
        with col_bg:
            bg_color = st.color_picker(
                "Background Color",
                value=layout_opts.get("background_color") or "#ffffff",
                key=f"edit_bg_{idx}"
            )
            if bg_color != "#ffffff":
                layout_opts["background_color"] = bg_color

        with col_trans:
            trans_names = [t[1] for t in REVEAL_TRANSITIONS]
            trans_values = [t[0] for t in REVEAL_TRANSITIONS]
            current_trans = layout_opts.get("transition_override")
            trans_idx = trans_values.index(current_trans) if current_trans in trans_values else 0

            trans_override = st.selectbox(
                "Transition Override",
                options=["(Use default)"] + trans_names,
                index=0 if not current_trans else trans_values.index(current_trans) + 1,
                key=f"edit_trans_{idx}"
            )
            if trans_override != "(Use default)":
                layout_opts["transition_override"] = trans_values[trans_names.index(trans_override)]
            elif "transition_override" in layout_opts:
                del layout_opts["transition_override"]

        slide["layout"] = layout_opts

    # Type-specific editor
    st.divider()

    if slide["slide_type"] == "figure":
        render_figure_editor(slide, idx)
    else:
        render_text_editor(slide, idx)

    # Delete button
    st.divider()
    col_del, col_spacer = st.columns([1, 3])
    with col_del:
        if st.button("🗑️ Delete Slide", key=f"del_slide_{idx}"):
            st.session_state.pb_slides.pop(idx)
            if st.session_state.pb_selected_slide_idx >= len(st.session_state.pb_slides):
                st.session_state.pb_selected_slide_idx = max(0, len(st.session_state.pb_slides) - 1)
            st.rerun()


def render_figure_editor(slide: Dict, idx: int):
    """Render the figure-specific editor."""
    st.markdown("**Figure Settings**")

    # Get completed runs
    runs = get_completed_runs()

    if not runs:
        st.warning("No completed model runs found.")
        return

    # Run selector
    run_options = {f"{r.get('name') or r['id'][:8]} ({get_model_display_name(r['model_type'])})": r["id"] for r in runs}
    run_display_list = list(run_options.keys())

    current_run_id = slide.get("model_run_id")
    current_run_display = None
    if current_run_id:
        for display, rid in run_options.items():
            if rid == current_run_id:
                current_run_display = display
                break

    col_run, col_fig = st.columns(2)

    with col_run:
        selected_run_display = st.selectbox(
            "Model Run",
            options=run_display_list,
            index=run_display_list.index(current_run_display) if current_run_display in run_display_list else 0,
            key=f"fig_run_{idx}"
        )
        slide["model_run_id"] = run_options.get(selected_run_display)

    # Clustering options
    with st.expander("🔍 Clustering Options", expanded=False):
        run_id = slide["model_run_id"]
        if run_id:
            current_settings = st.session_state.pb_clustering_settings.get(run_id, {
                "enabled": False,
                "n_clusters": None,
                "method": "kmeans"
            })

            enable_clustering = st.checkbox(
                "Enable Clustering",
                value=current_settings.get("enabled", False),
                key=f"clust_enable_{idx}"
            )

            if enable_clustering:
                col_clust1, col_clust2 = st.columns(2)
                with col_clust1:
                    auto_detect = st.checkbox(
                        "Auto-detect clusters",
                        value=current_settings.get("n_clusters") is None,
                        key=f"clust_auto_{idx}"
                    )
                    if not auto_detect:
                        n_clusters = st.number_input(
                            "Number of Clusters",
                            min_value=2,
                            max_value=15,
                            value=current_settings.get("n_clusters") or 3,
                            key=f"clust_n_{idx}"
                        )
                    else:
                        n_clusters = None

                with col_clust2:
                    clust_method = st.selectbox(
                        "Method",
                        options=["kmeans", "hierarchical"],
                        index=0 if current_settings.get("method", "kmeans") == "kmeans" else 1,
                        key=f"clust_method_{idx}"
                    )

                if st.button("Run Clustering", key=f"run_clust_{idx}"):
                    with st.spinner("Running clustering..."):
                        result = run_clustering(run_id, n_clusters=n_clusters, method=clust_method)
                        if result:
                            st.session_state.pb_clustering_settings[run_id] = {
                                "enabled": True,
                                "n_clusters": result.get("n_clusters"),
                                "method": clust_method,
                                "result": result
                            }
                            st.success(f"Found {result.get('n_clusters')} clusters!")
                            st.rerun()
                        else:
                            st.error("Clustering failed.")

                if current_settings.get("result"):
                    st.info(f"Active: {current_settings.get('n_clusters')} clusters ({current_settings.get('method')})")
            else:
                if run_id in st.session_state.pb_clustering_settings:
                    del st.session_state.pb_clustering_settings[run_id]

    # Figure type selector
    with col_fig:
        run_id = slide["model_run_id"]
        if run_id:
            # Get figures (with clustering if enabled)
            clust_settings = st.session_state.pb_clustering_settings.get(run_id, {})
            if clust_settings.get("enabled") and clust_settings.get("n_clusters"):
                figures = get_run_figures(
                    run_id,
                    n_clusters=clust_settings["n_clusters"],
                    clustering_method=clust_settings.get("method", "kmeans")
                )
            else:
                figures = get_run_figures(run_id)

            available = [f for f in figures if f.get("available", True)]

            if available:
                fig_options = {f["name"]: f["type"] for f in available}
                fig_display_list = list(fig_options.keys())

                current_fig_type = slide.get("figure_type")
                current_fig_display = None
                if current_fig_type:
                    for display, ftype in fig_options.items():
                        if ftype == current_fig_type:
                            current_fig_display = display
                            break

                selected_fig_display = st.selectbox(
                    "Figure Type",
                    options=fig_display_list,
                    index=fig_display_list.index(current_fig_display) if current_fig_display in fig_display_list else 0,
                    key=f"fig_type_{idx}"
                )
                slide["figure_type"] = fig_options.get(selected_fig_display)

                # Add clustering config if needed
                clustering_figure_types = ["clustered_biplot", "silhouette_analysis", "cluster_sizes", "dendrogram"]
                if slide["figure_type"] in clustering_figure_types and clust_settings.get("n_clusters"):
                    slide["figure_config"] = {
                        "n_clusters": clust_settings["n_clusters"],
                        "clustering_method": clust_settings.get("method", "kmeans")
                    }
            else:
                st.warning("No figures available.")
                slide["figure_type"] = None
        else:
            st.info("Select a model run first.")


def render_text_editor(slide: Dict, idx: int):
    """Render the text-specific editor."""
    st.markdown("**Text Content** (Markdown supported)")

    slide["text_content"] = st.text_area(
        "Content",
        value=slide.get("text_content", "") or "",
        height=200,
        key=f"text_content_{idx}",
        help="Use Markdown: **bold**, *italic*, - bullet points, ### headings, etc.",
        label_visibility="collapsed"
    )


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def render_export_view():
    """Render the full-width export view with preview."""
    # Back button at the top
    if st.button("← Back to Editor", type="secondary"):
        st.rerun()

    st.markdown(f"### 📑 {st.session_state.pb_name} - Export")
    st.divider()

    # Now do the actual export
    export_current_presentation()


def export_current_presentation():
    """Export the current presentation."""
    with st.spinner("Creating presentation..."):
        # Build branding options
        branding = {
            "theme": st.session_state.pb_theme,
            "transition": st.session_state.pb_transition,
            "primary_color": st.session_state.pb_primary_color,
            "secondary_color": st.session_state.pb_secondary_color,
            "client_logo_url": st.session_state.pb_client_logo_url or None,
            "agency_logo_url": st.session_state.pb_agency_logo_url or None,
            "slide_numbers": st.session_state.pb_slide_numbers,
            "progress_bar": st.session_state.pb_progress_bar,
            "controls": st.session_state.pb_controls,
        }

        # Create or update presentation
        if st.session_state.pb_presentation_id:
            # Update existing
            update_presentation(
                st.session_state.pb_presentation_id,
                name=st.session_state.pb_name,
                description=st.session_state.pb_desc,
                client_name=st.session_state.pb_client,
                project_name=st.session_state.pb_project,
                branding_options=branding
            )
            presentation_id = st.session_state.pb_presentation_id
        else:
            # Create new
            pres = create_presentation(
                name=st.session_state.pb_name,
                description=st.session_state.pb_desc,
                client_name=st.session_state.pb_client,
                project_name=st.session_state.pb_project,
                branding_options=branding
            )
            if not pres:
                st.error("Failed to create presentation.")
                return
            presentation_id = pres["id"]
            st.session_state.pb_presentation_id = presentation_id

        # Sync slides
        # For simplicity, delete existing and re-add
        existing = get_presentation(presentation_id)
        if existing:
            for s in existing.get("slides", []):
                delete_slide(presentation_id, s["id"])

        # Add current slides
        for slide in st.session_state.pb_slides:
            slide_data = {
                "title": slide.get("title", "Untitled"),
                "description": slide.get("description"),
                "slide_type": slide.get("slide_type", "figure"),
                "model_run_id": slide.get("model_run_id"),
                "figure_type": slide.get("figure_type"),
                "figure_config": slide.get("figure_config"),
                "text_content": slide.get("text_content"),
                "layout": slide.get("layout"),
            }
            add_slide(presentation_id, slide_data)

        # Export
        html_content = export_presentation(presentation_id, format="revealjs")

        if html_content:
            st.success("Presentation generated!")

            # Download button
            filename = f"{st.session_state.pb_name.replace(' ', '_')}_slides.html"
            st.download_button(
                label="📥 Download Presentation",
                data=html_content,
                file_name=filename,
                mime="text/html",
                type="primary"
            )

            # Full-width preview (no expander, no columns)
            st.subheader("Preview")
            st.components.v1.html(html_content.decode("utf-8"), height=800, scrolling=True)
        else:
            st.error("Failed to export presentation.")


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Presentation Builder",
        page_icon="📑",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Render based on mode
    if st.session_state.pb_mode == "select":
        render_select_mode()
    else:
        render_edit_mode()


if __name__ == "__main__":
    main()
