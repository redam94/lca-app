"""
Presentation Builder page for creating multi-run presentations.

Allows analysts to:
1. Select multiple completed model runs
2. Add and configure slides with figures or text
3. Preview and export HTML presentations
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any


# API configuration
API_BASE_URL = "http://localhost:8000"


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
    """Get available figures for a model run.

    Args:
        run_id: The model run ID
        n_clusters: Number of clusters (if provided, includes clustering figures)
        clustering_method: Clustering method ("kmeans" or "hierarchical")
    """
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
    """Run clustering on a model run.

    Args:
        run_id: The model run ID
        n_clusters: Number of clusters (None for auto-detection)
        max_k: Maximum k for auto-detection
        method: Clustering method ("kmeans" or "hierarchical")

    Returns:
        Clustering results dict or None on failure
    """
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


def create_presentation(name: str, description: str = "", client_name: str = "", project_name: str = "") -> Optional[Dict]:
    """Create a new presentation."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/presentations",
            json={
                "name": name,
                "description": description or None,
                "client_name": client_name or None,
                "project_name": project_name or None,
            },
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
        # Filter out None values
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


def export_presentation(presentation_id: str) -> Optional[bytes]:
    """Export presentation as HTML."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/presentations/{presentation_id}/export",
            timeout=60
        )
        if response.status_code == 200:
            return response.content
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
    if "pb_step" not in st.session_state:
        st.session_state.pb_step = 1
    if "pb_presentation_id" not in st.session_state:
        st.session_state.pb_presentation_id = None
    if "pb_selected_runs" not in st.session_state:
        st.session_state.pb_selected_runs = set()
    # Clustering settings per run: {run_id: {"enabled": bool, "n_clusters": int, "method": str}}
    if "pb_clustering_settings" not in st.session_state:
        st.session_state.pb_clustering_settings = {}


# =============================================================================
# PAGE COMPONENTS
# =============================================================================

def render_step_indicator(current_step: int):
    """Render step indicator."""
    steps = ["Select Runs", "Configure Slides", "Preview & Export"]
    cols = st.columns(len(steps))

    for i, (col, step_name) in enumerate(zip(cols, steps)):
        step_num = i + 1
        with col:
            if step_num < current_step:
                st.markdown(f"✅ **{step_num}. {step_name}**")
            elif step_num == current_step:
                st.markdown(f"🔵 **{step_num}. {step_name}**")
            else:
                st.markdown(f"⚪ {step_num}. {step_name}")


def render_step1_select_runs():
    """Step 1: Select model runs to include in presentation."""
    st.header("Step 1: Select Model Runs")
    st.write("Choose the completed model runs you want to include in your presentation.")

    # Get completed runs
    runs = get_completed_runs()

    if not runs:
        st.warning("No completed model runs found. Run some analyses first!")
        return

    # Group runs by model type
    runs_by_type: Dict[str, List[Dict]] = {}
    for run in runs:
        model_type = run.get("model_type", "unknown")
        if model_type not in runs_by_type:
            runs_by_type[model_type] = []
        runs_by_type[model_type].append(run)

    # Display runs grouped by type
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Available Runs")

        for model_type, type_runs in sorted(runs_by_type.items()):
            with st.expander(f"{get_model_display_name(model_type)} ({len(type_runs)} runs)", expanded=True):
                for run in type_runs:
                    run_id = run["id"]
                    run_name = run.get("name") or f"Run {run_id[:8]}"
                    completed_at = run.get("completed_at", "")
                    if completed_at:
                        try:
                            dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                            completed_str = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            completed_str = completed_at
                    else:
                        completed_str = ""

                    is_selected = run_id in st.session_state.pb_selected_runs

                    col_check, col_name, col_date = st.columns([0.5, 2, 1])
                    with col_check:
                        if st.checkbox("", value=is_selected, key=f"run_select_{run_id}"):
                            st.session_state.pb_selected_runs.add(run_id)
                        elif run_id in st.session_state.pb_selected_runs:
                            st.session_state.pb_selected_runs.discard(run_id)

                    with col_name:
                        st.write(run_name)

                    with col_date:
                        st.caption(completed_str)

    with col2:
        st.subheader("Selected Runs")
        if st.session_state.pb_selected_runs:
            st.success(f"{len(st.session_state.pb_selected_runs)} runs selected")
            for run_id in st.session_state.pb_selected_runs:
                # Find run info
                run_info = next((r for r in runs if r["id"] == run_id), None)
                if run_info:
                    run_name = run_info.get("name") or f"Run {run_id[:8]}"
                    st.write(f"• {run_name}")
        else:
            st.info("No runs selected yet")

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col3:
        if st.button("Next: Configure Slides →", type="primary", disabled=len(st.session_state.pb_selected_runs) == 0):
            st.session_state.pb_step = 2
            st.rerun()


def render_step2_configure_slides():
    """Step 2: Configure presentation slides."""
    st.header("Step 2: Configure Slides")

    # Presentation metadata
    with st.expander("Presentation Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            presentation_name = st.text_input(
                "Presentation Name",
                value=st.session_state.get("pb_name", "Market Structure Analysis"),
                key="pb_name_input"
            )
            client_name = st.text_input(
                "Client Name (optional)",
                value=st.session_state.get("pb_client", ""),
                key="pb_client_input"
            )
        with col2:
            presentation_desc = st.text_area(
                "Description (optional)",
                value=st.session_state.get("pb_desc", ""),
                key="pb_desc_input",
                height=68
            )
            project_name = st.text_input(
                "Project Name (optional)",
                value=st.session_state.get("pb_project", ""),
                key="pb_project_input"
            )

        # Save to session state
        st.session_state.pb_name = presentation_name
        st.session_state.pb_desc = presentation_desc
        st.session_state.pb_client = client_name
        st.session_state.pb_project = project_name

    # Initialize slides in session state if needed
    if "pb_slides" not in st.session_state:
        st.session_state.pb_slides = []

    st.subheader("Slides")

    # Display existing slides
    if st.session_state.pb_slides:
        for i, slide in enumerate(st.session_state.pb_slides):
            with st.container():
                col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])

                with col1:
                    st.write(f"**{i + 1}**")

                with col2:
                    slide_type_icon = "📊" if slide.get("slide_type") == "figure" else "📝"
                    st.write(f"{slide_type_icon} {slide.get('title', 'Untitled')}")
                    if slide.get("description"):
                        st.caption(slide["description"][:50] + "..." if len(slide.get("description", "")) > 50 else slide.get("description", ""))

                with col3:
                    # Move buttons
                    move_col1, move_col2 = st.columns(2)
                    with move_col1:
                        if st.button("↑", key=f"move_up_{i}", disabled=i == 0):
                            slides = st.session_state.pb_slides
                            slides[i], slides[i-1] = slides[i-1], slides[i]
                            st.rerun()
                    with move_col2:
                        if st.button("↓", key=f"move_down_{i}", disabled=i == len(st.session_state.pb_slides) - 1):
                            slides = st.session_state.pb_slides
                            slides[i], slides[i+1] = slides[i+1], slides[i]
                            st.rerun()

                with col4:
                    if st.button("🗑️", key=f"delete_slide_{i}"):
                        st.session_state.pb_slides.pop(i)
                        st.rerun()

                st.divider()
    else:
        st.info("No slides added yet. Use the form below to add slides.")

    # Add slide form
    st.subheader("Add New Slide")

    slide_type = st.radio(
        "Slide Type",
        ["Figure", "Text"],
        horizontal=True,
        key="new_slide_type"
    )

    new_slide_title = st.text_input("Slide Title", key="new_slide_title")
    new_slide_desc = st.text_area("Slide Description (optional)", key="new_slide_desc", height=68)

    if slide_type == "Figure":
        # Select run
        runs = get_completed_runs()
        selected_runs = [r for r in runs if r["id"] in st.session_state.pb_selected_runs]

        if selected_runs:
            run_options = {f"{r.get('name') or r['id'][:8]} ({r['model_type']})": r["id"] for r in selected_runs}
            selected_run_display = st.selectbox("Select Model Run", options=list(run_options.keys()), key="new_slide_run")
            selected_run_id = run_options.get(selected_run_display)

            # Clustering configuration for this run
            if selected_run_id:
                with st.expander("🔍 Clustering Options", expanded=False):
                    st.write("Enable clustering to add cluster-colored biplots and clustering analysis figures.")

                    # Get current clustering settings for this run
                    current_settings = st.session_state.pb_clustering_settings.get(selected_run_id, {
                        "enabled": False,
                        "n_clusters": None,
                        "method": "kmeans"
                    })

                    enable_clustering = st.checkbox(
                        "Enable Clustering",
                        value=current_settings.get("enabled", False),
                        key=f"clustering_enabled_{selected_run_id}"
                    )

                    if enable_clustering:
                        col_clust1, col_clust2 = st.columns(2)
                        with col_clust1:
                            auto_detect = st.checkbox(
                                "Auto-detect optimal clusters",
                                value=current_settings.get("n_clusters") is None,
                                key=f"auto_clusters_{selected_run_id}"
                            )

                            if not auto_detect:
                                n_clusters = st.number_input(
                                    "Number of Clusters",
                                    min_value=2,
                                    max_value=15,
                                    value=current_settings.get("n_clusters") or 3,
                                    key=f"n_clusters_{selected_run_id}"
                                )
                            else:
                                n_clusters = None

                        with col_clust2:
                            clustering_method = st.selectbox(
                                "Clustering Method",
                                options=["kmeans", "hierarchical"],
                                index=0 if current_settings.get("method", "kmeans") == "kmeans" else 1,
                                key=f"clustering_method_{selected_run_id}"
                            )

                        # Run clustering button
                        if st.button("Run Clustering", key=f"run_clustering_{selected_run_id}"):
                            with st.spinner("Running clustering analysis..."):
                                result = run_clustering(
                                    selected_run_id,
                                    n_clusters=n_clusters,
                                    method=clustering_method
                                )
                                if result:
                                    detected_k = result.get("n_clusters", n_clusters)
                                    st.session_state.pb_clustering_settings[selected_run_id] = {
                                        "enabled": True,
                                        "n_clusters": detected_k,
                                        "method": clustering_method,
                                        "result": result
                                    }
                                    st.success(f"Clustering complete! Found {detected_k} clusters.")
                                    st.rerun()
                                else:
                                    st.error("Clustering failed. Please try again.")

                        # Show current clustering status
                        if current_settings.get("result"):
                            st.info(f"Clustering active: {current_settings.get('n_clusters')} clusters ({current_settings.get('method')})")
                    else:
                        # Clear clustering settings if disabled
                        if selected_run_id in st.session_state.pb_clustering_settings:
                            del st.session_state.pb_clustering_settings[selected_run_id]

                # Get figures for selected run (with clustering if enabled)
                clustering_settings = st.session_state.pb_clustering_settings.get(selected_run_id, {})
                if clustering_settings.get("enabled") and clustering_settings.get("n_clusters"):
                    figures = get_run_figures(
                        selected_run_id,
                        n_clusters=clustering_settings["n_clusters"],
                        clustering_method=clustering_settings.get("method", "kmeans")
                    )
                else:
                    figures = get_run_figures(selected_run_id)

                available_figures = [f for f in figures if f.get("available", True)]

                if available_figures:
                    figure_options = {f"{f['name']}": f["type"] for f in available_figures}
                    selected_figure_display = st.selectbox("Select Figure", options=list(figure_options.keys()), key="new_slide_figure")
                    selected_figure_type = figure_options.get(selected_figure_display)
                else:
                    st.warning("No figures available for this run.")
                    selected_figure_type = None
            else:
                selected_figure_type = None
        else:
            st.warning("Please select at least one run in Step 1.")
            selected_run_id = None
            selected_figure_type = None
    else:
        # Text slide
        new_slide_text = st.text_area(
            "Content (Markdown supported)",
            key="new_slide_text",
            height=200,
            help="You can use Markdown formatting: **bold**, *italic*, - bullet points, etc."
        )
        selected_run_id = None
        selected_figure_type = None

    # Add slide button
    if st.button("Add Slide", type="primary"):
        if not new_slide_title:
            st.error("Please enter a slide title.")
        elif slide_type == "Figure" and (not selected_run_id or not selected_figure_type):
            st.error("Please select a model run and figure type.")
        else:
            new_slide = {
                "title": new_slide_title,
                "description": new_slide_desc or None,
                "slide_type": "figure" if slide_type == "Figure" else "text",
            }

            if slide_type == "Figure":
                new_slide["model_run_id"] = selected_run_id
                new_slide["figure_type"] = selected_figure_type

                # Include clustering config for clustering figures
                clustering_figure_types = ["clustered_biplot", "silhouette_analysis", "cluster_sizes", "dendrogram"]
                if selected_figure_type in clustering_figure_types:
                    clustering_settings = st.session_state.pb_clustering_settings.get(selected_run_id, {})
                    if clustering_settings.get("n_clusters"):
                        new_slide["figure_config"] = {
                            "n_clusters": clustering_settings["n_clusters"],
                            "clustering_method": clustering_settings.get("method", "kmeans")
                        }
            else:
                new_slide["text_content"] = new_slide_text

            st.session_state.pb_slides.append(new_slide)
            st.success("Slide added!")
            st.rerun()

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Back to Select Runs"):
            st.session_state.pb_step = 1
            st.rerun()

    with col3:
        if st.button("Next: Preview & Export →", type="primary", disabled=len(st.session_state.pb_slides) == 0):
            st.session_state.pb_step = 3
            st.rerun()


def render_step3_preview_export():
    """Step 3: Preview and export presentation."""
    st.header("Step 3: Preview & Export")

    # Summary
    st.subheader("Presentation Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Name", st.session_state.get("pb_name", "Untitled"))
    with col2:
        st.metric("Slides", len(st.session_state.get("pb_slides", [])))
    with col3:
        st.metric("Selected Runs", len(st.session_state.get("pb_selected_runs", set())))

    if st.session_state.get("pb_client"):
        st.write(f"**Client:** {st.session_state.pb_client}")
    if st.session_state.get("pb_project"):
        st.write(f"**Project:** {st.session_state.pb_project}")

    # Branding options
    with st.expander("Branding Options"):
        col1, col2 = st.columns(2)
        with col1:
            primary_color = st.color_picker(
                "Primary Color",
                value=st.session_state.get("pb_primary_color", "#667eea"),
                key="pb_primary_color_input"
            )
            st.session_state.pb_primary_color = primary_color
        with col2:
            secondary_color = st.color_picker(
                "Secondary Color",
                value=st.session_state.get("pb_secondary_color", "#764ba2"),
                key="pb_secondary_color_input"
            )
            st.session_state.pb_secondary_color = secondary_color

        logo_url = st.text_input(
            "Logo URL (optional)",
            value=st.session_state.get("pb_logo_url", ""),
            key="pb_logo_url_input"
        )
        st.session_state.pb_logo_url = logo_url

    # Slide list preview
    st.subheader("Slides")
    slides = st.session_state.get("pb_slides", [])
    for i, slide in enumerate(slides):
        slide_type_icon = "📊" if slide.get("slide_type") == "figure" else "📝"
        st.write(f"{i + 1}. {slide_type_icon} **{slide.get('title', 'Untitled')}**")
        if slide.get("description"):
            st.caption(f"   {slide['description']}")

    st.divider()

    # Generate and export
    st.subheader("Generate Presentation")

    if st.button("🚀 Generate HTML Presentation", type="primary"):
        with st.spinner("Creating presentation..."):
            # Create presentation
            presentation = create_presentation(
                name=st.session_state.get("pb_name", "Market Structure Analysis"),
                description=st.session_state.get("pb_desc", ""),
                client_name=st.session_state.get("pb_client", ""),
                project_name=st.session_state.get("pb_project", "")
            )

            if not presentation:
                st.error("Failed to create presentation. Check the API connection.")
                return

            presentation_id = presentation["id"]

            # Update branding if set
            branding_options = {
                "primary_color": st.session_state.get("pb_primary_color", "#667eea"),
                "secondary_color": st.session_state.get("pb_secondary_color", "#764ba2"),
            }
            if st.session_state.get("pb_logo_url"):
                branding_options["logo_url"] = st.session_state.pb_logo_url

            update_presentation(presentation_id, branding_options=branding_options)

            # Add slides
            for slide_data in slides:
                add_slide(presentation_id, slide_data)

            # Export HTML
            html_content = export_presentation(presentation_id)

            if html_content:
                st.success("Presentation generated successfully!")

                # Download button
                filename = f"{st.session_state.get('pb_name', 'presentation').replace(' ', '_')}.html"
                st.download_button(
                    label="📥 Download HTML Presentation",
                    data=html_content,
                    file_name=filename,
                    mime="text/html",
                    type="primary"
                )

                # Preview
                with st.expander("Preview Presentation", expanded=True):
                    st.components.v1.html(html_content.decode("utf-8"), height=800, scrolling=True)

                # Store presentation ID
                st.session_state.pb_presentation_id = presentation_id
            else:
                st.error("Failed to export presentation.")

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Back to Configure Slides"):
            st.session_state.pb_step = 2
            st.rerun()

    with col3:
        if st.button("Start New Presentation"):
            # Reset state
            st.session_state.pb_step = 1
            st.session_state.pb_selected_runs = set()
            st.session_state.pb_slides = []
            st.session_state.pb_presentation_id = None
            st.session_state.pb_name = "Market Structure Analysis"
            st.session_state.pb_desc = ""
            st.session_state.pb_client = ""
            st.session_state.pb_project = ""
            st.session_state.pb_clustering_settings = {}
            st.rerun()


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Presentation Builder",
        page_icon="📑",
        layout="wide"
    )

    st.title("📑 Presentation Builder")
    st.write("Create professional presentations by combining insights from multiple model analyses.")

    # Initialize session state
    init_session_state()

    # Step indicator
    render_step_indicator(st.session_state.pb_step)

    st.divider()

    # Render current step
    if st.session_state.pb_step == 1:
        render_step1_select_runs()
    elif st.session_state.pb_step == 2:
        render_step2_configure_slides()
    elif st.session_state.pb_step == 3:
        render_step3_preview_export()


if __name__ == "__main__":
    main()
