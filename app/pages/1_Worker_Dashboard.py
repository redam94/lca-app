"""
Worker Dashboard - Monitor ARQ Worker Status
=============================================

A Streamlit page for monitoring ARQ workers, job queues, and model runs.
Provides real-time visibility into the background task processing system.
"""

import streamlit as st
import requests
from datetime import datetime, timezone
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"


def get_health_status():
    """Fetch health status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_worker_status():
    """Fetch detailed worker status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/workers/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_recent_runs(limit: int = 20):
    """Fetch recent model runs from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/runs",
            params={"limit": limit, "order_by": "created_at", "order_dir": "desc"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_datetime(dt_str: str) -> str:
    """Format datetime string for display."""
    if not dt_str:
        return "-"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        # Convert to local time for display
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def get_status_color(status: str) -> str:
    """Get color for status badge."""
    colors = {
        "completed": "green",
        "running": "blue",
        "queued": "orange",
        "pending": "orange",
        "failed": "red",
        "cancelled": "gray",
    }
    return colors.get(status.lower(), "gray")


def main():
    st.set_page_config(
        page_title="Worker Dashboard",
        page_icon="",
        layout="wide"
    )

    st.title("Worker Dashboard")
    st.markdown("Monitor ARQ workers, job queues, and model run status.")

    # Auto-refresh control
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.selectbox("Interval", [5, 10, 30, 60], index=1)

    # Manual refresh button
    with col1:
        if st.button("Refresh Now"):
            st.rerun()

    # Health Status Section
    st.subheader("System Health")

    health = get_health_status()

    if health is None:
        st.error("Cannot connect to API. Is the backend running?")
        st.code("cd api && uv run backend-api")
        return

    # Health metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_emoji = "" if health.get("status") == "healthy" else ""
        st.metric("Status", f"{status_emoji} {health.get('status', 'unknown').title()}")

    with col2:
        redis_emoji = "" if health.get("redis_connected") else ""
        st.metric("Redis", f"{redis_emoji} {'Connected' if health.get('redis_connected') else 'Disconnected'}")

    with col3:
        db_emoji = "" if health.get("database_connected") else ""
        st.metric("Database", f"{db_emoji} {'Connected' if health.get('database_connected') else 'Disconnected'}")

    with col4:
        workers = health.get("workers_active", 0)
        worker_emoji = "" if workers > 0 else ""
        st.metric("Active Workers", f"{worker_emoji} {workers}")

    st.divider()

    # Detailed Worker Status
    st.subheader("Queue Status")

    worker_status = get_worker_status()

    if worker_status and worker_status.get("redis_connected"):
        queue_stats = worker_status.get("queue_stats", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Queued Jobs", queue_stats.get("queued_jobs", 0))
        with col2:
            st.metric("In Progress", queue_stats.get("in_progress_jobs", 0))
        with col3:
            st.metric("Completed (cached)", queue_stats.get("completed_jobs", 0))
        with col4:
            st.metric("Total Redis Keys", queue_stats.get("total_keys", 0))

        # Worker Details
        workers = worker_status.get("workers", [])
        if workers:
            st.markdown("**Active Workers:**")
            for w in workers:
                with st.expander(f"Worker: {w.get('worker_id', 'unknown')}", expanded=True):
                    if w.get("current_job"):
                        st.write(f"Current Job: `{w['current_job']}`")
                    else:
                        st.write("Idle")
                    if w.get("last_health_check"):
                        st.write(f"Last Health Check: {format_datetime(w['last_health_check'])}")
        else:
            st.warning("No active workers detected. Start the worker with:")
            st.code("cd api && uv run backend-worker")
    else:
        st.error("Cannot get queue status. Redis may be disconnected.")

    st.divider()

    # Pending/Running Model Runs
    st.subheader("Active Model Runs")

    if worker_status:
        pending_runs = worker_status.get("pending_runs", [])

        if pending_runs:
            for run in pending_runs:
                status = run.get("status", "unknown")
                color = get_status_color(status)

                with st.expander(
                    f":{color}[{status.upper()}] {run.get('model_type', 'unknown')} - {run.get('id', '')[:8]}...",
                    expanded=status == "running"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Run ID:** `{run.get('id')}`")
                        st.write(f"**Model Type:** {run.get('model_type')}")
                        st.write(f"**Name:** {run.get('name') or '-'}")
                    with col2:
                        st.write(f"**Created:** {format_datetime(run.get('created_at'))}")
                        st.write(f"**Started:** {format_datetime(run.get('started_at'))}")
                        progress = run.get("progress", 0)
                        st.progress(progress, text=f"{progress*100:.1f}%")

                    if run.get("progress_message"):
                        st.info(run["progress_message"])
        else:
            st.info("No active model runs.")

    st.divider()

    # Recent Model Runs Table
    st.subheader("Recent Model Runs")

    runs_response = get_recent_runs(limit=20)

    if runs_response and runs_response.get("runs"):
        runs = runs_response["runs"]

        # Create a data table
        table_data = []
        for run in runs:
            table_data.append({
                "Status": run.get("status", "").upper(),
                "Model": run.get("model_type", ""),
                "Name": run.get("name") or "-",
                "Created": format_datetime(run.get("created_at")),
                "Duration": format_duration(run.get("run_duration")),
                "Progress": f"{run.get('progress', 0)*100:.0f}%",
                "ID": run.get("id", "")[:8] + "...",
            })

        # Display as dataframe with colored status
        st.dataframe(
            table_data,
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="small",
                ),
                "Model": st.column_config.TextColumn(
                    "Model",
                    width="medium",
                ),
                "Progress": st.column_config.ProgressColumn(
                    "Progress",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
            }
        )

        # Summary stats
        st.caption(f"Showing {len(runs)} of {runs_response.get('total', len(runs))} total runs")

        # Status breakdown
        status_counts = {}
        for run in runs:
            status = run.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        cols = st.columns(len(status_counts))
        for i, (status, count) in enumerate(status_counts.items()):
            with cols[i]:
                st.metric(status.title(), count)

    else:
        st.info("No model runs found.")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
