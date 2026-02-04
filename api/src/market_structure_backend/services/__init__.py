"""Services for the market structure backend."""

from .figures import (
    get_available_figures,
    generate_figure,
    FigureType,
)
from .presentation_generator import generate_presentation_html

__all__ = [
    "get_available_figures",
    "generate_figure",
    "FigureType",
    "generate_presentation_html",
]
