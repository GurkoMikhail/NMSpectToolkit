"""Публичный интерфейс подпакета физики сцинтилляционного детектора и построения проекций."""

from __future__ import annotations

from nmspect.detector.projection_builder import (
    ProjectionBuilder,
    build_projection_for_energy_window,
    generate_projections_from_events,
)
from nmspect.detector.scintillation import ScintillationProcessor

__all__ = [
    "ProjectionBuilder",
    "ScintillationProcessor",
    "build_projection_for_energy_window",
    "generate_projections_from_events",
]
