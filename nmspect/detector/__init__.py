"""Публичный интерфейс подпакета физики сцинтилляционного детектора и построения проекций."""

from __future__ import annotations

from nmspect.detector.projection_builder import (
    ProjectionBuilder,
    build_projection_for_energy_window,
    generate_projections_from_events,
)
from nmspect.detector.scintillation import (
    FWHM_TO_SIGMA_FACTOR,
    ScintillationProcessor,
    aggregate_energy_weighted_segments,
    compute_spatial_offsets_2d,
    normalize_energy_resolution,
    smear_scintillation_energy,
)

__all__ = [
    "FWHM_TO_SIGMA_FACTOR",
    "ProjectionBuilder",
    "ScintillationProcessor",
    "aggregate_energy_weighted_segments",
    "build_projection_for_energy_window",
    "compute_spatial_offsets_2d",
    "generate_projections_from_events",
    "normalize_energy_resolution",
    "smear_scintillation_energy",
]
