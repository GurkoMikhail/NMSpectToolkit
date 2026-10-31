"""Публичный интерфейс подпакета ввода-вывода симуляционных данных и конвертации форматов."""

from __future__ import annotations

from nmspect.io.hdf5_reader import (
    SimulationDataExtractor,
    clear_hdf5_status_flags_if_needed,
    parse_gantry_angle_from_filename,
)
from nmspect.io.legacy_formats import (
    save_projections_as_npy,
    save_projections_as_spr,
    save_spr_as_png,
)

__all__ = [
    "SimulationDataExtractor",
    "clear_hdf5_status_flags_if_needed",
    "parse_gantry_angle_from_filename",
    "save_projections_as_npy",
    "save_projections_as_spr",
    "save_spr_as_png",
]
