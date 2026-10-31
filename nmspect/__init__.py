"""Пакет постобработки, спектрометрии, экспорта и 3D-реконструкции данных ОФЭКТ (NMSpectToolkit)."""

from __future__ import annotations

from nmspect.detector import (
    ProjectionBuilder,
    ScintillationProcessor,
    build_projection_for_energy_window,
    generate_projections_from_events,
)
from nmspect.exporters import DicomNMExporter, NeXusExporter, VtkParaviewExporter
from nmspect.io import (
    SimulationDataExtractor,
    clear_hdf5_status_flags_if_needed,
    parse_gantry_angle_from_filename,
)
from nmspect.pipelines import (
    CollectedProjectionDataset,
    collect_projection_data,
    export_projection_study,
    run_isotope_nema_pipeline,
)
from nmspect.reconstruction import (
    FBPReconstructor,
    MLEMReconstructor,
    OSEMReconstructor,
    ReconstructionResult,
    TomographicProjector,
)
from nmspect.spectrometry import (
    EnergyWindowConfig,
    SpectrumProcessor,
    SpectrumResult,
    calculate_dew_coefficient,
    calculate_tew_coefficients,
    get_energy_window_preset,
    list_available_presets,
)

__version__ = "0.2.0"

__all__ = [
    "CollectedProjectionDataset",
    "DicomNMExporter",
    "EnergyWindowConfig",
    "FBPReconstructor",
    "MLEMReconstructor",
    "NeXusExporter",
    "OSEMReconstructor",
    "ProjectionBuilder",
    "ReconstructionResult",
    "ScintillationProcessor",
    "SimulationDataExtractor",
    "SpectrumProcessor",
    "SpectrumResult",
    "TomographicProjector",
    "VtkParaviewExporter",
    "__version__",
    "build_projection_for_energy_window",
    "calculate_dew_coefficient",
    "calculate_tew_coefficients",
    "clear_hdf5_status_flags_if_needed",
    "collect_projection_data",
    "export_projection_study",
    "generate_projections_from_events",
    "get_energy_window_preset",
    "list_available_presets",
    "parse_gantry_angle_from_filename",
    "run_isotope_nema_pipeline",
]
