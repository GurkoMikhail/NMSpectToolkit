"""Пакет постобработки, спектрометрии, экспорта и 3D-реконструкции данных ОФЭКТ (NMSpectToolkit)."""

from __future__ import annotations

from nmspect.detector import (
    ProjectionBuilder,
    ScintillationProcessor,
    build_projection_for_energy_window,
    generate_projections_from_events,
)
from nmspect.exporters import DicomNMExporter, NeXusExporter, VtkParaviewExporter, VtkTrackExporter
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
from nmspect.reporting import (
    DocumentFormatConverter,
    ReportMetadataConfig,
    ScientificDocxBuilder,
    TrackStudyReportGenerator,
)
from nmspect.spectrometry import (
    ComponentAnalyzer,
    EmissionBandConfig,
    EnergyWindowConfig,
    PhotonComponent,
    SpectrumProcessor,
    SpectrumResult,
    calculate_dew_coefficient,
    calculate_tew_coefficients,
    fuse_calibrated_windows,
    get_energy_window_preset,
    list_available_presets,
)
from nmspect.tracks import (
    DEFAULT_TRACK_TAXONOMY,
    AuxiliaryMaskRule,
    FieldCondition,
    StreamingTrackStudyAnalyzer,
    TaxonomyScheme,
    TrackBatch,
    TrackClassRule,
    TrackProcessingPipeline,
    TrackStudyConfig,
    TrackStudySummary,
    TrackTopology,
    classify_tracks,
    extract_detector_events_from_tracks,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_TRACK_TAXONOMY",
    "AuxiliaryMaskRule",
    "CollectedProjectionDataset",
    "ComponentAnalyzer",
    "DicomNMExporter",
    "DocumentFormatConverter",
    "EmissionBandConfig",
    "EnergyWindowConfig",
    "FBPReconstructor",
    "FieldCondition",
    "MLEMReconstructor",
    "NeXusExporter",
    "OSEMReconstructor",
    "PhotonComponent",
    "ProjectionBuilder",
    "ReconstructionResult",
    "ReportMetadataConfig",
    "ScientificDocxBuilder",
    "ScintillationProcessor",
    "SimulationDataExtractor",
    "SpectrumProcessor",
    "SpectrumResult",
    "StreamingTrackStudyAnalyzer",
    "TaxonomyScheme",
    "TomographicProjector",
    "TrackBatch",
    "TrackClassRule",
    "TrackProcessingPipeline",
    "TrackStudyConfig",
    "TrackStudyReportGenerator",
    "TrackStudySummary",
    "TrackTopology",
    "VtkParaviewExporter",
    "VtkTrackExporter",
    "__version__",
    "build_projection_for_energy_window",
    "calculate_dew_coefficient",
    "calculate_tew_coefficients",
    "classify_tracks",
    "clear_hdf5_status_flags_if_needed",
    "collect_projection_data",
    "export_projection_study",
    "extract_detector_events_from_tracks",
    "fuse_calibrated_windows",
    "generate_projections_from_events",
    "get_energy_window_preset",
    "list_available_presets",
    "parse_gantry_angle_from_filename",
    "run_isotope_nema_pipeline",
]
