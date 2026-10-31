"""
Подпакет 2D и 3D томографической реконструкции данных ОФЭКТ (`nmspect.reconstruction`).

Объединяет геометрическое ядро проецирования (`TomographicProjector`), моделирование
затухания и поправку Чанга (`attenuation`), аналитическую фильтрованную обратную
проекцию (`FBPReconstructor`), статистические итерационные алгоритмы (`OSEMReconstructor`,
`MLEMReconstructor`) и утилиты загрузки/экспорта томографических объёмов (`io`).
"""

from __future__ import annotations

from nmspect.reconstruction.attenuation import (
    WATER_LINEAR_ATTENUATION_BY_ISOTOPE,
    apply_chang_attenuation_correction,
    compute_chang_correction_map,
    create_uniform_attenuation_map,
    estimate_attenuation_map_from_projections,
    get_water_linear_attenuation,
)
from nmspect.reconstruction.fbp import FBPReconstructor
from nmspect.reconstruction.io import (
    load_attenuation_volume,
    load_phantom_volume,
    load_projections,
    load_projections_from_dat,
    load_projections_from_dicom,
    load_projections_from_nexus,
    load_projections_from_npy,
    save_slices_preview_png,
    save_volume_as_npy,
    save_volume_as_vti,
)
from nmspect.reconstruction.iterative import (
    MLEMReconstructor,
    OSEMReconstructor,
    ReconstructionResult,
    generate_ordered_subsets,
)
from nmspect.reconstruction.projector import (
    TomographicProjector,
    create_circular_fov_mask,
    gaussian_smooth_volume,
    normalize_projection_angles,
)

__all__ = [
    "WATER_LINEAR_ATTENUATION_BY_ISOTOPE",
    "FBPReconstructor",
    "MLEMReconstructor",
    "OSEMReconstructor",
    "ReconstructionResult",
    "TomographicProjector",
    "apply_chang_attenuation_correction",
    "compute_chang_correction_map",
    "create_circular_fov_mask",
    "create_uniform_attenuation_map",
    "estimate_attenuation_map_from_projections",
    "gaussian_smooth_volume",
    "generate_ordered_subsets",
    "get_water_linear_attenuation",
    "load_attenuation_volume",
    "load_phantom_volume",
    "load_projections",
    "load_projections_from_dat",
    "load_projections_from_dicom",
    "load_projections_from_nexus",
    "load_projections_from_npy",
    "normalize_projection_angles",
    "save_slices_preview_png",
    "save_volume_as_npy",
    "save_volume_as_vti",
]
