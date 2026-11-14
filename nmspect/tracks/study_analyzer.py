"""Потоковый мультиракурсный анализатор треков с O(1) потреблением памяти и сериализацией сводки исследования."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from hepunits import keV, mm, rad
from numpy.typing import NDArray

from nmspect.detector.projection_builder import build_projection_for_energy_window
from nmspect.io.hdf5_reader import parse_gantry_angle_from_filename
from nmspect.spectrometry.models import EmissionBandConfig, EnergyWindowConfig, PhotonComponent
from nmspect.spectrometry.presets import (
    TB161_EMISSION_BANDS,
    TC99M_EMISSION_BANDS,
    infer_emission_bands_from_energies,
    normalize_isotope_name,
)
from nmspect.spectrometry.scatter import calculate_dew_coefficient, calculate_tew_coefficients
from nmspect.tracks.analytics import compute_interdetector_crosstalk
from nmspect.tracks.models import TrackBatch
from nmspect.tracks.pipeline import TrackProcessingPipeline
from nmspect.tracks.taxonomy import (
    DEFAULT_TOPOLOGY_COLORS,
    DEFAULT_TRACK_TAXONOMY,
    EXCLUSIVE_TOPOLOGY_NAMES,
    TOPOLOGY_LABELS_RU,
    TaxonomyScheme,
    classify_tracks,
)


@dataclass(frozen=True)
class ScatterCalibrationWindowSpec:
    """Спецификация фотопикового энергетического окна с парными подокнами оценки рассеяния DEW/TEW.

    Все границы энергий задаются в базовых единицах `hepunits` (`MeV`).
    """

    name: str
    label: str
    main_lower: float
    main_upper: float
    low_sub_lower: float | None = None
    low_sub_upper: float | None = None
    high_sub_lower: float | None = None
    high_sub_upper: float | None = None
    nominal_dew_factor: float = 0.5

    def __post_init__(self) -> None:
        """Проверяет согласованность границ основного окна и вспомогательных подокон рассеяния."""
        if self.main_lower < 0.0 or self.main_upper <= self.main_lower:
            raise ValueError(f"Некорректные границы основного окна '{self.name}': [{self.main_lower}, {self.main_upper}].")
        if (self.low_sub_lower is None) != (self.low_sub_upper is None):
            raise ValueError(f"Для нижнего подокна '{self.name}' должны быть заданы обе границы либо обе None.")
        if (
            self.low_sub_lower is not None
            and self.low_sub_upper is not None
            and (self.low_sub_lower < 0.0 or self.low_sub_upper <= self.low_sub_lower)
        ):
            raise ValueError(f"Некорректные границы нижнего подокна '{self.name}'.")
        if (self.high_sub_lower is None) != (self.high_sub_upper is None):
            raise ValueError(f"Для верхнего подокна '{self.name}' должны быть заданы обе границы либо обе None.")
        if (
            self.high_sub_lower is not None
            and self.high_sub_upper is not None
            and (self.high_sub_lower < 0.0 or self.high_sub_upper <= self.high_sub_lower)
        ):
            raise ValueError(f"Некорректные границы верхнего подокна '{self.name}'.")

    @property
    def has_dew(self) -> bool:
        """Возвращает `True`, если задано нижнее вспомогательное подокно для оценки DEW."""
        return self.low_sub_lower is not None and self.low_sub_upper is not None

    @property
    def has_tew(self) -> bool:
        """Возвращает `True`, если заданы оба боковых подокна для трапецеидальной оценки TEW."""
        return (
            self.low_sub_lower is not None
            and self.low_sub_upper is not None
            and self.high_sub_lower is not None
            and self.high_sub_upper is not None
        )

    def dew_standard_weight(self) -> float:
        """Возвращает положительный весовой множитель стандартного метода DEW."""
        if not self.has_dew or self.low_sub_lower is None or self.low_sub_upper is None:
            return 0.0
        return abs(
            calculate_dew_coefficient(
                self.main_lower,
                self.main_upper,
                self.low_sub_lower,
                self.low_sub_upper,
                k_factor=self.nominal_dew_factor,
            )
        )

    def tew_standard_weights(self) -> tuple[float, float]:
        """Возвращает пару положительных весов `(weight_low, weight_high)` стандартной трапеции Огавы TEW."""
        if (
            not self.has_tew
            or self.low_sub_lower is None
            or self.low_sub_upper is None
            or self.high_sub_lower is None
            or self.high_sub_upper is None
        ):
            return 0.0, 0.0
        wt_low, wt_high = calculate_tew_coefficients(
            self.main_lower,
            self.main_upper,
            self.low_sub_lower,
            self.low_sub_upper,
            self.high_sub_lower,
            self.high_sub_upper,
        )
        return abs(float(wt_low)), abs(float(wt_high))

    def to_energy_window_config(self) -> EnergyWindowConfig:
        """Преобразует спецификацию в стандартный объект `EnergyWindowConfig`."""
        assoc_list: list[EnergyWindowConfig] = []
        if self.has_tew and self.low_sub_lower is not None and self.low_sub_upper is not None and self.high_sub_lower is not None and self.high_sub_upper is not None:
            wt_low, wt_high = self.tew_standard_weights()
            assoc_list.append(
                EnergyWindowConfig(
                    name=f"{self.name}_low_sub",
                    lower_bound=self.low_sub_lower,
                    upper_bound=self.low_sub_upper,
                    is_scatter_window=True,
                    scatter_weight=-wt_low,
                )
            )
            assoc_list.append(
                EnergyWindowConfig(
                    name=f"{self.name}_high_sub",
                    lower_bound=self.high_sub_lower,
                    upper_bound=self.high_sub_upper,
                    is_scatter_window=True,
                    scatter_weight=-wt_high,
                )
            )
        elif self.has_dew and self.low_sub_lower is not None and self.low_sub_upper is not None:
            dew_wt = self.dew_standard_weight()
            assoc_list.append(
                EnergyWindowConfig(
                    name=f"{self.name}_low_sub",
                    lower_bound=self.low_sub_lower,
                    upper_bound=self.low_sub_upper,
                    is_scatter_window=True,
                    scatter_weight=-dew_wt,
                )
            )
        return EnergyWindowConfig(
            name=self.name,
            lower_bound=self.main_lower,
            upper_bound=self.main_upper,
            primary_peak=0.5 * (self.main_lower + self.main_upper),
            description=self.label,
            associated_windows=tuple(assoc_list),
        )


def default_calibration_windows(isotope: str = "161Tb") -> tuple[ScatterCalibrationWindowSpec, ...]:
    """Возвращает канонический набор аналитических окон с подокнами DEW/TEW для заданного радионуклида."""
    canonical = normalize_isotope_name(isotope)
    if canonical == "161Tb":
        return (
            ScatterCalibrationWindowSpec(
                name="peak75",
                label="Окно 74,6 кэВ (67,1–82,0 кэВ)",
                main_lower=67.1 * keV,
                main_upper=82.0 * keV,
                low_sub_lower=59.7 * keV,
                low_sub_upper=67.1 * keV,
                high_sub_lower=82.0 * keV,
                high_sub_upper=89.5 * keV,
            ),
            ScatterCalibrationWindowSpec(
                name="peak49",
                label="Окно 48,9 кэВ (44,0–53,8 кэВ)",
                main_lower=44.0 * keV,
                main_upper=53.8 * keV,
                low_sub_lower=39.1 * keV,
                low_sub_upper=44.0 * keV,
                high_sub_lower=53.8 * keV,
                high_sub_upper=58.7 * keV,
            ),
        )
    return (
        ScatterCalibrationWindowSpec(
            name="peak140",
            label="Окно 140,5 кэВ (126,5–154,6 кэВ)",
            main_lower=126.5 * keV,
            main_upper=154.6 * keV,
            low_sub_lower=112.4 * keV,
            low_sub_upper=126.5 * keV,
            high_sub_lower=154.6 * keV,
            high_sub_upper=168.6 * keV,
        ),
    )


def default_emission_bands(isotope: str = "161Tb") -> tuple[EmissionBandConfig, ...]:
    """Возвращает предустановленные энергетические полосы начальной эмиссии для радионуклида."""
    canonical = normalize_isotope_name(isotope)
    if canonical == "161Tb":
        return TB161_EMISSION_BANDS
    return TC99M_EMISSION_BANDS


@dataclass(frozen=True)
class TrackStudyConfig:
    """Конфигурация потокового многоракурсного исследования треков ОФЭКТ.

    Все физические величины задаются строго в базовых единицах `hepunits` (`mm`, `MeV`, `rad`).
    """

    study_title: str = "Исследование физических компонент и топологии треков ОФЭКТ"
    isotope: str = "161Tb"
    calibration_windows: tuple[ScatterCalibrationWindowSpec, ...] = field(
        default_factory=lambda: default_calibration_windows("161Tb")
    )
    emission_bands: tuple[EmissionBandConfig, ...] | None = field(
        default_factory=lambda: default_emission_bands("161Tb")
    )
    spectrum_max_energy: float = 620.0 * keV
    num_spectrum_bins: int = 1240
    psf_fine_extent: float = 0.25
    num_psf_fine_bins: int = 200
    psf_wide_extent: float = 0.80
    num_psf_wide_bins: int = 320
    psf_2d_extent: float = 0.24
    num_psf_2d_bins: int = 160
    azimuth_ring_min: float = 0.055
    azimuth_ring_max: float = 0.24
    num_azimuth_bins: int = 72
    crystal_thickness: float = 9.5 * mm
    num_depth_bins: int = 190
    detector_width: float = 540.0 * mm
    detector_height: float = 400.0 * mm
    projection_pixel_size: float = 4.0 * mm
    reference_view_index: int = 0
    vtk_sample_per_view: int = 200
    energy_resolution: float = 0.099
    reference_energy: float = 140.5 * keV
    intrinsic_resolution: float = 3.8 * mm
    require_energy_deposit: bool = False
    random_seed: int = 42
    taxonomy_scheme: TaxonomyScheme = field(default_factory=lambda: DEFAULT_TRACK_TAXONOMY)

    @classmethod
    def for_isotope(
        cls,
        isotope: str = "161Tb",
        study_title: str | None = None,
        **overrides: Any,
    ) -> TrackStudyConfig:
        """Создаёт преднастроенную конфигурацию исследования для выбранного радионуклида."""
        canonical = normalize_isotope_name(isotope)
        max_energy = 620.0 * keV if canonical == "161Tb" else 220.0 * keV
        num_bins = 1240 if canonical == "161Tb" else 440
        resolved_title = (
            study_title
            if study_title is not None
            else f"Физико-статистический анализ треков гамма-квантов {canonical} ОФЭКТ"
        )
        base_kwargs: dict[str, Any] = {
            "study_title": resolved_title,
            "isotope": canonical,
            "calibration_windows": default_calibration_windows(canonical),
            "emission_bands": default_emission_bands(canonical),
            "spectrum_max_energy": max_energy,
            "num_spectrum_bins": num_bins,
        }
        base_kwargs.update(overrides)
        return cls(**base_kwargs)


@dataclass
class TrackStudySummary:
    """Агрегированная сводка многоракурсного исследования треков, готовая к экспорту и генерации отчёта."""

    study_title: str
    isotope: str
    num_files: int
    num_detectors: int
    num_views: int
    total_tracks: int
    total_points: int
    unique_particles: int
    shared_particles: int
    calibration_windows: tuple[ScatterCalibrationWindowSpec, ...]
    emission_bands: tuple[EmissionBandConfig, ...]
    window_totals: dict[str, int]
    window_topology_counts: dict[str, dict[str, int]]
    window_band_counts: dict[str, dict[str, int]]
    window_band_topology_counts: dict[str, dict[str, dict[str, int]]]
    window_mean_tan_sq: dict[str, dict[str, float]]
    window_mean_tan_radial_sq: dict[str, dict[str, float]]
    window_mean_depth: dict[str, dict[str, float]]
    window_mean_septa: dict[str, dict[str, float]]
    optimal_calibration_factors: dict[str, dict[str, float]]
    arrays: dict[str, NDArray[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    sampled_tracks: TrackBatch | None = None
    taxonomy_scheme_name: str = "default_10class"
    exclusive_class_names: tuple[str, ...] = EXCLUSIVE_TOPOLOGY_NAMES
    primary_class_names: tuple[str, ...] = ("geometric_full",)
    topology_labels: dict[str, str] = field(default_factory=lambda: dict(TOPOLOGY_LABELS_RU))
    topology_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TOPOLOGY_COLORS))

    def primary_counts(self, window_key: str) -> int:
        """Возвращает суммарное число треков целевого первичного сигнала (`is_primary_signal`) в указанном окне."""
        if window_key not in self.window_topology_counts:
            raise KeyError(f"Неизвестный ключ энергетического окна '{window_key}'.")
        counts_map = self.window_topology_counts[window_key]
        return sum(int(counts_map.get(cls_name, 0)) for cls_name in self.primary_class_names)

    def global_topology_fractions(self, window_key: str = "full_spectrum") -> dict[str, float]:
        """Возвращает доли взаимоисключающих классов таксономии (в интервале `[0, 1]`) для указанного окна."""
        if window_key not in self.window_totals:
            raise KeyError(f"Неизвестный ключ энергетического окна '{window_key}'.")
        total_cnt = max(self.window_totals[window_key], 1)
        counts_map = self.window_topology_counts[window_key]
        return {
            topo_name: float(counts_map[topo_name]) / float(total_cnt)
            for topo_name in self.exclusive_class_names
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Преобразует скалярную статистику и метаданные сводки в JSON-сериализуемый словарь."""
        return {
            "study_title": self.study_title,
            "isotope": self.isotope,
            "num_files": self.num_files,
            "num_detectors": self.num_detectors,
            "num_views": self.num_views,
            "total_tracks": self.total_tracks,
            "total_points": self.total_points,
            "unique_particles": self.unique_particles,
            "shared_particles": self.shared_particles,
            "taxonomy_scheme_name": self.taxonomy_scheme_name,
            "exclusive_class_names": list(self.exclusive_class_names),
            "primary_class_names": list(self.primary_class_names),
            "topology_labels": self.topology_labels,
            "topology_colors": self.topology_colors,
            "calibration_windows": [
                {
                    "name": win_spec.name,
                    "label": win_spec.label,
                    "main_lower": win_spec.main_lower,
                    "main_upper": win_spec.main_upper,
                    "low_sub_lower": win_spec.low_sub_lower,
                    "low_sub_upper": win_spec.low_sub_upper,
                    "high_sub_lower": win_spec.high_sub_lower,
                    "high_sub_upper": win_spec.high_sub_upper,
                    "nominal_dew_factor": win_spec.nominal_dew_factor,
                }
                for win_spec in self.calibration_windows
            ],
            "emission_bands": [
                {
                    "name": band_cfg.name,
                    "lower_bound": band_cfg.lower_bound,
                    "upper_bound": None if np.isinf(band_cfg.upper_bound) else band_cfg.upper_bound,
                    "label": band_cfg.label,
                    "include_lower": band_cfg.include_lower,
                }
                for band_cfg in self.emission_bands
            ],
            "window_totals": self.window_totals,
            "window_topology_counts": self.window_topology_counts,
            "window_band_counts": self.window_band_counts,
            "window_band_topology_counts": self.window_band_topology_counts,
            "window_mean_tan_sq": self.window_mean_tan_sq,
            "window_mean_tan_radial_sq": self.window_mean_tan_radial_sq,
            "window_mean_depth": self.window_mean_depth,
            "window_mean_septa": self.window_mean_septa,
            "optimal_calibration_factors": self.optimal_calibration_factors,
            "metadata_summary": _sanitize_metadata_for_json(self.metadata),
        }

    def save_bundle(self, json_path: str | Path, npz_path: str | Path) -> tuple[Path, Path]:
        """Сохраняет сводку исследования в пару файлов `.json` (скалярные таблицы) и `.npz` (гистограммы/карты)."""
        resolved_json = Path(json_path)
        resolved_npz = Path(npz_path)
        resolved_json.parent.mkdir(parents=True, exist_ok=True)
        resolved_npz.parent.mkdir(parents=True, exist_ok=True)

        resolved_json.write_text(
            json.dumps(self.to_summary_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez_compressed(resolved_npz, allow_pickle=False, **self.arrays)
        return resolved_json, resolved_npz

    @classmethod
    def load_bundle(cls, json_path: str | Path, npz_path: str | Path) -> TrackStudySummary:
        """Восстанавливает экземпляр `TrackStudySummary` из сохранённых файлов `.json` и `.npz` (без pickle)."""
        raw_doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
        with np.load(Path(npz_path), allow_pickle=False) as archive:
            loaded_arrays = {arr_key: archive[arr_key] for arr_key in archive.files}

        cal_windows = tuple(
            ScatterCalibrationWindowSpec(
                name=str(item["name"]),
                label=str(item["label"]),
                main_lower=float(item["main_lower"]),
                main_upper=float(item["main_upper"]),
                low_sub_lower=float(item["low_sub_lower"]) if item["low_sub_lower"] is not None else None,
                low_sub_upper=float(item["low_sub_upper"]) if item["low_sub_upper"] is not None else None,
                high_sub_lower=float(item["high_sub_lower"]) if item["high_sub_lower"] is not None else None,
                high_sub_upper=float(item["high_sub_upper"]) if item["high_sub_upper"] is not None else None,
                nominal_dew_factor=float(item["nominal_dew_factor"]),
            )
            for item in raw_doc["calibration_windows"]
        )
        em_bands = tuple(
            EmissionBandConfig(
                name=str(item["name"]),
                lower_bound=float(item["lower_bound"]),
                upper_bound=float("inf") if item["upper_bound"] is None else float(item["upper_bound"]),
                label=str(item["label"]),
                include_lower=bool(item["include_lower"]),
            )
            for item in raw_doc["emission_bands"]
        )

        raw_exc_names = raw_doc.get("exclusive_class_names")
        resolved_exc_names = (
            tuple(str(name_item) for name_item in raw_exc_names)
            if raw_exc_names is not None
            else EXCLUSIVE_TOPOLOGY_NAMES
        )
        raw_pri_names = raw_doc.get("primary_class_names")
        resolved_pri_names = (
            tuple(str(name_item) for name_item in raw_pri_names)
            if raw_pri_names is not None
            else ("geometric_full",)
        )
        raw_labels = raw_doc.get("topology_labels")
        resolved_labels = (
            {str(key_l): str(val_l) for key_l, val_l in raw_labels.items()}
            if isinstance(raw_labels, dict)
            else dict(TOPOLOGY_LABELS_RU)
        )
        raw_colors = raw_doc.get("topology_colors")
        resolved_colors = (
            {str(key_c): str(val_c) for key_c, val_c in raw_colors.items()}
            if isinstance(raw_colors, dict)
            else dict(DEFAULT_TOPOLOGY_COLORS)
        )

        return cls(
            study_title=str(raw_doc["study_title"]),
            isotope=str(raw_doc["isotope"]),
            num_files=int(raw_doc["num_files"]),
            num_detectors=int(raw_doc["num_detectors"]),
            num_views=int(raw_doc["num_views"]),
            total_tracks=int(raw_doc["total_tracks"]),
            total_points=int(raw_doc["total_points"]),
            unique_particles=int(raw_doc["unique_particles"]),
            shared_particles=int(raw_doc["shared_particles"]),
            calibration_windows=cal_windows,
            emission_bands=em_bands,
            window_totals={str(key_w): int(val_w) for key_w, val_w in raw_doc["window_totals"].items()},
            window_topology_counts=raw_doc["window_topology_counts"],
            window_band_counts=raw_doc["window_band_counts"],
            window_band_topology_counts=raw_doc["window_band_topology_counts"],
            window_mean_tan_sq=raw_doc["window_mean_tan_sq"],
            window_mean_tan_radial_sq=raw_doc["window_mean_tan_radial_sq"],
            window_mean_depth=raw_doc["window_mean_depth"],
            window_mean_septa=raw_doc["window_mean_septa"],
            optimal_calibration_factors=raw_doc["optimal_calibration_factors"],
            arrays=loaded_arrays,
            metadata=raw_doc.get("metadata_summary", {}),
            sampled_tracks=None,
            taxonomy_scheme_name=str(raw_doc.get("taxonomy_scheme_name", "default_10class")),
            exclusive_class_names=resolved_exc_names,
            primary_class_names=resolved_pri_names,
            topology_labels=resolved_labels,
            topology_colors=resolved_colors,
        )


def _sanitize_metadata_for_json(value: Any) -> Any:
    """Рекурсивно приводит массивы NumPy и скаляры в метаданных к базовым типам JSON."""
    if isinstance(value, dict):
        return {str(key_item): _sanitize_metadata_for_json(val_item) for key_item, val_item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_for_json(elem) for elem in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _detect_num_heads_from_batch_and_meta(tracks: TrackBatch, metadata: dict[str, Any]) -> tuple[int, ...]:
    """Определяет упорядоченный кортеж идентификаторов детекторных головок (`1, 2, ...`)."""
    found_ids: set[int] = set()
    if isinstance(metadata, dict):
        vol_mats = metadata.get("volume_matrices")
        if isinstance(vol_mats, dict):
            for vol_key in vol_mats:
                if vol_key.startswith("Detector_"):
                    suffix_str = vol_key.removeprefix("Detector_")
                    if suffix_str.isdigit():
                        found_ids.add(int(suffix_str))
    if "detector_id" in tracks.cell_data and tracks.num_tracks > 0:
        for raw_det in np.unique(tracks.cell_data["detector_id"]):
            found_ids.add(int(raw_det))
    if not found_ids:
        return (1,)
    return tuple(sorted(found_ids))


def _resolve_canonical_diagnostic_masks(
    tracks: TrackBatch,
    all_masks: dict[str, NDArray[np.bool_]],
    component_codes: NDArray[np.uint8],
    primary_mask: NDArray[np.bool_],
) -> dict[str, NDArray[np.bool_]]:
    """Формирует канонические физические маски для построения ФРТ, DOI и азимута при любой схеме таксономии."""
    full_abs: NDArray[np.bool_] = (
        np.asarray(tracks.cell_data["crystal_photoelectric"]) >= 1
        if "crystal_photoelectric" in tracks.cell_data
        else np.ones(tracks.num_tracks, dtype=np.bool_)
    )
    septal_comp = component_codes == np.uint8(int(PhotonComponent.SEPTAL_PENETRATION))

    if "phantom_compton_single" in all_masks and "phantom_inelastic_multi" in all_masks:
        phantom_compton_all = all_masks["phantom_compton_single"] | all_masks["phantom_inelastic_multi"]
    elif "phantom_compton_all" in all_masks:
        phantom_compton_all = all_masks["phantom_compton_all"]
    else:
        phantom_compton_all = component_codes == np.uint8(int(PhotonComponent.PHANTOM_SCATTER))

    return {
        "geometric_full": all_masks.get("geometric_full", primary_mask),
        "non_geometric": all_masks.get("non_geometric", ~primary_mask),
        "any_ext_scatter": all_masks.get(
            "any_ext_scatter",
            np.isin(
                component_codes,
                (
                    int(PhotonComponent.PHANTOM_SCATTER),
                    int(PhotonComponent.COLLIMATOR_SCATTER),
                    int(PhotonComponent.BACKSCATTER),
                ),
            ),
        ),
        "septal_full": all_masks.get("septal_full", septal_comp & full_abs),
        "septal_escape": all_masks.get("septal_escape", septal_comp & (~full_abs)),
        "phantom_compton_all": phantom_compton_all,
        "collimator_scatter_no_back": all_masks.get(
            "collimator_scatter_no_back",
            component_codes == np.uint8(int(PhotonComponent.COLLIMATOR_SCATTER)),
        ),
        "backscatter": all_masks.get(
            "backscatter",
            component_codes == np.uint8(int(PhotonComponent.BACKSCATTER)),
        ),
    }


class StreamingTrackStudyAnalyzer:
    """Потоковый однопроходный анализатор серии HDF5-файлов ОФЭКТ с ограниченным потреблением ОЗУ."""

    def __init__(
        self,
        config: TrackStudyConfig | None = None,
        pipeline: TrackProcessingPipeline | None = None,
    ) -> None:
        """Инициализирует потоковый анализатор по конфигурации `TrackStudyConfig`."""
        self.config: TrackStudyConfig = config if config is not None else TrackStudyConfig()
        self.rng: np.random.Generator = np.random.default_rng(self.config.random_seed)
        self.pipeline: TrackProcessingPipeline = (
            pipeline
            if pipeline is not None
            else TrackProcessingPipeline(
                max_tracks_per_detector=None,
                energy_resolution=self.config.energy_resolution,
                reference_energy=self.config.reference_energy,
                intrinsic_resolution=self.config.intrinsic_resolution,
                random_seed=self.config.random_seed,
                rng=self.rng,
                require_energy_deposit=self.config.require_energy_deposit,
                taxonomy_scheme=self.config.taxonomy_scheme,
            )
        )

    def analyze_directory(
        self,
        input_dir: str | Path,
        max_files: int | None = None,
    ) -> TrackStudySummary:
        """Находит и сортирует по углу гантри все HDF5-файлы в директории и выполняет их потоковый анализ."""
        dir_path = Path(input_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Директория с файлами симуляции не найдена: {dir_path}")
        if max_files is not None and max_files <= 0:
            raise ValueError(f"Параметр max_files должен быть положительным: {max_files}.")

        hdf_candidates = sorted(
            file_item
            for file_item in dir_path.iterdir()
            if file_item.is_file() and file_item.suffix.lower() in (".hdf", ".hdf5", ".h5")
        )
        if not hdf_candidates:
            raise FileNotFoundError(f"В директории '{dir_path}' не найдено файлов HDF5 (.hdf, .hdf5, .h5).")

        def _sort_key(path_item: Path) -> tuple[float, str]:
            parsed_angle = parse_gantry_angle_from_filename(path_item.name)
            return (float(parsed_angle) if parsed_angle is not None else float("inf"), path_item.name)

        ordered_files = sorted(hdf_candidates, key=_sort_key)
        if max_files is not None:
            ordered_files = ordered_files[:max_files]
        return self.analyze_files(ordered_files)

    def analyze_files(self, file_paths: Sequence[str | Path]) -> TrackStudySummary:
        """Выполняет полный однопроходный статистический анализ списка файлов HDF5."""
        resolved_paths = [Path(item) for item in file_paths]
        if not resolved_paths:
            raise ValueError("Список входных файлов file_paths не может быть пустым.")
        for path_item in resolved_paths:
            if not path_item.is_file():
                raise FileNotFoundError(f"Входной файл симуляции не найден: {path_item}")

        num_files = len(resolved_paths)
        cfg = self.config

        first_angle = parse_gantry_angle_from_filename(resolved_paths[0].name)
        first_batch, primary_metadata = self.pipeline.process_file(
            file_path=resolved_paths[0],
            projection_index=0,
            gantry_angle=float(first_angle) if first_angle is not None else None,
        )
        detector_ids_tuple = _detect_num_heads_from_batch_and_meta(first_batch, primary_metadata)
        num_detectors = len(detector_ids_tuple)
        num_views = num_files * num_detectors

        resolved_bands: tuple[EmissionBandConfig, ...]
        if cfg.emission_bands is not None and len(cfg.emission_bands) > 0:
            resolved_bands = cfg.emission_bands
        elif first_batch.num_tracks > 0 and "emission_energy" in first_batch.cell_data:
            resolved_bands = infer_emission_bands_from_energies(first_batch.cell_data["emission_energy"])
        else:
            resolved_bands = default_emission_bands(cfg.isotope)

        scheme = cfg.taxonomy_scheme
        exclusive_names = scheme.exclusive_names
        all_mask_names = scheme.all_mask_names
        primary_names = scheme.primary_class_names

        window_keys_list: list[str] = ["full_spectrum"]
        for win_spec in cfg.calibration_windows:
            window_keys_list.append(win_spec.name)
            if win_spec.has_dew:
                window_keys_list.append(f"{win_spec.name}_low_sub")
            if win_spec.has_tew:
                window_keys_list.append(f"{win_spec.name}_high_sub")
        if len(cfg.calibration_windows) >= 2:
            window_keys_list.append("dual_main")
        window_keys = tuple(window_keys_list)
        band_names = tuple(band_item.name for band_item in resolved_bands)
        metric_group_names = tuple(dict.fromkeys(("total", *all_mask_names, *band_names)))

        window_totals: dict[str, int] = {win_k: 0 for win_k in window_keys}
        window_topology_counts: dict[str, dict[str, int]] = {
            win_k: {comp_k: 0 for comp_k in all_mask_names} for win_k in window_keys
        }
        window_band_counts: dict[str, dict[str, int]] = {
            win_k: {band_k: 0 for band_k in band_names} for win_k in window_keys
        }
        window_band_topology_counts: dict[str, dict[str, dict[str, int]]] = {
            win_k: {
                band_k: {comp_k: 0 for comp_k in all_mask_names}
                for band_k in band_names
            }
            for win_k in window_keys
        }

        tan_sq_sums: dict[str, dict[str, float]] = {
            win_k: {grp_k: 0.0 for grp_k in metric_group_names} for win_k in window_keys
        }
        tan_rad_sq_sums: dict[str, dict[str, float]] = {
            win_k: {grp_k: 0.0 for grp_k in metric_group_names} for win_k in window_keys
        }
        depth_sums: dict[str, dict[str, float]] = {
            win_k: {grp_k: 0.0 for grp_k in metric_group_names} for win_k in window_keys
        }
        septa_sums: dict[str, dict[str, int]] = {
            win_k: {grp_k: 0 for grp_k in metric_group_names} for win_k in window_keys
        }
        group_counts: dict[str, dict[str, int]] = {
            win_k: {grp_k: 0 for grp_k in metric_group_names} for win_k in window_keys
        }

        spectrum_edges = np.linspace(0.0, cfg.spectrum_max_energy, cfg.num_spectrum_bins + 1, dtype=np.float64)
        psf_fine_edges = np.linspace(-cfg.psf_fine_extent, cfg.psf_fine_extent, cfg.num_psf_fine_bins + 1, dtype=np.float64)
        psf_wide_edges = np.linspace(-cfg.psf_wide_extent, cfg.psf_wide_extent, cfg.num_psf_wide_bins + 1, dtype=np.float64)
        psf_2d_edges = np.linspace(-cfg.psf_2d_extent, cfg.psf_2d_extent, cfg.num_psf_2d_bins + 1, dtype=np.float64)
        azimuth_edges = np.linspace(0.0, float(2.0 * np.pi * rad), cfg.num_azimuth_bins + 1, dtype=np.float64)
        depth_edges = np.linspace(0.0, cfg.crystal_thickness, cfg.num_depth_bins + 1, dtype=np.float64)

        det_size_2d = (cfg.detector_width, cfg.detector_height)
        num_proj_u = max(1, round(cfg.detector_width / cfg.projection_pixel_size))
        num_proj_v = max(1, round(cfg.detector_height / cfg.projection_pixel_size))
        proj_u_edges = np.linspace(-0.5 * cfg.detector_width, 0.5 * cfg.detector_width, num_proj_u + 1, dtype=np.float64)
        proj_v_edges = np.linspace(-0.5 * cfg.detector_height, 0.5 * cfg.detector_height, num_proj_v + 1, dtype=np.float64)

        proj_win_configs: dict[str, EnergyWindowConfig] = {}
        for win_spec in cfg.calibration_windows:
            proj_win_configs[win_spec.name] = EnergyWindowConfig(
                name=win_spec.name,
                lower_bound=win_spec.main_lower,
                upper_bound=win_spec.main_upper,
            )
            if win_spec.has_dew and win_spec.low_sub_lower is not None and win_spec.low_sub_upper is not None:
                proj_win_configs[f"{win_spec.name}_low_sub"] = EnergyWindowConfig(
                    name=f"{win_spec.name}_low_sub",
                    lower_bound=win_spec.low_sub_lower,
                    upper_bound=win_spec.low_sub_upper,
                )
            if win_spec.has_tew and win_spec.high_sub_lower is not None and win_spec.high_sub_upper is not None:
                proj_win_configs[f"{win_spec.name}_high_sub"] = EnergyWindowConfig(
                    name=f"{win_spec.name}_high_sub",
                    lower_bound=win_spec.high_sub_lower,
                    upper_bound=win_spec.high_sub_upper,
                )

        arrays: dict[str, NDArray[Any]] = {
            "spectrum_edges": spectrum_edges,
            "psf_fine_edges": psf_fine_edges,
            "psf_wide_edges": psf_wide_edges,
            "psf_2d_edges": psf_2d_edges,
            "azimuth_edges": azimuth_edges,
            "depth_edges": depth_edges,
            "proj_u_edges": proj_u_edges,
            "proj_v_edges": proj_v_edges,
            "view_angles": np.zeros(num_views, dtype=np.float64),
            "spec_total": np.zeros(cfg.num_spectrum_bins, dtype=np.int64),
        }

        for mask_name in (*all_mask_names, *band_names):
            arrays[f"spec_{mask_name}"] = np.zeros(cfg.num_spectrum_bins, dtype=np.int64)

        psf_subset_names = tuple(
            dict.fromkeys(
                (
                    "total",
                    "geometric_full",
                    *exclusive_names,
                    "non_geometric",
                    "any_ext_scatter",
                    *band_names,
                )
            )
        )
        psf2d_subset_names = (
            "total",
            "geometric_full",
            "septal_full",
            "septal_escape",
            "phantom_compton_all",
            "collimator_scatter_no_back",
            "backscatter",
        )
        depth_subset_names = (
            "total",
            "geometric_full",
            "septal_full",
            "septal_escape",
            "phantom_compton_all",
            "collimator_scatter_no_back",
            "backscatter",
        )

        for dep_tag in depth_subset_names:
            arrays[f"depth_full_spectrum_{dep_tag}"] = np.zeros(cfg.num_depth_bins, dtype=np.int64)
            arrays[f"septa_full_spectrum_{dep_tag}"] = np.zeros(50, dtype=np.int64)

        for band_k in band_names:
            arrays[f"depth_band_{band_k}"] = np.zeros(cfg.num_depth_bins, dtype=np.int64)
            arrays[f"septa_band_{band_k}"] = np.zeros(50, dtype=np.int64)

        for win_spec in cfg.calibration_windows:
            win_sub_keys = [win_spec.name]
            if win_spec.has_dew:
                win_sub_keys.append(f"{win_spec.name}_low_sub")
            if win_spec.has_tew:
                win_sub_keys.append(f"{win_spec.name}_high_sub")

            for sub_key in win_sub_keys:
                for grp_tag in psf_subset_names:
                    arrays[f"psf_fine_{sub_key}_{grp_tag}"] = np.zeros(cfg.num_psf_fine_bins, dtype=np.int64)
                    arrays[f"psf_wide_{sub_key}_{grp_tag}"] = np.zeros(cfg.num_psf_wide_bins, dtype=np.int64)
                    arrays[f"psf_tails_{sub_key}_{grp_tag}"] = np.zeros(3, dtype=np.int64)

            for map_tag in psf2d_subset_names:
                arrays[f"psf2d_{win_spec.name}_{map_tag}"] = np.zeros(
                    (cfg.num_psf_2d_bins, cfg.num_psf_2d_bins), dtype=np.int64
                )
            for az_tag in ("geometric_full", "septal_full", "septal_escape", "phantom_compton_all"):
                arrays[f"azimuth_{win_spec.name}_{az_tag}"] = np.zeros(cfg.num_azimuth_bins, dtype=np.int64)
            for dep_tag in depth_subset_names:
                arrays[f"depth_{win_spec.name}_{dep_tag}"] = np.zeros(cfg.num_depth_bins, dtype=np.int64)
                arrays[f"septa_{win_spec.name}_{dep_tag}"] = np.zeros(50, dtype=np.int64)

            for proj_layer in ("total", "primary", "non_geometric", "low_sub", "high_sub"):
                arrays[f"ref_proj_{win_spec.name}_{proj_layer}"] = np.zeros((num_proj_u, num_proj_v), dtype=np.int64)

            for view_metric in (
                "total_counts",
                "geometric_counts",
                "low_sub_counts",
                "high_sub_counts",
                "true_scatter_fraction",
                "ext_scatter_fraction",
                "septal_fraction",
                "dew_std_fraction",
                "tew_std_fraction",
                "optimal_k_dew",
                "optimal_k_tew",
                "pixel_mean_sf_true",
                "pixel_mean_sf_dew",
                "pixel_mean_sf_tew",
                "neg_pixel_fraction_dew",
                "neg_pixel_fraction_tew",
                "std_tan_total",
                "std_tan_geometric",
                "std_tan_non_geometric",
            ):
                arrays[f"view_{win_spec.name}_{view_metric}"] = np.zeros(num_views, dtype=np.float64)

        total_tracks = 0
        total_points = 0
        unique_particles_total = 0
        shared_particles_total = 0
        sampled_batches: list[TrackBatch] = []

        for file_idx, hdf_path in enumerate(resolved_paths):
            if file_idx == 0:
                batch = first_batch
            else:
                parsed_angle = parse_gantry_angle_from_filename(hdf_path.name)
                batch, _ = self.pipeline.process_file(
                    file_path=hdf_path,
                    projection_index=file_idx,
                    gantry_angle=float(parsed_angle) if parsed_angle is not None else None,
                )

            crosstalk_file = compute_interdetector_crosstalk(batch)
            unique_particles_total += crosstalk_file.unique_particles
            shared_particles_total += crosstalk_file.shared_particles

            det_ids = np.asarray(batch.cell_data["detector_id"], dtype=np.int32)
            det_rank_lookup = {det_val: rank_idx for rank_idx, det_val in enumerate(detector_ids_tuple)}
            rank_offsets = np.zeros(batch.num_tracks, dtype=np.int32)
            for det_val, rank_idx in det_rank_lookup.items():
                rank_offsets[det_ids == det_val] = rank_idx * num_files
            batch.cell_data["projection_index"] = (
                np.asarray(batch.cell_data["projection_index"], dtype=np.int32) + rank_offsets
            )

            total_tracks += batch.num_tracks
            total_points += batch.num_points

            if cfg.vtk_sample_per_view > 0 and batch.num_tracks > 0:
                for det_val in detector_ids_tuple:
                    det_indices = np.flatnonzero(det_ids == det_val)
                    if det_indices.size == 0:
                        continue
                    if det_indices.size > cfg.vtk_sample_per_view:
                        chosen_idx = np.sort(
                            self.rng.choice(det_indices, size=cfg.vtk_sample_per_view, replace=False)
                        )
                    else:
                        chosen_idx = det_indices
                    sampled_batches.append(batch.slice_tracks(chosen_idx))

            if batch.num_tracks == 0:
                continue

            meas_energy = np.asarray(batch.cell_data["measured_energy"], dtype=np.float64)
            emiss_energy = np.asarray(batch.cell_data["emission_energy"], dtype=np.float64)
            tan_u = np.asarray(batch.cell_data["tan_theta"], dtype=np.float64)
            tan_v = np.asarray(batch.cell_data["tan_theta_axial"], dtype=np.float64)
            tan_rad_sq = tan_u * tan_u + tan_v * tan_v
            rad_tan = np.sqrt(tan_rad_sq)
            depth_vals = np.asarray(batch.cell_data["crystal_depth"], dtype=np.float64)
            local_pos = np.asarray(batch.cell_data["detector_local_position"], dtype=np.float64)
            septa_raw = np.asarray(batch.cell_data["septal_crossings"], dtype=np.int64)
            septa_clipped = np.clip(septa_raw, 0, 49)

            classification = classify_tracks(batch, scheme=scheme)
            all_topo_masks = classification.all_masks
            diag_masks = _resolve_canonical_diagnostic_masks(
                tracks=batch,
                all_masks=all_topo_masks,
                component_codes=classification.component_codes,
                primary_mask=classification.primary_mask,
            )
            phantom_compton_all = diag_masks["phantom_compton_all"]

            band_masks: dict[str, NDArray[np.bool_]] = {
                band_cfg.name: band_cfg.matches(emiss_energy) for band_cfg in resolved_bands
            }

            win_masks: dict[str, NDArray[np.bool_]] = {
                "full_spectrum": meas_energy >= 0.0,
            }
            dual_mask = np.zeros(batch.num_tracks, dtype=np.bool_)
            for win_spec in cfg.calibration_windows:
                main_m = (meas_energy >= win_spec.main_lower) & (meas_energy <= win_spec.main_upper)
                win_masks[win_spec.name] = main_m
                dual_mask |= main_m
                if win_spec.has_dew and win_spec.low_sub_lower is not None and win_spec.low_sub_upper is not None:
                    win_masks[f"{win_spec.name}_low_sub"] = (meas_energy >= win_spec.low_sub_lower) & (
                        meas_energy < win_spec.low_sub_upper
                    )
                if win_spec.has_tew and win_spec.high_sub_lower is not None and win_spec.high_sub_upper is not None:
                    win_masks[f"{win_spec.name}_high_sub"] = (meas_energy > win_spec.high_sub_lower) & (
                        meas_energy <= win_spec.high_sub_upper
                    )
            if "dual_main" in window_totals:
                win_masks["dual_main"] = dual_mask

            arrays["spec_total"] += np.histogram(meas_energy, bins=spectrum_edges)[0]
            for topo_name, topo_mask in all_topo_masks.items():
                if bool(np.any(topo_mask)):
                    arrays[f"spec_{topo_name}"] += np.histogram(meas_energy[topo_mask], bins=spectrum_edges)[0]
            for band_name, band_mask in band_masks.items():
                if bool(np.any(band_mask)):
                    arrays[f"spec_{band_name}"] += np.histogram(meas_energy[band_mask], bins=spectrum_edges)[0]
                    arrays[f"depth_band_{band_name}"] += np.histogram(depth_vals[band_mask], bins=depth_edges)[0]
                    arrays[f"septa_band_{band_name}"] += np.bincount(
                        septa_clipped[band_mask], minlength=50
                    )[:50]

            for win_k, win_m in win_masks.items():
                win_cnt = int(np.count_nonzero(win_m))
                window_totals[win_k] += win_cnt
                if win_cnt == 0:
                    continue

                group_counts[win_k]["total"] += win_cnt
                tan_sq_sums[win_k]["total"] += float(np.sum(tan_u[win_m] ** 2))
                tan_rad_sq_sums[win_k]["total"] += float(np.sum(tan_rad_sq[win_m]))
                depth_sums[win_k]["total"] += float(np.sum(depth_vals[win_m]))
                septa_sums[win_k]["total"] += int(np.sum(septa_raw[win_m]))

                for topo_name, topo_mask in all_topo_masks.items():
                    joint_m = win_m & topo_mask
                    joint_cnt = int(np.count_nonzero(joint_m))
                    window_topology_counts[win_k][topo_name] += joint_cnt
                    if joint_cnt > 0:
                        group_counts[win_k][topo_name] += joint_cnt
                        tan_sq_sums[win_k][topo_name] += float(np.sum(tan_u[joint_m] ** 2))
                        tan_rad_sq_sums[win_k][topo_name] += float(np.sum(tan_rad_sq[joint_m]))
                        depth_sums[win_k][topo_name] += float(np.sum(depth_vals[joint_m]))
                        septa_sums[win_k][topo_name] += int(np.sum(septa_raw[joint_m]))

                for band_name, band_mask in band_masks.items():
                    bw_mask = win_m & band_mask
                    bw_cnt = int(np.count_nonzero(bw_mask))
                    window_band_counts[win_k][band_name] += bw_cnt
                    if bw_cnt > 0:
                        group_counts[win_k][band_name] += bw_cnt
                        tan_sq_sums[win_k][band_name] += float(np.sum(tan_u[bw_mask] ** 2))
                        tan_rad_sq_sums[win_k][band_name] += float(np.sum(tan_rad_sq[bw_mask]))
                        depth_sums[win_k][band_name] += float(np.sum(depth_vals[bw_mask]))
                        septa_sums[win_k][band_name] += int(np.sum(septa_raw[bw_mask]))
                    for topo_name, topo_mask in all_topo_masks.items():
                        if bw_cnt > 0:
                            window_band_topology_counts[win_k][band_name][topo_name] += int(
                                np.count_nonzero(bw_mask & topo_mask)
                            )

                if f"psf_fine_{win_k}_total" in arrays:
                    combined_groups_map: dict[str, NDArray[np.bool_]] = {
                        "total": win_m,
                        "geometric_full": win_m & diag_masks["geometric_full"],
                    }
                    for exc_k in exclusive_names:
                        combined_groups_map[exc_k] = win_m & all_topo_masks[exc_k]
                    combined_groups_map["non_geometric"] = win_m & diag_masks["non_geometric"]
                    combined_groups_map["any_ext_scatter"] = win_m & diag_masks["any_ext_scatter"]
                    for bnd_k in band_names:
                        combined_groups_map[bnd_k] = win_m & band_masks[bnd_k]

                    for grp_tag, grp_mask in combined_groups_map.items():
                        if not bool(np.any(grp_mask)):
                            continue
                        sub_tu = tan_u[grp_mask]
                        abs_tu = np.abs(sub_tu)
                        arrays[f"psf_fine_{win_k}_{grp_tag}"] += np.histogram(sub_tu, bins=psf_fine_edges)[0]
                        arrays[f"psf_wide_{win_k}_{grp_tag}"] += np.histogram(sub_tu, bins=psf_wide_edges)[0]
                        arrays[f"psf_tails_{win_k}_{grp_tag}"] += np.array(
                            [
                                sub_tu.size,
                                int(np.count_nonzero(abs_tu > 0.25)),
                                int(np.count_nonzero(abs_tu > 0.60)),
                            ],
                            dtype=np.int64,
                        )

                if f"depth_{win_k}_total" in arrays:
                    depth_map = {
                        "total": win_m,
                        "geometric_full": win_m & diag_masks["geometric_full"],
                        "septal_full": win_m & diag_masks["septal_full"],
                        "septal_escape": win_m & diag_masks["septal_escape"],
                        "phantom_compton_all": win_m & phantom_compton_all,
                        "collimator_scatter_no_back": win_m & diag_masks["collimator_scatter_no_back"],
                        "backscatter": win_m & diag_masks["backscatter"],
                    }
                    for dep_tag, dep_mask in depth_map.items():
                        if bool(np.any(dep_mask)):
                            arrays[f"depth_{win_k}_{dep_tag}"] += np.histogram(
                                depth_vals[dep_mask], bins=depth_edges
                            )[0]
                            arrays[f"septa_{win_k}_{dep_tag}"] += np.bincount(
                                septa_clipped[dep_mask], minlength=50
                            )[:50]

            azimuth_angles = np.mod(np.arctan2(tan_v, tan_u), float(2.0 * np.pi * rad))
            ring_mask = (rad_tan >= cfg.azimuth_ring_min) & (rad_tan <= cfg.azimuth_ring_max)
            geom_ring_mask = (rad_tan >= 0.015) & (rad_tan <= 0.045)

            for win_spec in cfg.calibration_windows:
                win_m = win_masks[win_spec.name]
                if not bool(np.any(win_m)):
                    continue
                maps_2d = {
                    "total": win_m,
                    "geometric_full": win_m & diag_masks["geometric_full"],
                    "septal_full": win_m & diag_masks["septal_full"],
                    "septal_escape": win_m & diag_masks["septal_escape"],
                    "phantom_compton_all": win_m & phantom_compton_all,
                    "collimator_scatter_no_back": win_m & diag_masks["collimator_scatter_no_back"],
                    "backscatter": win_m & diag_masks["backscatter"],
                }
                for map_tag, map_mask in maps_2d.items():
                    if bool(np.any(map_mask)):
                        arrays[f"psf2d_{win_spec.name}_{map_tag}"] += np.histogram2d(
                            tan_u[map_mask],
                            tan_v[map_mask],
                            bins=(psf_2d_edges, psf_2d_edges),
                        )[0].astype(np.int64)

                az_selections = {
                    "geometric_full": win_m & diag_masks["geometric_full"] & geom_ring_mask,
                    "septal_full": win_m & diag_masks["septal_full"] & ((rad_tan >= 0.025) & (rad_tan <= 0.080)),
                    "septal_escape": win_m & diag_masks["septal_escape"] & ring_mask,
                    "phantom_compton_all": win_m & phantom_compton_all & ring_mask,
                }
                for az_tag, az_mask in az_selections.items():
                    if bool(np.any(az_mask)):
                        arrays[f"azimuth_{win_spec.name}_{az_tag}"] += np.histogram(
                            azimuth_angles[az_mask], bins=azimuth_edges
                        )[0]

            gantry_arr = np.asarray(batch.cell_data["gantry_angle"], dtype=np.float64)
            base_file_angle = float(gantry_arr[0]) if gantry_arr.size > 0 else 0.0
            coord_u = local_pos[:, 0]
            coord_v = local_pos[:, 1]

            for rank_idx, det_val in enumerate(detector_ids_tuple):
                proj_idx = file_idx + rank_idx * num_files
                det_mask = det_ids == det_val
                head_shift = (2.0 * np.pi * float(rank_idx) / float(max(num_detectors, 1))) * float(rad)
                arrays["view_angles"][proj_idx] = base_file_angle + head_shift

                for win_spec in cfg.calibration_windows:
                    prefix = f"view_{win_spec.name}"
                    main_det = det_mask & win_masks[win_spec.name]
                    low_det = (
                        det_mask & win_masks[f"{win_spec.name}_low_sub"]
                        if win_spec.has_dew
                        else np.zeros_like(det_mask)
                    )
                    high_det = (
                        det_mask & win_masks[f"{win_spec.name}_high_sub"]
                        if win_spec.has_tew
                        else np.zeros_like(det_mask)
                    )

                    geom_det = main_det & classification.primary_mask
                    non_geom_det = main_det & (~classification.primary_mask)

                    cnt_main = int(np.count_nonzero(main_det))
                    cnt_geom = int(np.count_nonzero(geom_det))
                    cnt_non_geom = cnt_main - cnt_geom
                    cnt_low = int(np.count_nonzero(low_det))
                    cnt_high = int(np.count_nonzero(high_det))
                    safe_main = float(max(cnt_main, 1))

                    dew_wt = win_spec.dew_standard_weight()
                    tew_wt_low, tew_wt_high = win_spec.tew_standard_weights()
                    dew_est = dew_wt * float(cnt_low)
                    tew_est = tew_wt_low * float(cnt_low) + tew_wt_high * float(cnt_high)

                    arrays[f"{prefix}_total_counts"][proj_idx] = float(cnt_main)
                    arrays[f"{prefix}_geometric_counts"][proj_idx] = float(cnt_geom)
                    arrays[f"{prefix}_low_sub_counts"][proj_idx] = float(cnt_low)
                    arrays[f"{prefix}_high_sub_counts"][proj_idx] = float(cnt_high)
                    arrays[f"{prefix}_true_scatter_fraction"][proj_idx] = float(cnt_non_geom) / safe_main
                    arrays[f"{prefix}_ext_scatter_fraction"][proj_idx] = (
                        float(np.count_nonzero(main_det & diag_masks["any_ext_scatter"])) / safe_main
                    )
                    arrays[f"{prefix}_septal_fraction"][proj_idx] = (
                        float(
                            np.count_nonzero(
                                main_det & (diag_masks["septal_full"] | diag_masks["septal_escape"])
                            )
                        )
                        / safe_main
                    )
                    arrays[f"{prefix}_dew_std_fraction"][proj_idx] = min(1.0, dew_est / safe_main)
                    arrays[f"{prefix}_tew_std_fraction"][proj_idx] = min(1.0, tew_est / safe_main)
                    arrays[f"{prefix}_optimal_k_dew"][proj_idx] = (
                        float(cnt_non_geom) / float(cnt_low) if cnt_low > 0 else 0.0
                    )
                    arrays[f"{prefix}_optimal_k_tew"][proj_idx] = (
                        float(cnt_non_geom) / tew_est if tew_est > 0.0 else 0.0
                    )

                    arrays[f"{prefix}_std_tan_total"][proj_idx] = (
                        float(np.std(tan_u[main_det])) if cnt_main >= 2 else 0.0
                    )
                    arrays[f"{prefix}_std_tan_geometric"][proj_idx] = (
                        float(np.std(tan_u[geom_det])) if cnt_geom >= 2 else 0.0
                    )
                    arrays[f"{prefix}_std_tan_non_geometric"][proj_idx] = (
                        float(np.std(tan_u[non_geom_det])) if cnt_non_geom >= 2 else 0.0
                    )

                    main_win_cfg = proj_win_configs[win_spec.name]
                    grid_main = build_projection_for_energy_window(
                        energies=meas_energy[main_det],
                        coords_x=coord_u[main_det],
                        coords_y=coord_v[main_det],
                        detector_size=det_size_2d,
                        pixel_size=cfg.projection_pixel_size,
                        window_config=main_win_cfg,
                        clip_negative=False,
                    )
                    grid_non_geom = build_projection_for_energy_window(
                        energies=meas_energy[non_geom_det],
                        coords_x=coord_u[non_geom_det],
                        coords_y=coord_v[non_geom_det],
                        detector_size=det_size_2d,
                        pixel_size=cfg.projection_pixel_size,
                        window_config=main_win_cfg,
                        clip_negative=False,
                    )
                    if win_spec.has_dew:
                        grid_low = build_projection_for_energy_window(
                            energies=meas_energy[low_det],
                            coords_x=coord_u[low_det],
                            coords_y=coord_v[low_det],
                            detector_size=det_size_2d,
                            pixel_size=cfg.projection_pixel_size,
                            window_config=proj_win_configs[f"{win_spec.name}_low_sub"],
                            clip_negative=False,
                        )
                    else:
                        grid_low = np.zeros((num_proj_u, num_proj_v), dtype=np.float64)

                    if win_spec.has_tew:
                        grid_high = build_projection_for_energy_window(
                            energies=meas_energy[high_det],
                            coords_x=coord_u[high_det],
                            coords_y=coord_v[high_det],
                            detector_size=det_size_2d,
                            pixel_size=cfg.projection_pixel_size,
                            window_config=proj_win_configs[f"{win_spec.name}_high_sub"],
                            clip_negative=False,
                        )
                    else:
                        grid_high = np.zeros((num_proj_u, num_proj_v), dtype=np.float64)

                    nz_pixels = grid_main > 0
                    if bool(np.any(nz_pixels)):
                        denom_pix = grid_main[nz_pixels].astype(np.float64)
                        pix_true = grid_non_geom[nz_pixels] / denom_pix
                        pix_dew = np.minimum(1.0, (dew_wt * grid_low[nz_pixels]) / denom_pix)
                        pix_tew = np.minimum(
                            1.0,
                            (tew_wt_low * grid_low[nz_pixels] + tew_wt_high * grid_high[nz_pixels]) / denom_pix,
                        )
                        arrays[f"{prefix}_pixel_mean_sf_true"][proj_idx] = float(np.mean(pix_true))
                        arrays[f"{prefix}_pixel_mean_sf_dew"][proj_idx] = float(np.mean(pix_dew))
                        arrays[f"{prefix}_pixel_mean_sf_tew"][proj_idx] = float(np.mean(pix_tew))

                        num_nz = float(np.count_nonzero(nz_pixels))
                        raw_dew = grid_main - dew_wt * grid_low
                        raw_tew = grid_main - (tew_wt_low * grid_low + tew_wt_high * grid_high)
                        arrays[f"{prefix}_neg_pixel_fraction_dew"][proj_idx] = (
                            float(np.count_nonzero(nz_pixels & (raw_dew < 0.0))) / num_nz
                        )
                        arrays[f"{prefix}_neg_pixel_fraction_tew"][proj_idx] = (
                            float(np.count_nonzero(nz_pixels & (raw_tew < 0.0))) / num_nz
                        )

                    if proj_idx == cfg.reference_view_index:
                        grid_pri = build_projection_for_energy_window(
                            energies=meas_energy[geom_det],
                            coords_x=coord_u[geom_det],
                            coords_y=coord_v[geom_det],
                            detector_size=det_size_2d,
                            pixel_size=cfg.projection_pixel_size,
                            window_config=main_win_cfg,
                            clip_negative=False,
                        )
                        arrays[f"ref_proj_{win_spec.name}_total"] += grid_main.astype(np.int64)
                        arrays[f"ref_proj_{win_spec.name}_primary"] += grid_pri.astype(np.int64)
                        arrays[f"ref_proj_{win_spec.name}_non_geometric"] += grid_non_geom.astype(np.int64)
                        arrays[f"ref_proj_{win_spec.name}_low_sub"] += grid_low.astype(np.int64)
                        arrays[f"ref_proj_{win_spec.name}_high_sub"] += grid_high.astype(np.int64)

        window_mean_tan_sq: dict[str, dict[str, float]] = {}
        window_mean_tan_radial_sq: dict[str, dict[str, float]] = {}
        window_mean_depth: dict[str, dict[str, float]] = {}
        window_mean_septa: dict[str, dict[str, float]] = {}

        for win_k in window_keys:
            window_mean_tan_sq[win_k] = {}
            window_mean_tan_radial_sq[win_k] = {}
            window_mean_depth[win_k] = {}
            window_mean_septa[win_k] = {}
            for grp_k in metric_group_names:
                cnt_val = group_counts[win_k][grp_k]
                if cnt_val > 0:
                    inv_cnt = 1.0 / float(cnt_val)
                    window_mean_tan_sq[win_k][grp_k] = tan_sq_sums[win_k][grp_k] * inv_cnt
                    window_mean_tan_radial_sq[win_k][grp_k] = tan_rad_sq_sums[win_k][grp_k] * inv_cnt
                    window_mean_depth[win_k][grp_k] = depth_sums[win_k][grp_k] * inv_cnt
                    window_mean_septa[win_k][grp_k] = float(septa_sums[win_k][grp_k]) * inv_cnt
                else:
                    window_mean_tan_sq[win_k][grp_k] = 0.0
                    window_mean_tan_radial_sq[win_k][grp_k] = 0.0
                    window_mean_depth[win_k][grp_k] = 0.0
                    window_mean_septa[win_k][grp_k] = 0.0

        optimal_factors: dict[str, dict[str, float]] = {}
        for win_spec in cfg.calibration_windows:
            tot_main = window_totals[win_spec.name]
            tot_geom = sum(
                int(window_topology_counts[win_spec.name].get(pri_cls, 0)) for pri_cls in primary_names
            )
            tot_non_geom = float(tot_main - tot_geom)
            tot_low = float(window_totals.get(f"{win_spec.name}_low_sub", 0))
            tot_high = float(window_totals.get(f"{win_spec.name}_high_sub", 0))
            tew_wl, tew_wh = win_spec.tew_standard_weights()
            tew_denom = tew_wl * tot_low + tew_wh * tot_high
            optimal_factors[win_spec.name] = {
                "optimal_k_dew": tot_non_geom / tot_low if tot_low > 0.0 else 0.0,
                "optimal_k_tew": tot_non_geom / tew_denom if tew_denom > 0.0 else 0.0,
                "dew_standard_weight": win_spec.dew_standard_weight(),
                "tew_low_weight": tew_wl,
                "tew_high_weight": tew_wh,
            }

        sampled_concat = TrackBatch.concat(sampled_batches) if sampled_batches else None

        return TrackStudySummary(
            study_title=cfg.study_title,
            isotope=cfg.isotope,
            num_files=num_files,
            num_detectors=num_detectors,
            num_views=num_views,
            total_tracks=total_tracks,
            total_points=total_points,
            unique_particles=unique_particles_total,
            shared_particles=shared_particles_total,
            calibration_windows=cfg.calibration_windows,
            emission_bands=resolved_bands,
            window_totals=window_totals,
            window_topology_counts=window_topology_counts,
            window_band_counts=window_band_counts,
            window_band_topology_counts=window_band_topology_counts,
            window_mean_tan_sq=window_mean_tan_sq,
            window_mean_tan_radial_sq=window_mean_tan_radial_sq,
            window_mean_depth=window_mean_depth,
            window_mean_septa=window_mean_septa,
            optimal_calibration_factors=optimal_factors,
            arrays=arrays,
            metadata=primary_metadata,
            sampled_tracks=sampled_concat,
            taxonomy_scheme_name=scheme.name,
            exclusive_class_names=exclusive_names,
            primary_class_names=primary_names,
            topology_labels=scheme.labels_map,
            topology_colors=scheme.colors_map,
        )
