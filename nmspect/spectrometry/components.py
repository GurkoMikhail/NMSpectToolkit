"""Детерминированная классификация физических компонент гамма-излучения, анализ функции рассеяния точки (ФРТ)
и количественная оценка планарного качества проекций фантома с многопиковым слиянием окон.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from hepunits import mm
from numpy.typing import NDArray

from nmspect.spectrometry.models import EnergyWindowConfig, PhotonComponent
from nmspect.tracks.geometry import (
    extract_orthonormal_frame_from_pose,
    locate_hexagonal_channel_axial,
    locate_square_channel_grid,
)
from nmspect.tracks.models import TrackBatch
from nmspect.tracks.taxonomy import classify_tracks


@dataclass(frozen=True)
class WindowComponentBreakdown:
    """Количественная и пространственная декомпозиция физического состава энергетического окна (5 компонент).

    Все пространственные радиусы выражены в базовых единицах `hepunits` (`mm`).
    """

    window_name: str
    raw_total_counts: float
    raw_geometric_counts: float
    raw_septal_counts: float
    raw_collimator_scatter_counts: float
    raw_phantom_scatter_counts: float
    raw_backscatter_counts: float
    effective_total_counts: float
    effective_geometric_counts: float
    effective_septal_counts: float
    effective_collimator_scatter_counts: float
    effective_phantom_scatter_counts: float
    effective_backscatter_counts: float
    geometric_fraction: float
    septal_fraction: float
    collimator_scatter_fraction: float
    phantom_scatter_fraction: float
    backscatter_fraction: float
    geometric_median_radius: float = 0.0
    septal_median_radius: float = 0.0
    collimator_scatter_median_radius: float = 0.0
    phantom_scatter_median_radius: float = 0.0
    backscatter_median_radius: float = 0.0
    geometric_p90_radius: float = 0.0
    septal_p90_radius: float = 0.0
    collimator_scatter_p90_radius: float = 0.0
    phantom_scatter_p90_radius: float = 0.0
    backscatter_p90_radius: float = 0.0


@dataclass(frozen=True)
class ComponentPSFResult:
    """Результат двумерного и радиального разложения функции рассеяния точки (ФРТ) по 5 физическим компонентам.

    Все пространственные координаты и размеры бинов выражены в базовых единицах `hepunits` (`mm`).
    """

    edges_x: NDArray[np.float64]
    edges_y: NDArray[np.float64]
    centers_x: NDArray[np.float64]
    centers_y: NDArray[np.float64]
    psf_total: NDArray[np.float64]
    psf_geometric: NDArray[np.float64]
    psf_septal_penetration: NDArray[np.float64]
    psf_collimator_scatter: NDArray[np.float64]
    psf_phantom_scatter: NDArray[np.float64]
    psf_backscatter: NDArray[np.float64]
    radial_edges: NDArray[np.float64]
    radial_centers: NDArray[np.float64]
    radial_total: NDArray[np.float64]
    radial_geometric: NDArray[np.float64]
    radial_septal_penetration: NDArray[np.float64]
    radial_collimator_scatter: NDArray[np.float64]
    radial_phantom_scatter: NDArray[np.float64]
    radial_backscatter: NDArray[np.float64]
    pixel_size: float
    half_extent: float
    breakdown: WindowComponentBreakdown


@dataclass(frozen=True)
class SphericalInsertSpec:
    """Геометрическое описание сферического очага фантома в мировых 3D-координатах (`mm`)."""

    name: str
    center_world: tuple[float, float, float]
    diameter: float
    activity_concentration_ratio: float = 10.0

    def __post_init__(self) -> None:
        """Проверяет положительность диаметра сферы."""
        if self.diameter <= 0.0:
            raise ValueError(f"Диаметр сферического очага '{self.name}' должен быть строго положительным: {self.diameter}.")


@dataclass(frozen=True)
class SphereContrastMetrics:
    """Количественные метрики контраста и отношения контраст/шум (CNR) одного сферического очага на проекции."""

    name: str
    diameter: float
    center_transverse: float
    center_axial: float
    mean_sphere_signal: float
    local_contrast: float
    contrast_recovery: float
    cnr: float


@dataclass(frozen=True)
class PlanarPhantomQualityResult:
    """Сводный результат оценки качества планарной проекции фантома с очагами."""

    background_mean: float
    background_std: float
    background_cv: float
    nrmse_vs_primary: float
    relative_bias_vs_primary: float
    sphere_metrics: tuple[SphereContrastMetrics, ...]


def classify_collimator_passage(
    birth_local_collimator: NDArray[np.floating[Any]],
    detector_local_collimator: NDArray[np.floating[Any]],
    hole_length: float,
    hole_diameter: float,
    septa_thickness: float,
    hole_shape: str = "hexagonal",
) -> NDArray[np.bool_]:
    """Определяет для нерассеянных фотонов прохождение внутри одного канала коллиматора (`True` — геометрическое).

    Использует единую реализацию гексагональной/квадратной решётки из `nmspect.tracks.geometry`.
    """
    if hole_length <= 0.0:
        raise ValueError(f"Длина каналов коллиматора должна быть положительной: {hole_length}")
    if hole_diameter <= 0.0:
        raise ValueError(f"Диаметр отверстий коллиматора должен быть положительным: {hole_diameter}")
    if septa_thickness <= 0.0:
        raise ValueError(f"Толщина септы должна быть положительной: {septa_thickness}")

    birth_pts = np.asarray(birth_local_collimator, dtype=np.float64)
    det_pts = np.asarray(detector_local_collimator, dtype=np.float64)
    if birth_pts.shape != det_pts.shape or birth_pts.ndim != 2 or birth_pts.shape[1] != 3:
        raise ValueError(
            f"Ожидались совпадающие массивы координат формы (num_photons, 3), "
            f"получено: {birth_pts.shape} и {det_pts.shape}"
        )

    num_photons = int(birth_pts.shape[0])
    if num_photons == 0:
        return np.empty(0, dtype=np.bool_)

    half_length = 0.5 * float(hole_length)
    delta_depth = det_pts[:, 2] - birth_pts[:, 2]
    safe_delta_depth = np.where(
        np.abs(delta_depth) > 1e-12,
        delta_depth,
        np.where(delta_depth >= 0.0, 1e-12, -1e-12),
    )

    direction_sign = np.where(birth_pts[:, 2] >= det_pts[:, 2], 1.0, -1.0)
    depth_enter = direction_sign * half_length
    depth_exit = -direction_sign * half_length

    ratio_enter = (depth_enter - birth_pts[:, 2]) / safe_delta_depth
    ratio_exit = (depth_exit - birth_pts[:, 2]) / safe_delta_depth

    enter_trans = birth_pts[:, 0] + ratio_enter * (det_pts[:, 0] - birth_pts[:, 0])
    enter_axial = birth_pts[:, 1] + ratio_enter * (det_pts[:, 1] - birth_pts[:, 1])
    exit_trans = birth_pts[:, 0] + ratio_exit * (det_pts[:, 0] - birth_pts[:, 0])
    exit_axial = birth_pts[:, 1] + ratio_exit * (det_pts[:, 1] - birth_pts[:, 1])

    if "square" in hole_shape.lower():
        in_hole_enter, idx_first_enter, idx_second_enter = locate_square_channel_grid(
            enter_trans,
            enter_axial,
            hole_diameter=hole_diameter,
            septa_thickness=septa_thickness,
        )
        in_hole_exit, idx_first_exit, idx_second_exit = locate_square_channel_grid(
            exit_trans,
            exit_axial,
            hole_diameter=hole_diameter,
            septa_thickness=septa_thickness,
        )
    else:
        in_hole_enter, idx_first_enter, idx_second_enter = locate_hexagonal_channel_axial(
            enter_trans,
            enter_axial,
            hole_diameter=hole_diameter,
            septa_thickness=septa_thickness,
        )
        in_hole_exit, idx_first_exit, idx_second_exit = locate_hexagonal_channel_axial(
            exit_trans,
            exit_axial,
            hole_diameter=hole_diameter,
            septa_thickness=septa_thickness,
        )

    same_channel = (
        in_hole_enter
        & in_hole_exit
        & (idx_first_enter == idx_first_exit)
        & (idx_second_enter == idx_second_exit)
    )
    return np.asarray(same_channel, dtype=np.bool_)


def _compute_energy_window_weights(
    energies_arr: NDArray[np.float64],
    active_windows: tuple[EnergyWindowConfig, ...],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Возвращает булеву маску попадания в основные окна и вектор весов с учётом вычитания DEW/TEW."""
    num_events = int(energies_arr.size)
    if num_events == 0:
        return np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)

    if not active_windows:
        all_true = np.ones(num_events, dtype=np.bool_)
        all_ones = np.ones(num_events, dtype=np.float64)
        return all_true, all_ones

    in_primary_window_mask = np.zeros(num_events, dtype=np.bool_)
    event_weights = np.zeros(num_events, dtype=np.float64)

    for win_cfg in active_windows:
        main_weight = (
            float(win_cfg.scatter_weight)
            if win_cfg.is_scatter_window and win_cfg.scatter_weight != 0.0
            else 1.0
        )
        main_mask = (energies_arr >= win_cfg.lower_bound) & (energies_arr <= win_cfg.upper_bound)
        in_primary_window_mask |= main_mask
        event_weights[main_mask] += main_weight

        for assoc_win in win_cfg.associated_windows:
            low_cond = (
                energies_arr > assoc_win.lower_bound
                if np.isclose(assoc_win.lower_bound, win_cfg.upper_bound, rtol=0.0, atol=1e-12)
                else energies_arr >= assoc_win.lower_bound
            )
            high_cond = (
                energies_arr < assoc_win.upper_bound
                if np.isclose(assoc_win.upper_bound, win_cfg.lower_bound, rtol=0.0, atol=1e-12)
                else energies_arr <= assoc_win.upper_bound
            )
            assoc_mask = low_cond & high_cond
            event_weights[assoc_mask] += float(assoc_win.scatter_weight)

    return in_primary_window_mask, event_weights


class ComponentAnalyzer:
    """Анализатор физического состава энергетических окон (5 компонент `PhotonComponent`) и ФРТ."""

    def __init__(
        self,
        psf_half_extent: float = 120.0 * mm,
        psf_pixel_size: float = 1.5 * mm,
        num_radial_bins: int = 80,
    ) -> None:
        """Инициализирует параметры пространственной сетки построения функции рассеяния точки (ФРТ)."""
        if psf_half_extent <= 0.0:
            raise ValueError(f"Полуширина области ФРТ должна быть строго положительной: {psf_half_extent}")
        if psf_pixel_size <= 0.0:
            raise ValueError(f"Шаг сетки ФРТ должен быть строго положительным: {psf_pixel_size}")
        if num_radial_bins <= 0:
            raise ValueError(f"Число радиальных бинов должно быть положительным: {num_radial_bins}")

        self.psf_half_extent: float = float(psf_half_extent)
        self.psf_pixel_size: float = float(psf_pixel_size)
        self.num_radial_bins: int = int(num_radial_bins)

    @staticmethod
    def _normalize_windows(
        window_config: EnergyWindowConfig | Sequence[EnergyWindowConfig] | None,
    ) -> tuple[EnergyWindowConfig, ...]:
        """Приводит аргумент энергетического окна к кортежу `EnergyWindowConfig`."""
        if isinstance(window_config, EnergyWindowConfig):
            return (window_config,)
        if window_config is not None:
            return tuple(window_config)
        return ()

    def analyze_window(
        self,
        energies: NDArray[np.floating[Any]],
        photon_components: NDArray[np.integer[Any]],
        window_config: EnergyWindowConfig | Sequence[EnergyWindowConfig] | None = None,
        psf_offsets: NDArray[np.floating[Any]] | None = None,
    ) -> WindowComponentBreakdown:
        """Вычисляет количественный и долевой вклад 5 физических компонент в заданном энергетическом окне."""
        energies_arr = np.asarray(energies, dtype=np.float64).ravel()
        components_arr = np.asarray(photon_components, dtype=np.int8).ravel()
        if energies_arr.size != components_arr.size:
            raise ValueError(
                f"Несовпадение размерностей energies ({energies_arr.size}) "
                f"и photon_components ({components_arr.size})."
            )

        radial_distances: NDArray[np.float64] | None = None
        if psf_offsets is not None:
            offsets_arr = np.asarray(psf_offsets, dtype=np.float64)
            if offsets_arr.ndim != 2 or offsets_arr.shape[0] != energies_arr.size or offsets_arr.shape[1] < 2:
                raise ValueError(
                    f"Массив psf_offsets должен иметь форму ({energies_arr.size}, 2), "
                    f"получено: {offsets_arr.shape}"
                )
            radial_distances = np.hypot(offsets_arr[:, 0], offsets_arr[:, 1])

        active_windows = self._normalize_windows(window_config)
        window_label = "+".join(win.name for win in active_windows) if active_windows else "full_spectrum"

        in_main_mask, event_weights = _compute_energy_window_weights(energies_arr, active_windows)

        geom_mask = components_arr == int(PhotonComponent.GEOMETRIC)
        septal_mask = components_arr == int(PhotonComponent.SEPTAL_PENETRATION)
        coll_scat_mask = components_arr == int(PhotonComponent.COLLIMATOR_SCATTER)
        phant_scat_mask = components_arr == int(PhotonComponent.PHANTOM_SCATTER)
        back_mask = components_arr == int(PhotonComponent.BACKSCATTER)

        raw_geom = float(np.count_nonzero(in_main_mask & geom_mask))
        raw_septal = float(np.count_nonzero(in_main_mask & septal_mask))
        raw_coll_scat = float(np.count_nonzero(in_main_mask & coll_scat_mask))
        raw_phant_scat = float(np.count_nonzero(in_main_mask & phant_scat_mask))
        raw_back = float(np.count_nonzero(in_main_mask & back_mask))
        raw_total = raw_geom + raw_septal + raw_coll_scat + raw_phant_scat + raw_back

        net_geom = float(np.sum(event_weights[geom_mask]))
        net_septal = float(np.sum(event_weights[septal_mask]))
        net_coll_scat = float(np.sum(event_weights[coll_scat_mask]))
        net_phant_scat = float(np.sum(event_weights[phant_scat_mask]))
        net_back = float(np.sum(event_weights[back_mask]))

        # Истинный эффективный счёт окна определяется суммарным потоком весов (строгая аддитивность DEW/TEW)
        eff_total = max(0.0, float(np.sum(event_weights)))
        eff_geom = max(0.0, net_geom)
        eff_septal = max(0.0, net_septal)
        eff_coll_scat = max(0.0, net_coll_scat)
        eff_phant_scat = max(0.0, net_phant_scat)
        eff_back = max(0.0, net_back)

        comp_pos_sum = eff_geom + eff_septal + eff_coll_scat + eff_phant_scat + eff_back
        if comp_pos_sum > 0.0:
            geom_frac = eff_geom / comp_pos_sum
            septal_frac = eff_septal / comp_pos_sum
            coll_scat_frac = eff_coll_scat / comp_pos_sum
            phant_scat_frac = eff_phant_scat / comp_pos_sum
            back_frac = eff_back / comp_pos_sum
        else:
            geom_frac = 0.0
            septal_frac = 0.0
            coll_scat_frac = 0.0
            phant_scat_frac = 0.0
            back_frac = 0.0

        def _radial_quantiles(comp_mask: NDArray[np.bool_]) -> tuple[float, float]:
            if radial_distances is None:
                return 0.0, 0.0
            selected = in_main_mask & comp_mask
            if not bool(np.any(selected)):
                return 0.0, 0.0
            comp_radii = radial_distances[selected]
            return float(np.median(comp_radii)), float(np.percentile(comp_radii, 90.0))

        geom_med, geom_p90 = _radial_quantiles(geom_mask)
        sept_med, sept_p90 = _radial_quantiles(septal_mask)
        coll_med, coll_p90 = _radial_quantiles(coll_scat_mask)
        phant_med, phant_p90 = _radial_quantiles(phant_scat_mask)
        back_med, back_p90 = _radial_quantiles(back_mask)

        return WindowComponentBreakdown(
            window_name=window_label,
            raw_total_counts=raw_total,
            raw_geometric_counts=raw_geom,
            raw_septal_counts=raw_septal,
            raw_collimator_scatter_counts=raw_coll_scat,
            raw_phantom_scatter_counts=raw_phant_scat,
            raw_backscatter_counts=raw_back,
            effective_total_counts=eff_total,
            effective_geometric_counts=eff_geom,
            effective_septal_counts=eff_septal,
            effective_collimator_scatter_counts=eff_coll_scat,
            effective_phantom_scatter_counts=eff_phant_scat,
            effective_backscatter_counts=eff_back,
            geometric_fraction=geom_frac,
            septal_fraction=septal_frac,
            collimator_scatter_fraction=coll_scat_frac,
            phantom_scatter_fraction=phant_scat_frac,
            backscatter_fraction=back_frac,
            geometric_median_radius=geom_med,
            septal_median_radius=sept_med,
            collimator_scatter_median_radius=coll_med,
            phantom_scatter_median_radius=phant_med,
            backscatter_median_radius=back_med,
            geometric_p90_radius=geom_p90,
            septal_p90_radius=sept_p90,
            collimator_scatter_p90_radius=coll_p90,
            phantom_scatter_p90_radius=phant_p90,
            backscatter_p90_radius=back_p90,
        )

    def analyze_track_batch(
        self,
        tracks: TrackBatch,
        window_config: EnergyWindowConfig | Sequence[EnergyWindowConfig] | None = None,
    ) -> WindowComponentBreakdown:
        """Анализирует физический состав энергетического окна напрямую по аннотированному пакету `TrackBatch`."""
        if "measured_energy" not in tracks.cell_data:
            raise KeyError("В tracks.cell_data отсутствует поле 'measured_energy'.")

        if "photon_component" in tracks.cell_data:
            comp_codes = np.asarray(tracks.cell_data["photon_component"], dtype=np.uint8)
        else:
            comp_codes = classify_tracks(tracks).component_codes

        energies_arr = np.asarray(tracks.cell_data["measured_energy"], dtype=np.float64)
        return self.analyze_window(
            energies=energies_arr,
            photon_components=comp_codes,
            window_config=window_config,
        )

    def compute_psf(
        self,
        energies: NDArray[np.floating[Any]],
        psf_offsets: NDArray[np.floating[Any]],
        photon_components: NDArray[np.integer[Any]],
        window_config: EnergyWindowConfig | Sequence[EnergyWindowConfig] | None = None,
        clip_negative: bool = True,
    ) -> ComponentPSFResult:
        """Строит двумерные карты и радиальные профили функции рассеяния точки (ФРТ) по 5 компонентам."""
        energies_arr = np.asarray(energies, dtype=np.float64).ravel()
        components_arr = np.asarray(photon_components, dtype=np.int8).ravel()
        offsets_arr = np.asarray(psf_offsets, dtype=np.float64)

        if energies_arr.size != components_arr.size:
            raise ValueError(
                f"Несовпадение длины energies ({energies_arr.size}) и photon_components ({components_arr.size})."
            )
        if offsets_arr.ndim != 2 or offsets_arr.shape[0] != energies_arr.size or offsets_arr.shape[1] < 2:
            raise ValueError(
                f"Матрица psf_offsets должна иметь форму ({energies_arr.size}, 2), получено: {offsets_arr.shape}"
            )

        num_bins_2d = max(2, round(2.0 * self.psf_half_extent / self.psf_pixel_size))
        edges_x = np.linspace(-self.psf_half_extent, self.psf_half_extent, num_bins_2d + 1, dtype=np.float64)
        edges_y = np.linspace(-self.psf_half_extent, self.psf_half_extent, num_bins_2d + 1, dtype=np.float64)
        centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
        centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])

        radial_edges = np.linspace(0.0, self.psf_half_extent, self.num_radial_bins + 1, dtype=np.float64)
        radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

        active_windows = self._normalize_windows(window_config)
        _, event_weights = _compute_energy_window_weights(energies_arr, active_windows)
        nonzero_weight_mask = event_weights != 0.0

        delta_first = offsets_arr[:, 0]
        delta_second = offsets_arr[:, 1]
        radial_dist = np.hypot(delta_first, delta_second)

        def _build_unclipped_maps(
            selection_mask: NDArray[np.bool_],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            if not bool(np.any(selection_mask)):
                return (
                    np.zeros((num_bins_2d, num_bins_2d), dtype=np.float64),
                    np.zeros(self.num_radial_bins, dtype=np.float64),
                )
            hist_2d, _, _ = np.histogram2d(
                delta_first[selection_mask],
                delta_second[selection_mask],
                bins=(edges_x, edges_y),
                weights=event_weights[selection_mask],
            )
            hist_radial, _ = np.histogram(
                radial_dist[selection_mask],
                bins=radial_edges,
                weights=event_weights[selection_mask],
            )
            return hist_2d.astype(np.float64), hist_radial.astype(np.float64)

        psf_geom_raw, rad_geom_raw = _build_unclipped_maps(
            nonzero_weight_mask & (components_arr == int(PhotonComponent.GEOMETRIC))
        )
        psf_septal_raw, rad_septal_raw = _build_unclipped_maps(
            nonzero_weight_mask & (components_arr == int(PhotonComponent.SEPTAL_PENETRATION))
        )
        psf_coll_raw, rad_coll_raw = _build_unclipped_maps(
            nonzero_weight_mask & (components_arr == int(PhotonComponent.COLLIMATOR_SCATTER))
        )
        psf_phant_raw, rad_phant_raw = _build_unclipped_maps(
            nonzero_weight_mask & (components_arr == int(PhotonComponent.PHANTOM_SCATTER))
        )
        psf_back_raw, rad_back_raw = _build_unclipped_maps(
            nonzero_weight_mask & (components_arr == int(PhotonComponent.BACKSCATTER))
        )

        # Суммарная ФРТ складывается ДО клиппинга в нуль (сохранение аддитивности вычитания рассеяния)
        psf_total = psf_geom_raw + psf_septal_raw + psf_coll_raw + psf_phant_raw + psf_back_raw
        rad_total = rad_geom_raw + rad_septal_raw + rad_coll_raw + rad_phant_raw + rad_back_raw

        psf_geom = np.maximum(0.0, psf_geom_raw) if clip_negative else psf_geom_raw
        psf_septal = np.maximum(0.0, psf_septal_raw) if clip_negative else psf_septal_raw
        psf_coll = np.maximum(0.0, psf_coll_raw) if clip_negative else psf_coll_raw
        psf_phant = np.maximum(0.0, psf_phant_raw) if clip_negative else psf_phant_raw
        psf_back = np.maximum(0.0, psf_back_raw) if clip_negative else psf_back_raw
        if clip_negative:
            np.clip(psf_total, 0.0, None, out=psf_total)
            np.clip(rad_total, 0.0, None, out=rad_total)

        rad_geom = np.maximum(0.0, rad_geom_raw) if clip_negative else rad_geom_raw
        rad_septal = np.maximum(0.0, rad_septal_raw) if clip_negative else rad_septal_raw
        rad_coll = np.maximum(0.0, rad_coll_raw) if clip_negative else rad_coll_raw
        rad_phant = np.maximum(0.0, rad_phant_raw) if clip_negative else rad_phant_raw
        rad_back = np.maximum(0.0, rad_back_raw) if clip_negative else rad_back_raw

        breakdown = self.analyze_window(
            energies=energies_arr,
            photon_components=components_arr,
            window_config=active_windows,
            psf_offsets=offsets_arr,
        )

        return ComponentPSFResult(
            edges_x=edges_x,
            edges_y=edges_y,
            centers_x=centers_x,
            centers_y=centers_y,
            psf_total=psf_total,
            psf_geometric=psf_geom,
            psf_septal_penetration=psf_septal,
            psf_collimator_scatter=psf_coll,
            psf_phantom_scatter=psf_phant,
            psf_backscatter=psf_back,
            radial_edges=radial_edges,
            radial_centers=radial_centers,
            radial_total=rad_total,
            radial_geometric=rad_geom,
            radial_septal_penetration=rad_septal,
            radial_collimator_scatter=rad_coll,
            radial_phantom_scatter=rad_phant,
            radial_backscatter=rad_back,
            pixel_size=float(edges_x[1] - edges_x[0]),
            half_extent=self.psf_half_extent,
            breakdown=breakdown,
        )

    @staticmethod
    def build_component_projections(
        events_list: Sequence[NDArray[np.floating[Any]]],
        components_list: Sequence[NDArray[np.integer[Any]]],
        detector_size: Sequence[float],
        pixel_size: float,
        energy_windows: Sequence[EnergyWindowConfig],
        clip_negative: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """Формирует 3D-стеки проекций `(views, nx, ny)` отдельно для каждой из 5 физических компонент."""
        if len(events_list) != len(components_list):
            raise ValueError(
                f"Длины списков events_list ({len(events_list)}) и components_list ({len(components_list)}) "
                "должны совпадать."
            )
        if not energy_windows:
            raise ValueError("Список энергетических окон energy_windows не может быть пустым.")
        if pixel_size <= 0.0:
            raise ValueError(f"Размер пикселя должен быть строго положительным: {pixel_size}")
        if len(detector_size) < 2 or detector_size[0] <= 0.0 or detector_size[1] <= 0.0:
            raise ValueError(f"Некорректные габариты детектора detector_size: {detector_size}")

        num_views = len(events_list)
        det_width = float(detector_size[0])
        det_height = float(detector_size[1])
        num_bins_x = max(1, round(det_width / float(pixel_size)))
        num_bins_y = max(1, round(det_height / float(pixel_size)))
        half_width = 0.5 * det_width
        half_height = 0.5 * det_height
        edges_x = np.linspace(-half_width, half_width, num_bins_x + 1, dtype=np.float64)
        edges_y = np.linspace(-half_height, half_height, num_bins_y + 1, dtype=np.float64)

        active_windows = tuple(energy_windows)
        component_keys: tuple[tuple[str, PhotonComponent], ...] = (
            ("geometric", PhotonComponent.GEOMETRIC),
            ("septal_penetration", PhotonComponent.SEPTAL_PENETRATION),
            ("collimator_scatter", PhotonComponent.COLLIMATOR_SCATTER),
            ("phantom_scatter", PhotonComponent.PHANTOM_SCATTER),
            ("backscatter", PhotonComponent.BACKSCATTER),
        )
        stacks: dict[str, NDArray[np.float64]] = {
            comp_name: np.zeros((num_views, num_bins_x, num_bins_y), dtype=np.float64)
            for comp_name, _ in component_keys
        }

        for view_idx, (events_matrix, comp_vector) in enumerate(zip(events_list, components_list, strict=True)):
            events_arr = np.asarray(events_matrix, dtype=np.float64)
            comp_arr = np.asarray(comp_vector, dtype=np.int8).ravel()
            if events_arr.size == 0:
                continue
            if events_arr.ndim != 2 or events_arr.shape[1] < 5:
                raise ValueError(
                    f"Элемент events_list[{view_idx}] должен иметь форму (num_events, 5), получено: {events_arr.shape}"
                )
            if events_arr.shape[0] != comp_arr.size:
                raise ValueError(
                    f"Несовпадение числа вспышек в ракурсе {view_idx}: "
                    f"events ({events_arr.shape[0]}) vs components ({comp_arr.size})."
                )

            energies = events_arr[:, 2]
            coords_first = events_arr[:, 3]
            coords_second = events_arr[:, 4]
            _, event_weights = _compute_energy_window_weights(energies, active_windows)
            nonzero_mask = event_weights != 0.0

            for comp_name, comp_enum in component_keys:
                selected = nonzero_mask & (comp_arr == int(comp_enum))
                if not bool(np.any(selected)):
                    continue
                hist_2d, _, _ = np.histogram2d(
                    coords_first[selected],
                    coords_second[selected],
                    bins=(edges_x, edges_y),
                    weights=event_weights[selected],
                )
                proj_slice = hist_2d.astype(np.float64)
                if clip_negative:
                    np.clip(proj_slice, 0.0, None, out=proj_slice)
                stacks[comp_name][view_idx] = proj_slice

        return stacks


def fuse_calibrated_windows(
    window_projections: Sequence[NDArray[np.floating[Any]]],
    scatter_estimates: Sequence[NDArray[np.floating[Any]]],
    calibration_factors: Sequence[float] | None = None,
    window_weights: Sequence[float] | None = None,
    clip_each_window: bool = True,
) -> NDArray[np.float64]:
    """Выполняет взвешенное слияние нескольких энергетических окон с индивидуально калиброванным вычитанием рассеяния.

    Реализует протокол `sum_w weight_w * [M_w - k_w * S_w]_+` (или клиппинг после суммирования).
    """
    num_windows = len(window_projections)
    if num_windows == 0:
        raise ValueError("Список window_projections не может быть пустым.")
    if len(scatter_estimates) != num_windows:
        raise ValueError("Длины window_projections и scatter_estimates должны совпадать.")

    resolved_factors = (
        tuple(float(val) for val in calibration_factors)
        if calibration_factors is not None
        else tuple(1.0 for _ in range(num_windows))
    )
    resolved_weights = (
        tuple(float(val) for val in window_weights)
        if window_weights is not None
        else tuple(1.0 for _ in range(num_windows))
    )
    if len(resolved_factors) != num_windows or len(resolved_weights) != num_windows:
        raise ValueError("Длины calibration_factors и window_weights должны совпадать с числом окон.")

    first_shape = np.asarray(window_projections[0]).shape
    fused_map = np.zeros(first_shape, dtype=np.float64)
    for proj_raw, scat_raw, cal_factor, win_weight in zip(
        window_projections,
        scatter_estimates,
        resolved_factors,
        resolved_weights,
        strict=True,
    ):
        proj_arr = np.asarray(proj_raw, dtype=np.float64)
        scat_arr = np.asarray(scat_raw, dtype=np.float64)
        if proj_arr.shape != first_shape or scat_arr.shape != first_shape:
            raise ValueError("Все матрицы проекций и оценок рассеяния должны иметь одинаковую форму.")
        corrected = proj_arr - cal_factor * scat_arr
        if clip_each_window:
            np.clip(corrected, 0.0, None, out=corrected)
        fused_map += win_weight * corrected

    np.clip(fused_map, 0.0, None, out=fused_map)
    return fused_map


def project_spheres_to_detector_plane(
    spheres: Sequence[SphericalInsertSpec],
    detector_pose_matrix: NDArray[np.floating[Any]],
) -> list[tuple[SphericalInsertSpec, float, float]]:
    """Ортогонально проецирует 3D-центры сферических очагов фантома на плоскость детектора `(transverse, axial)`."""
    det_center, _, axis_transverse, axis_axial = extract_orthonormal_frame_from_pose(detector_pose_matrix)
    projected: list[tuple[SphericalInsertSpec, float, float]] = []
    for sphere_spec in spheres:
        world_pt = np.asarray(sphere_spec.center_world, dtype=np.float64) * float(mm)
        rel_vec = world_pt - det_center
        coord_trans = float(np.dot(rel_vec, axis_transverse))
        coord_axial = float(np.dot(rel_vec, axis_axial))
        projected.append((sphere_spec, coord_trans, coord_axial))
    return projected


def evaluate_planar_phantom_quality(
    projection: NDArray[np.floating[Any]],
    reference_primary: NDArray[np.floating[Any]],
    edges_transverse: NDArray[np.floating[Any]],
    edges_axial: NDArray[np.floating[Any]],
    detector_pose_matrix: NDArray[np.floating[Any]],
    spheres: Sequence[SphericalInsertSpec],
    background_mask: NDArray[np.bool_],
    body_mask: NDArray[np.bool_] | None = None,
) -> PlanarPhantomQualityResult:
    """Рассчитывает планарный контраст, CRC, CNR сферических очагов, шум фона и ошибку относительно первичного эталона."""
    proj_arr = np.asarray(projection, dtype=np.float64)
    ref_arr = np.asarray(reference_primary, dtype=np.float64)
    bg_mask = np.asarray(background_mask, dtype=np.bool_)
    if proj_arr.shape != ref_arr.shape or proj_arr.shape != bg_mask.shape:
        raise ValueError("Формы projection, reference_primary и background_mask должны совпадать.")
    if not bool(np.any(bg_mask)):
        raise ValueError("Маска фона background_mask должна содержать хотя бы один активный пиксель.")

    edges_u = np.asarray(edges_transverse, dtype=np.float64)
    edges_v = np.asarray(edges_axial, dtype=np.float64)
    centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    centers_v = 0.5 * (edges_v[:-1] + edges_v[1:])
    if proj_arr.shape != (centers_u.size, centers_v.size):
        raise ValueError(
            f"Форма проекции {proj_arr.shape} не соответствует сетке бинов ({centers_u.size}, {centers_v.size})."
        )

    bg_values = proj_arr[bg_mask]
    bg_mean = float(np.mean(bg_values))
    bg_std = float(np.std(bg_values, ddof=1)) if bg_values.size > 1 else 0.0
    bg_cv = bg_std / bg_mean if bg_mean > 0.0 else 0.0

    eval_mask = np.asarray(body_mask, dtype=np.bool_) if body_mask is not None else np.ones_like(bg_mask)
    ref_body = ref_arr[eval_mask]
    proj_body = proj_arr[eval_mask]
    ref_rms = float(np.sqrt(np.mean(ref_body * ref_body))) if ref_body.size > 0 else 0.0
    diff_rms = float(np.sqrt(np.mean((proj_body - ref_body) ** 2))) if ref_body.size > 0 else 0.0
    nrmse_val = diff_rms / ref_rms if ref_rms > 0.0 else 0.0

    ref_sum = float(np.sum(ref_body))
    rel_bias = (float(np.sum(proj_body)) - ref_sum) / ref_sum if ref_sum > 0.0 else 0.0

    grid_u, grid_v = np.meshgrid(centers_u, centers_v, indexing="ij")
    projected_spheres = project_spheres_to_detector_plane(spheres, detector_pose_matrix)

    sphere_results: list[SphereContrastMetrics] = []
    for sphere_spec, center_u, center_v in projected_spheres:
        radius = 0.5 * float(sphere_spec.diameter)
        roi_mask = ((grid_u - center_u) ** 2 + (grid_v - center_v) ** 2) <= (radius * radius)
        if not bool(np.any(roi_mask)):
            nearest_u = int(np.argmin(np.abs(centers_u - center_u)))
            nearest_v = int(np.argmin(np.abs(centers_v - center_v)))
            roi_mask[nearest_u, nearest_v] = True

        mean_sig = float(np.mean(proj_arr[roi_mask]))
        local_contrast = (mean_sig - bg_mean) / bg_mean if bg_mean > 0.0 else 0.0
        denom_ratio = float(sphere_spec.activity_concentration_ratio) - 1.0
        crc_val = local_contrast / denom_ratio if abs(denom_ratio) > 1e-9 else 0.0
        cnr_val = (mean_sig - bg_mean) / bg_std if bg_std > 0.0 else 0.0

        sphere_results.append(
            SphereContrastMetrics(
                name=sphere_spec.name,
                diameter=float(sphere_spec.diameter),
                center_transverse=center_u,
                center_axial=center_v,
                mean_sphere_signal=mean_sig,
                local_contrast=local_contrast,
                contrast_recovery=crc_val,
                cnr=cnr_val,
            )
        )

    return PlanarPhantomQualityResult(
        background_mean=bg_mean,
        background_std=bg_std,
        background_cv=bg_cv,
        nrmse_vs_primary=nrmse_val,
        relative_bias_vs_primary=rel_bias,
        sphere_metrics=tuple(sphere_results),
    )
