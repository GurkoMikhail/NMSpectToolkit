"""Численная аналитика треков: метрики угловой ФРТ, гексагональной анизотропии, селективности DOI и междетекторного перелёта."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from hepunits import mm, rad
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from nmspect.tracks.models import TrackBatch
from nmspect.tracks.scintillation_annotator import FWHM_TO_SIGMA


@dataclass(frozen=True)
class AngularPsfMetrics:
    """Количественные характеристики угловой функции рассеяния точки (ФРТ) в координатах наклона `tan(theta)`.

    Attributes:
        count: Эффективное число треков (или сумма весов).
        mean_tan_transverse: Средний поперечный наклон `tan(theta_u)`.
        rms_tan_transverse: Среднеквадратичное значение поперечного наклона `sqrt(mean(tan_u^2))`.
        std_tan_transverse: Стандартное отклонение поперечного наклона.
        rms_tan_radial: Радиальное СКО наклона `sqrt(mean(tan_u^2 + tan_v^2))`.
        fwhm_tan: Полная ширина на полувысоте (FWHM) поперечного профиля `tan(theta_u)`.
        fwtm_tan: Полная ширина на десятой высоты (FWTM) поперечного профиля `tan(theta_u)`.
        quantile_50_radial: Медианный радиальный наклон `r_50`.
        quantile_90_radial: Девяностый процентиль радиального наклона `r_90`.
        quantile_99_radial: Девяносто девятый процентиль радиального наклона `r_99`.
        wing_fraction_moderate: Доля событий в умеренных крыльях `|tan(theta_u)| > moderate_wing_threshold`.
        wing_fraction_wide: Доля событий в далёких крыльях `|tan(theta_u)| > wide_wing_threshold`.
    """

    count: float
    mean_tan_transverse: float
    rms_tan_transverse: float
    std_tan_transverse: float
    rms_tan_radial: float
    fwhm_tan: float
    fwtm_tan: float
    quantile_50_radial: float
    quantile_90_radial: float
    quantile_99_radial: float
    wing_fraction_moderate: float
    wing_fraction_wide: float


@dataclass(frozen=True)
class AzimuthalAnisotropyResult:
    """Характеристики азимутальной анизотропии 2D ФРТ (лучевой артефакт септ коллиматора).

    Все угловые координаты азимута возвращаются в базовых единицах `hepunits` (`rad` в диапазоне `[0, 2*pi)`).
    """

    azimuth_edges: NDArray[np.float64]
    azimuth_centers: NDArray[np.float64]
    counts: NDArray[np.float64]
    normalized_profile: NDArray[np.float64]
    peak_to_valley_ratio: float
    modulation_contrast: float
    dominant_harmonic_order: int
    dominant_harmonic_amplitude: float


@dataclass(frozen=True)
class DoiDiscriminationResult:
    """Результаты ROC-анализа аппаратной селекции по глубине взаимодействия в кристалле (DOI `z <= z_max`).

    Пороги `thresholds` и `optimal_threshold` выражены в базовых единицах `hepunits` (`mm`).
    """

    thresholds: NDArray[np.float64]
    signal_efficiency: NDArray[np.float64]
    background_rejection: NDArray[np.float64]
    roc_fpr: NDArray[np.float64]
    roc_tpr: NDArray[np.float64]
    auc: float
    optimal_threshold: float
    depth_resolution_fwhm: float


@dataclass(frozen=True)
class InterDetectorCrosstalkResult:
    """Статистика междетекторного перелёта фотонов (регистрация одной частицы более чем в одной головке)."""

    total_tracks: int
    unique_particles: int
    shared_particles: int
    shared_track_count: int
    shared_track_fraction: float
    detector_ids: tuple[int, ...]
    pair_matrix: NDArray[np.int64]


def compute_profile_width_at_fraction(
    bin_centers: NDArray[np.floating[Any]],
    profile_values: NDArray[np.floating[Any]],
    level_fraction: float = 0.5,
) -> float:
    """Вычисляет полную ширину одномерного профиля на заданной доле максимума с линейной интерполяцией."""
    if not (0.0 < level_fraction < 1.0):
        raise ValueError(f"Доля высоты профиля должна лежать в интервале (0, 1), получено: {level_fraction}.")

    centers_arr = np.asarray(bin_centers, dtype=np.float64).ravel()
    values_arr = np.asarray(profile_values, dtype=np.float64).ravel()
    if centers_arr.size != values_arr.size:
        raise ValueError("Размеры bin_centers и profile_values должны совпадать.")
    if centers_arr.size < 2:
        return 0.0

    peak_idx = int(np.argmax(values_arr))
    peak_val = float(values_arr[peak_idx])
    if peak_val <= 0.0:
        return 0.0

    target_val = peak_val * float(level_fraction)

    left_coord = float(centers_arr[0])
    for idx_left in range(peak_idx, 0, -1):
        curr_val = float(values_arr[idx_left])
        prev_val = float(values_arr[idx_left - 1])
        if prev_val <= target_val <= curr_val:
            denom = curr_val - prev_val
            interp_frac = (target_val - prev_val) / denom if abs(denom) > 1e-15 else 0.5
            left_coord = float(centers_arr[idx_left - 1]) + interp_frac * float(
                centers_arr[idx_left] - centers_arr[idx_left - 1]
            )
            break

    right_coord = float(centers_arr[-1])
    for idx_right in range(peak_idx, centers_arr.size - 1):
        curr_val = float(values_arr[idx_right])
        next_val = float(values_arr[idx_right + 1])
        if next_val <= target_val <= curr_val:
            denom = curr_val - next_val
            interp_frac = (curr_val - target_val) / denom if abs(denom) > 1e-15 else 0.5
            right_coord = float(centers_arr[idx_right]) + interp_frac * float(
                centers_arr[idx_right + 1] - centers_arr[idx_right]
            )
            break

    return max(0.0, right_coord - left_coord)


def _weighted_quantiles(
    values: NDArray[np.float64],
    quantiles: tuple[float, ...],
    weights: NDArray[np.float64] | None = None,
) -> tuple[float, ...]:
    """Вычисляет квантили одномерного массива с опциональными неотрицательными весами."""
    if values.size == 0:
        return tuple(0.0 for _ in quantiles)
    if weights is None:
        return tuple(float(np.quantile(values, q_val)) for q_val in quantiles)

    positive_mask = weights > 0.0
    if not bool(np.any(positive_mask)):
        return tuple(0.0 for _ in quantiles)

    vals_pos = values[positive_mask]
    wts_pos = weights[positive_mask]
    order = np.argsort(vals_pos)
    sorted_vals = vals_pos[order]
    cum_wts = np.cumsum(wts_pos[order])
    total_wt = float(cum_wts[-1])
    if total_wt <= 0.0:
        return tuple(0.0 for _ in quantiles)

    norm_cdf = cum_wts / total_wt
    return tuple(float(np.interp(q_val, norm_cdf, sorted_vals)) for q_val in quantiles)


def compute_angular_psf_metrics(
    tan_transverse: NDArray[np.floating[Any]],
    tan_axial: NDArray[np.floating[Any]] | None = None,
    hist_edges: NDArray[np.floating[Any]] | None = None,
    weights: NDArray[np.floating[Any]] | None = None,
    moderate_wing_threshold: float = 0.25,
    wide_wing_threshold: float = 0.60,
) -> AngularPsfMetrics:
    """Рассчитывает полный набор параметров угловой ФРТ (FWHM, FWTM, СКО, радиальные квантили и доли крыльев)."""
    tan_u = np.asarray(tan_transverse, dtype=np.float64).ravel()
    tan_v = (
        np.asarray(tan_axial, dtype=np.float64).ravel()
        if tan_axial is not None
        else np.zeros_like(tan_u)
    )
    if tan_u.size != tan_v.size:
        raise ValueError("Длины массивов tan_transverse и tan_axial должны совпадать.")

    wts_arr: NDArray[np.float64] | None = None
    if weights is not None:
        wts_arr = np.asarray(weights, dtype=np.float64).ravel()
        if wts_arr.size != tan_u.size:
            raise ValueError("Длина массива weights должна совпадать с длиной tan_transverse.")

    resolved_edges = (
        np.asarray(hist_edges, dtype=np.float64)
        if hist_edges is not None
        else np.linspace(-0.25, 0.25, 201, dtype=np.float64)
    )
    if resolved_edges.ndim != 1 or resolved_edges.size < 3:
        raise ValueError("Сетка hist_edges должна быть одномерной и содержать не менее 3 границ.")

    if tan_u.size == 0:
        return AngularPsfMetrics(
            count=0.0,
            mean_tan_transverse=0.0,
            rms_tan_transverse=0.0,
            std_tan_transverse=0.0,
            rms_tan_radial=0.0,
            fwhm_tan=0.0,
            fwtm_tan=0.0,
            quantile_50_radial=0.0,
            quantile_90_radial=0.0,
            quantile_99_radial=0.0,
            wing_fraction_moderate=0.0,
            wing_fraction_wide=0.0,
        )

    eff_count = float(np.sum(wts_arr)) if wts_arr is not None else float(tan_u.size)
    if eff_count <= 0.0:
        return AngularPsfMetrics(
            count=0.0,
            mean_tan_transverse=0.0,
            rms_tan_transverse=0.0,
            std_tan_transverse=0.0,
            rms_tan_radial=0.0,
            fwhm_tan=0.0,
            fwtm_tan=0.0,
            quantile_50_radial=0.0,
            quantile_90_radial=0.0,
            quantile_99_radial=0.0,
            wing_fraction_moderate=0.0,
            wing_fraction_wide=0.0,
        )

    radial_sq = tan_u * tan_u + tan_v * tan_v
    radial_tan = np.sqrt(radial_sq)
    abs_tan_u = np.abs(tan_u)

    if wts_arr is None:
        mean_u = float(np.mean(tan_u))
        rms_u = float(np.sqrt(np.mean(tan_u * tan_u)))
        std_u = float(np.std(tan_u))
        rms_radial = float(np.sqrt(np.mean(radial_sq)))
        wing_mod = float(np.count_nonzero(abs_tan_u > moderate_wing_threshold)) / eff_count
        wing_wide = float(np.count_nonzero(abs_tan_u > wide_wing_threshold)) / eff_count
    else:
        pos_wts = np.maximum(0.0, wts_arr)
        pos_sum = float(np.sum(pos_wts))
        if pos_sum > 0.0:
            mean_u = float(np.sum(pos_wts * tan_u) / pos_sum)
            rms_u = float(np.sqrt(np.sum(pos_wts * tan_u * tan_u) / pos_sum))
            var_u = max(0.0, float(np.sum(pos_wts * (tan_u - mean_u) ** 2) / pos_sum))
            std_u = float(np.sqrt(var_u))
            rms_radial = float(np.sqrt(np.sum(pos_wts * radial_sq) / pos_sum))
            wing_mod = float(np.sum(pos_wts[abs_tan_u > moderate_wing_threshold]) / pos_sum)
            wing_wide = float(np.sum(pos_wts[abs_tan_u > wide_wing_threshold]) / pos_sum)
        else:
            mean_u, rms_u, std_u, rms_radial, wing_mod, wing_wide = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    profile_hist, _ = np.histogram(tan_u, bins=resolved_edges, weights=wts_arr)
    profile_clean = np.maximum(0.0, profile_hist.astype(np.float64))
    centers = 0.5 * (resolved_edges[:-1] + resolved_edges[1:])
    fwhm_val = compute_profile_width_at_fraction(centers, profile_clean, level_fraction=0.5)
    fwtm_val = compute_profile_width_at_fraction(centers, profile_clean, level_fraction=0.1)

    q50, q90, q99 = _weighted_quantiles(radial_tan, (0.50, 0.90, 0.99), weights=wts_arr)

    return AngularPsfMetrics(
        count=eff_count,
        mean_tan_transverse=mean_u,
        rms_tan_transverse=rms_u,
        std_tan_transverse=std_u,
        rms_tan_radial=rms_radial,
        fwhm_tan=fwhm_val,
        fwtm_tan=fwtm_val,
        quantile_50_radial=q50,
        quantile_90_radial=q90,
        quantile_99_radial=q99,
        wing_fraction_moderate=wing_mod,
        wing_fraction_wide=wing_wide,
    )


def compute_azimuthal_anisotropy(
    tan_transverse: NDArray[np.floating[Any]],
    tan_axial: NDArray[np.floating[Any]],
    radial_min: float = 0.055,
    radial_max: float = 0.24,
    num_azimuth_bins: int = 72,
    max_harmonic_search: int = 12,
) -> AzimuthalAnisotropyResult:
    """Анализирует угловую модуляцию 2D ФРТ в кольце `[radial_min, radial_max]` (6-лучевой артефакт септ)."""
    if radial_min < 0.0 or radial_max <= radial_min:
        raise ValueError(f"Некорректные границы кольца наклонов: [{radial_min}, {radial_max}].")
    if num_azimuth_bins < 6:
        raise ValueError(f"Число бинов по азимуту должно быть не менее 6: {num_azimuth_bins}.")

    tan_u = np.asarray(tan_transverse, dtype=np.float64).ravel()
    tan_v = np.asarray(tan_axial, dtype=np.float64).ravel()
    if tan_u.size != tan_v.size:
        raise ValueError("Длины tan_transverse и tan_axial должны совпадать.")

    full_circle = float(2.0 * np.pi * rad)
    az_edges = np.linspace(0.0, full_circle, num_azimuth_bins + 1, dtype=np.float64)
    az_centers = 0.5 * (az_edges[:-1] + az_edges[1:])

    rad_tan = np.hypot(tan_u, tan_v)
    ring_mask = (rad_tan >= radial_min) & (rad_tan <= radial_max)
    if not bool(np.any(ring_mask)):
        zeros_arr = np.zeros(num_azimuth_bins, dtype=np.float64)
        return AzimuthalAnisotropyResult(
            azimuth_edges=az_edges,
            azimuth_centers=az_centers,
            counts=zeros_arr,
            normalized_profile=zeros_arr.copy(),
            peak_to_valley_ratio=1.0,
            modulation_contrast=0.0,
            dominant_harmonic_order=0,
            dominant_harmonic_amplitude=0.0,
        )

    azimuth_vals = np.mod(np.arctan2(tan_v[ring_mask], tan_u[ring_mask]), full_circle)
    counts_hist, _ = np.histogram(azimuth_vals, bins=az_edges)
    counts_float = counts_hist.astype(np.float64)

    mean_count = float(np.mean(counts_float))
    norm_profile = counts_float / mean_count if mean_count > 0.0 else np.zeros_like(counts_float)

    max_val = float(np.max(counts_float))
    positive_vals = counts_float[counts_float > 0.0]
    min_pos_val = float(np.min(positive_vals)) if positive_vals.size > 0 else 1.0
    min_raw_val = float(np.min(counts_float))
    peak_to_valley = max_val / min_pos_val if max_val > 0.0 else 1.0
    modulation = (max_val - min_raw_val) / (max_val + min_raw_val) if (max_val + min_raw_val) > 0.0 else 0.0

    fft_coeffs = np.fft.rfft(norm_profile - 1.0) / float(num_azimuth_bins)
    amplitudes = 2.0 * np.abs(fft_coeffs)
    upper_order = min(max_harmonic_search, amplitudes.size - 1)
    if upper_order >= 1 and mean_count > 0.0:
        search_slice = amplitudes[1 : upper_order + 1]
        best_offset = int(np.argmax(search_slice))
        dom_order = best_offset + 1
        dom_amp = float(search_slice[best_offset])
    else:
        dom_order = 0
        dom_amp = 0.0

    return AzimuthalAnisotropyResult(
        azimuth_edges=az_edges,
        azimuth_centers=az_centers,
        counts=counts_float,
        normalized_profile=norm_profile,
        peak_to_valley_ratio=peak_to_valley,
        modulation_contrast=modulation,
        dominant_harmonic_order=dom_order,
        dominant_harmonic_amplitude=dom_amp,
    )


def compute_doi_roc_curve(
    signal_depth_hist: NDArray[np.floating[Any]],
    background_depth_hist: NDArray[np.floating[Any]],
    depth_edges: NDArray[np.floating[Any]],
    depth_resolution_fwhm: float = 0.0 * mm,
) -> DoiDiscriminationResult:
    """Вычисляет характеристики и ROC-кривую DOI-дискриминатора `z_hat <= z_max` с учётом разрешения по глубине.

    Гауссово размытие оценки глубины `z_hat = z + N(0, sigma_DOI^2)` выполняется на симметрично расширенной
    за пределы кристалла сетке с нулевым заполнением вне `[z_min, z_max]`, а площадь под ROC-кривой (AUC)
    интегрируется на полном носителе `(0, 0) -> (1, 1)`, что гарантирует точную математическую монотонность
    `AUC` при ухудшении аппаратного разрешения `depth_resolution_fwhm`.
    """
    if depth_resolution_fwhm < 0.0:
        raise ValueError(f"Разрешение глубины взаимодействия не может быть отрицательным: {depth_resolution_fwhm}.")

    sig_hist = np.asarray(signal_depth_hist, dtype=np.float64).ravel()
    bkg_hist = np.asarray(background_depth_hist, dtype=np.float64).ravel()
    edges_arr = np.asarray(depth_edges, dtype=np.float64).ravel()

    if sig_hist.size == 0 or sig_hist.size != bkg_hist.size:
        raise ValueError("Гистограммы signal_depth_hist и background_depth_hist должны быть непустыми и равной длины.")
    if edges_arr.size != sig_hist.size + 1:
        raise ValueError(
            f"Длина depth_edges ({edges_arr.size}) должна быть на 1 больше числа бинов ({sig_hist.size})."
        )

    bin_steps = np.diff(edges_arr)
    if bool(np.any(bin_steps <= 0.0)):
        raise ValueError("Границы depth_edges должны строго возрастать.")
    step_size = float(np.mean(bin_steps))

    thresholds = edges_arr[1:].copy()
    sig_clean = np.maximum(0.0, sig_hist)
    bkg_clean = np.maximum(0.0, bkg_hist)
    sig_sum = float(np.sum(sig_clean))
    bkg_sum = float(np.sum(bkg_clean))

    if sig_sum <= 0.0 or bkg_sum <= 0.0:
        zeros_grid = np.zeros_like(thresholds)
        return DoiDiscriminationResult(
            thresholds=thresholds,
            signal_efficiency=zeros_grid.copy(),
            background_rejection=zeros_grid.copy(),
            roc_fpr=np.array([0.0, 1.0], dtype=np.float64),
            roc_tpr=np.array([0.0, 1.0], dtype=np.float64),
            auc=0.5,
            optimal_threshold=float(thresholds[0]),
            depth_resolution_fwhm=float(depth_resolution_fwhm),
        )

    sigma_depth = float(depth_resolution_fwhm) * FWHM_TO_SIGMA
    if sigma_depth > 0.0:
        sigma_bins = sigma_depth / step_size
        pad_bins = max(4, int(np.ceil(6.0 * sigma_bins)))
        sig_padded = np.pad(sig_clean, pad_width=pad_bins, mode="constant", constant_values=0.0)
        bkg_padded = np.pad(bkg_clean, pad_width=pad_bins, mode="constant", constant_values=0.0)
        sig_blurred = gaussian_filter1d(sig_padded, sigma=sigma_bins, mode="constant", cval=0.0)
        bkg_blurred = gaussian_filter1d(bkg_padded, sigma=sigma_bins, mode="constant", cval=0.0)
    else:
        pad_bins = 0
        sig_blurred = sig_clean
        bkg_blurred = bkg_clean

    sig_cdf_ext = np.cumsum(sig_blurred) / max(float(np.sum(sig_blurred)), 1e-15)
    bkg_cdf_ext = np.cumsum(bkg_blurred) / max(float(np.sum(bkg_blurred)), 1e-15)
    np.clip(sig_cdf_ext, 0.0, 1.0, out=sig_cdf_ext)
    np.clip(bkg_cdf_ext, 0.0, 1.0, out=bkg_cdf_ext)

    # Полная ROC-кривая на всём носителе зашумлённой оценки глубины с явными концами (0, 0) и (1, 1)
    roc_fpr = np.concatenate(([0.0], bkg_cdf_ext, [1.0]))
    roc_tpr = np.concatenate(([0.0], sig_cdf_ext, [1.0]))
    auc_val = float(np.trapezoid(roc_tpr, roc_fpr))
    auc_val = float(np.clip(auc_val, 0.0, 1.0))

    # Значения удержания сигнала и подавления фона при порогах z_max внутри кристалла [z_min, z_max]
    crystal_slice = slice(pad_bins, pad_bins + sig_hist.size)
    sig_eff_crystal = sig_cdf_ext[crystal_slice].copy()
    bkg_rej_crystal = 1.0 - bkg_cdf_ext[crystal_slice]

    youden_index = sig_eff_crystal + bkg_rej_crystal - 1.0
    best_idx = int(np.argmax(youden_index))
    opt_threshold = float(thresholds[best_idx])

    return DoiDiscriminationResult(
        thresholds=thresholds,
        signal_efficiency=sig_eff_crystal,
        background_rejection=bkg_rej_crystal,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        auc=auc_val,
        optimal_threshold=opt_threshold,
        depth_resolution_fwhm=float(depth_resolution_fwhm),
    )


def compute_interdetector_crosstalk(tracks: TrackBatch) -> InterDetectorCrosstalkResult:
    """Подсчитывает долю и матрицу перекрёстной регистрации одних и тех же `particle_id` в разных детекторах."""
    for req_key in ("particle_id", "detector_id"):
        if req_key not in tracks.cell_data:
            raise KeyError(f"В tracks.cell_data отсутствует поле '{req_key}' для анализа междетекторного перелёта.")

    num_tracks = tracks.num_tracks
    if num_tracks == 0:
        return InterDetectorCrosstalkResult(
            total_tracks=0,
            unique_particles=0,
            shared_particles=0,
            shared_track_count=0,
            shared_track_fraction=0.0,
            detector_ids=(),
            pair_matrix=np.zeros((0, 0), dtype=np.int64),
        )

    part_ids = np.asarray(tracks.cell_data["particle_id"], dtype=np.int64)
    det_ids = np.asarray(tracks.cell_data["detector_id"], dtype=np.int32)
    proj_indices = (
        np.asarray(tracks.cell_data["projection_index"], dtype=np.int64)
        if "projection_index" in tracks.cell_data
        else np.zeros(num_tracks, dtype=np.int64)
    )
    gantry_angles = (
        np.round(np.asarray(tracks.cell_data["gantry_angle"], dtype=np.float64), decimals=6)
        if "gantry_angle" in tracks.cell_data
        else np.zeros(num_tracks, dtype=np.float64)
    )

    unique_dets_arr = np.unique(det_ids)
    det_tuple = tuple(int(val) for val in unique_dets_arr)
    num_dets = len(det_tuple)
    pair_matrix = np.zeros((num_dets, num_dets), dtype=np.int64)

    # Группируем по (gantry_angle, particle_id), чтобы не смешивать независимые ракурсы с одинаковыми локальными ID
    composite_dtype = np.dtype([("angle", np.float64), ("part", np.int64)])
    composite_keys = np.empty(num_tracks, dtype=composite_dtype)
    composite_keys["angle"] = gantry_angles if "gantry_angle" in tracks.cell_data else proj_indices.astype(np.float64)
    composite_keys["part"] = part_ids

    order = np.argsort(composite_keys)
    sorted_keys = composite_keys[order]
    sorted_dets = det_ids[order]

    is_new_group = np.empty(num_tracks, dtype=np.bool_)
    is_new_group[0] = True
    if num_tracks > 1:
        is_new_group[1:] = sorted_keys[1:] != sorted_keys[:-1]

    group_starts = np.flatnonzero(is_new_group)
    group_counts = np.diff(np.append(group_starts, num_tracks))
    unique_particles = int(group_starts.size)

    multi_group_indices = np.flatnonzero(group_counts > 1)
    if multi_group_indices.size == 0:
        return InterDetectorCrosstalkResult(
            total_tracks=num_tracks,
            unique_particles=unique_particles,
            shared_particles=0,
            shared_track_count=0,
            shared_track_fraction=0.0,
            detector_ids=det_tuple,
            pair_matrix=pair_matrix,
        )

    det_index_map = {det_val: idx_pos for idx_pos, det_val in enumerate(det_tuple)}
    shared_particles = 0
    shared_tracks = 0
    for grp_idx in multi_group_indices:
        start_pos = int(group_starts[grp_idx])
        end_pos = start_pos + int(group_counts[grp_idx])
        grp_dets = np.unique(sorted_dets[start_pos:end_pos])
        if grp_dets.size > 1:
            shared_particles += 1
            shared_tracks += end_pos - start_pos
            for idx_first in range(grp_dets.size):
                for idx_second in range(idx_first + 1, grp_dets.size):
                    pos_first = det_index_map[int(grp_dets[idx_first])]
                    pos_second = det_index_map[int(grp_dets[idx_second])]
                    pair_matrix[pos_first, pos_second] += 1
                    pair_matrix[pos_second, pos_first] += 1

    return InterDetectorCrosstalkResult(
        total_tracks=num_tracks,
        unique_particles=unique_particles,
        shared_particles=shared_particles,
        shared_track_count=shared_tracks,
        shared_track_fraction=float(shared_tracks) / float(num_tracks),
        detector_ids=det_tuple,
        pair_matrix=pair_matrix,
    )
