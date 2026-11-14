"""Векторный аннотатор сцинтилляционного отклика треков и конвертер в события детектора для проекций."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from hepunits import keV, mm, ns
from numpy.typing import NDArray

from nmspect.detector.scintillation import (
    FWHM_TO_SIGMA_FACTOR,
    aggregate_energy_weighted_segments,
    compute_spatial_offsets_2d,
    normalize_energy_resolution,
    smear_scintillation_energy,
)
from nmspect.tracks.geometry import resolve_detector_frames, resolve_metadata_gantry_angle
from nmspect.tracks.models import CRYSTAL_NODE_FLAG, TrackAnnotator, TrackBatch

SPEED_OF_LIGHT: Final[float] = 299.792458 * (mm / ns)
FWHM_TO_SIGMA: Final[float] = FWHM_TO_SIGMA_FACTOR

REQUIRED_POINT_DATA_KEYS: Final[tuple[str, ...]] = (
    "energy_deposit",
    "distance_traveled",
    "is_crystal_node",
)

REQUIRED_CELL_DATA_KEYS: Final[tuple[str, ...]] = (
    "emission_time",
    "detector_id",
)


class ScintillationTrackAnnotator(TrackAnnotator):
    """Моделирует сцинтилляционную вспышку и аппаратное размытие (ФЭУ) для каждого трека в `tracks.cell_data`.

    Использует общее физическое ядро `nmspect.detector.scintillation` для расчёта агрегированных по узлам
    внутри целевого кристалла характеристик вспышки (`deposited_energy`, `has_scintillation`,
    `scintillation_position`, `scintillation_time`, `crystal_depth`) и зарегистрированных после энергетического
    и пространственного размытия величин (`measured_energy`, `registered_position`, `detector_local_position`).
    """

    def __init__(
        self,
        energy_resolution: float = 0.099,
        reference_energy: float = 140.5 * keV,
        intrinsic_resolution: float = 3.8 * mm,
        crystal_thickness: float = 9.5 * mm,
        orbit_radius: float = 250.0 * mm,
        random_seed: int | None = 42,
        rng: np.random.Generator | None = None,
        scale_spatial_with_energy: bool = False,
    ) -> None:
        """Инициализирует параметры энергетического и пространственного разрешения сцинтилляционного детектора.

        Args:
            energy_resolution: Относительное энергетическое разрешение FWHM на референсной энергии
                (в долях единицы или процентах: если `> 1.0`, автоматически делится на `100.0`).
            reference_energy: Референсная энергия калибровки разрешения в базовых единицах `hepunits` (`MeV`).
            intrinsic_resolution: Внутреннее пространственное разрешение кристалл+ФЭУ FWHM
                в базовых единицах `hepunits` (`mm`).
            crystal_thickness: Толщина сцинтилляционного кристалла вдоль нормали по умолчанию в `hepunits` (`mm`).
            orbit_radius: Радиус орбиты вращения кристалла по умолчанию в базовых единицах `hepunits` (`mm`).
            random_seed: Зерно генератора псевдослучайных чисел для воспроизводимого размытия.
            rng: Опциональный готовый экземпляр генератора `np.random.Generator` (приоритетнее `random_seed`).
            scale_spatial_with_energy: Масштабировать ли пространственное разрешение ФЭУ как `sqrt(E_ref / E_dep)`.
        """
        resolved_energy_resolution = normalize_energy_resolution(energy_resolution)
        if reference_energy <= 0.0:
            raise ValueError(f"Референсная энергия должна быть строго положительной: {reference_energy}.")
        if intrinsic_resolution < 0.0:
            raise ValueError(
                f"Внутреннее пространственное разрешение не может быть отрицательным: {intrinsic_resolution}."
            )
        if crystal_thickness <= 0.0:
            raise ValueError(f"Толщина кристалла должна быть строго положительной: {crystal_thickness}.")
        if orbit_radius <= 0.0:
            raise ValueError(f"Радиус орбиты должен быть строго положительным: {orbit_radius}.")

        self.energy_resolution: float = resolved_energy_resolution
        self.reference_energy: float = float(reference_energy)
        self.intrinsic_resolution: float = float(intrinsic_resolution)
        self.crystal_thickness: float = float(crystal_thickness)
        self.orbit_radius: float = float(orbit_radius)
        self.random_seed: int | None = random_seed
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(random_seed)
        self.scale_spatial_with_energy: bool = bool(scale_spatial_with_energy)

    def _resolve_crystal_thickness(self, metadata: dict[str, Any] | None) -> float:
        """Извлекает толщину кристалла в базовых единицах `mm` из `metadata['geometry']` или возвращает `self.crystal_thickness`."""
        if not isinstance(metadata, dict):
            return self.crystal_thickness

        geom_meta = metadata.get("geometry")
        if isinstance(geom_meta, dict) and "crystal_dimensions" in geom_meta:
            raw_dims = np.asarray(geom_meta["crystal_dimensions"], dtype=np.float64).ravel()
            if raw_dims.size >= 3 and float(raw_dims[2]) > 0.0:
                return float(raw_dims[2]) * float(mm)

        return self.crystal_thickness

    def annotate(self, tracks: TrackBatch, metadata: dict[str, Any] | None = None) -> None:
        """Вычисляет и записывает в `tracks.cell_data` параметры сцинтилляционной вспышки и её регистрации.

        Args:
            tracks: Пакет треков `TrackBatch` в CSR-представлении.
            metadata: Опциональный словарь метаданных симуляции, содержащий `"volume_matrices"`.

        Raises:
            KeyError: Если в `tracks.point_data` или `tracks.cell_data` отсутствуют обязательные входные поля.
        """
        for point_key in REQUIRED_POINT_DATA_KEYS:
            if point_key not in tracks.point_data:
                raise KeyError(
                    f"В tracks.point_data отсутствует обязательный ключ '{point_key}' "
                    f"для расчёта сцинтилляционного отклика."
                )

        for cell_key in REQUIRED_CELL_DATA_KEYS:
            if cell_key not in tracks.cell_data:
                raise KeyError(
                    f"В tracks.cell_data отсутствует обязательный ключ '{cell_key}' "
                    f"для расчёта сцинтилляционного отклика."
                )

        if tracks.num_tracks == 0:
            tracks.cell_data["deposited_energy"] = np.empty(0, dtype=np.float32)
            tracks.cell_data["measured_energy"] = np.empty(0, dtype=np.float32)
            tracks.cell_data["has_scintillation"] = np.empty(0, dtype=np.uint8)
            tracks.cell_data["scintillation_time"] = np.empty(0, dtype=np.float32)
            tracks.cell_data["scintillation_position"] = np.empty((0, 3), dtype=np.float32)
            tracks.cell_data["registered_position"] = np.empty((0, 3), dtype=np.float32)
            tracks.cell_data["crystal_depth"] = np.empty(0, dtype=np.float32)
            tracks.cell_data["detector_local_position"] = np.empty((0, 2), dtype=np.float32)
            return

        num_tracks = tracks.num_tracks
        track_starts = tracks.offsets[:-1]
        track_lengths = tracks.track_lengths
        track_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), track_lengths)

        energy_deposit = np.asarray(tracks.point_data["energy_deposit"], dtype=np.float64)
        distance_traveled = np.asarray(tracks.point_data["distance_traveled"], dtype=np.float64)
        is_crystal_node = tracks.point_data["is_crystal_node"]
        points_coords = np.asarray(tracks.points, dtype=np.float64)
        emission_time = np.asarray(tracks.cell_data["emission_time"], dtype=np.float64)

        crystal_mask: NDArray[np.bool_] = is_crystal_node == CRYSTAL_NODE_FLAG
        crystal_flag = crystal_mask.astype(np.float64)
        crystal_edep = np.where(crystal_mask, energy_deposit, 0.0)
        node_time = emission_time[track_indices] + distance_traveled / SPEED_OF_LIGHT

        # Резервные геометрические центры для треков без энерговыделения в кристалле (упругое рассеяние / пролёт)
        crystal_node_counts = np.add.reduceat(crystal_flag, track_starts)
        safe_crystal_counts = np.maximum(crystal_node_counts, 1.0)
        sum_crystal_positions = np.add.reduceat(points_coords * crystal_flag[:, np.newaxis], track_starts, axis=0)
        fallback_positions = np.where(
            crystal_node_counts[:, np.newaxis] > 0.0,
            sum_crystal_positions / safe_crystal_counts[:, np.newaxis],
            points_coords[track_starts],
        )
        sum_crystal_time = np.add.reduceat(node_time * crystal_flag, track_starts)
        fallback_time = np.where(
            crystal_node_counts > 0.0,
            sum_crystal_time / safe_crystal_counts,
            emission_time,
        )

        # 1-3. Единое векторное ядро суммирования энергии, центроида вспышки и времени сцинтилляции
        deposited_energy_f64, positive_edep_mask, scintillation_position_f64, scintillation_time_f64 = (
            aggregate_energy_weighted_segments(
                segment_starts=track_starts,
                energy_deposits=crystal_edep,
                coordinates=points_coords,
                times=node_time,
                fallback_coordinates=fallback_positions,
                fallback_times=fallback_time,
            )
        )
        deposited_energy = deposited_energy_f64.astype(np.float32)
        scintillation_position = scintillation_position_f64.astype(np.float32)
        scintillation_time = scintillation_time_f64.astype(np.float32)

        # 4. Единое гауссово энергетическое размытие ФЭУ
        measured_energy = smear_scintillation_energy(
            deposited_energies=deposited_energy_f64,
            energy_resolution=self.energy_resolution,
            reference_energy=self.reference_energy,
            rng=self.rng,
            active_mask=positive_edep_mask,
        ).astype(np.float32)

        # 5. Единое 2D пространственное размытие ФЭУ, перевод в локальный репер детектора и DOI (crystal_depth)
        if "gantry_angle" in tracks.cell_data:
            gantry_angles = np.asarray(tracks.cell_data["gantry_angle"], dtype=np.float64)
        else:
            fallback_angle = resolve_metadata_gantry_angle(metadata)
            gantry_angles = np.full(num_tracks, fallback_angle, dtype=np.float64)

        detector_ids = np.asarray(tracks.cell_data["detector_id"])
        detector_centers, axis_normal, axis_transverse, axis_axial = resolve_detector_frames(
            detector_ids=detector_ids,
            gantry_angles=gantry_angles,
            metadata=metadata,
            default_orbit_radius=self.orbit_radius,
        )
        resolved_thickness = self._resolve_crystal_thickness(metadata)

        spatial_offsets = compute_spatial_offsets_2d(
            num_events=num_tracks,
            intrinsic_spatial_resolution=self.intrinsic_resolution,
            rng=self.rng,
            deposited_energies=deposited_energy_f64,
            reference_energy=self.reference_energy,
            scale_with_energy=self.scale_spatial_with_energy,
            active_mask=positive_edep_mask,
        )
        registered_position_f64 = (
            scintillation_position_f64
            + spatial_offsets[:, 0:1] * axis_transverse
            + spatial_offsets[:, 1:2] * axis_axial
        )
        registered_position = registered_position_f64.astype(np.float32)

        rel_scint = scintillation_position_f64 - detector_centers
        raw_depth = np.sum(rel_scint * axis_normal, axis=1) + 0.5 * resolved_thickness
        crystal_depth = np.clip(raw_depth, 0.0, resolved_thickness).astype(np.float32)

        rel_registered = registered_position_f64 - detector_centers
        local_coord_trans = np.sum(rel_registered * axis_transverse, axis=1)
        local_coord_axial = np.sum(rel_registered * axis_axial, axis=1)
        detector_local_position = np.column_stack((local_coord_trans, local_coord_axial)).astype(np.float32)

        tracks.cell_data["deposited_energy"] = deposited_energy
        tracks.cell_data["measured_energy"] = measured_energy
        tracks.cell_data["has_scintillation"] = positive_edep_mask.astype(np.uint8)
        tracks.cell_data["scintillation_time"] = scintillation_time
        tracks.cell_data["scintillation_position"] = scintillation_position
        tracks.cell_data["registered_position"] = registered_position
        tracks.cell_data["crystal_depth"] = crystal_depth
        tracks.cell_data["detector_local_position"] = detector_local_position


def extract_detector_events_from_tracks(
    tracks: TrackBatch,
    *,
    detector_id: int | None = None,
    projection_index: int | None = None,
    only_scintillating: bool = True,
    use_measured_energy: bool = True,
    selection_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Извлекает из аннотированного `TrackBatch` матрицу вспышек `(M, 5)` для построения проекций ОФЭКТ.

    Формат выходной матрицы полностью совместим с `ProjectionBuilder.build_projections(...)`,
    `generate_projections_from_events(...)` и `SpectrumProcessor`:
    `[particle_id, scintillation_time, energy, local_u, local_v]` в базовых единицах `hepunits`.

    Args:
        tracks: Аннотированный пакет треков `TrackBatch` (после `ScintillationTrackAnnotator`).
        detector_id: Опциональный фильтр по идентификатору детекторной головки (`1, 2, ...`).
        projection_index: Опциональный фильтр по индексу проекционного ракурса.
        only_scintillating: Отбирать ли только треки с ненулевым энерговыделением (`has_scintillation == 1`).
        use_measured_energy: Использовать ли размытую `measured_energy` (`True`) или истинную `deposited_energy` (`False`).
        selection_mask: Дополнительная булева маска отбора треков формы `(num_tracks,)` (например, отдельного класса таксономии).

    Returns:
        Двумерный массив событий формы `(M, 5)` типа `float64`.
    """
    energy_key = "measured_energy" if use_measured_energy else "deposited_energy"
    required_keys: list[str] = ["scintillation_time", energy_key, "detector_local_position"]
    if only_scintillating:
        required_keys.append("has_scintillation")
    if detector_id is not None:
        required_keys.append("detector_id")
    if projection_index is not None:
        required_keys.append("projection_index")

    for req_key in required_keys:
        if req_key not in tracks.cell_data:
            raise KeyError(
                f"В tracks.cell_data отсутствует ключ '{req_key}', необходимый для извлечения событий детектора."
            )

    num_tracks = tracks.num_tracks
    if num_tracks == 0:
        return np.empty((0, 5), dtype=np.float64)

    active_mask = np.ones(num_tracks, dtype=np.bool_)
    if only_scintillating:
        active_mask &= np.asarray(tracks.cell_data["has_scintillation"]) != 0
    if detector_id is not None:
        active_mask &= np.asarray(tracks.cell_data["detector_id"], dtype=np.int64) == int(detector_id)
    if projection_index is not None:
        active_mask &= np.asarray(tracks.cell_data["projection_index"], dtype=np.int64) == int(projection_index)
    if selection_mask is not None:
        custom_mask = np.asarray(selection_mask, dtype=np.bool_).ravel()
        if custom_mask.size != num_tracks:
            raise ValueError(
                f"Длина selection_mask ({custom_mask.size}) не совпадает с числом треков ({num_tracks})."
            )
        active_mask &= custom_mask

    if not bool(np.any(active_mask)):
        return np.empty((0, 5), dtype=np.float64)

    if "particle_id" in tracks.cell_data:
        particle_ids = np.asarray(tracks.cell_data["particle_id"], dtype=np.float64)[active_mask]
    else:
        particle_ids = np.flatnonzero(active_mask).astype(np.float64)

    scint_times = np.asarray(tracks.cell_data["scintillation_time"], dtype=np.float64)[active_mask]
    energies = np.asarray(tracks.cell_data[energy_key], dtype=np.float64)[active_mask]
    local_coords = np.asarray(tracks.cell_data["detector_local_position"], dtype=np.float64)[active_mask]
    if local_coords.ndim != 2 or local_coords.shape[1] < 2:
        raise ValueError(
            f"Поле 'detector_local_position' должно иметь форму (num_tracks, 2), получено: {local_coords.shape}."
        )

    return np.column_stack((particle_ids, scint_times, energies, local_coords[:, 0], local_coords[:, 1]))
