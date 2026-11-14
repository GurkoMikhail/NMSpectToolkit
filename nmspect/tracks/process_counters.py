"""Векторный аннотатор счётчиков физических процессов вдоль треков частиц по зонам геометрии."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nmspect.tracks.models import (
    COHERENT_PROCESS_CODE,
    COMPTON_PROCESS_CODE,
    CRYSTAL_NODE_FLAG,
    EMISSION_PROCESS_CODE,
    PHOTOELECTRIC_PROCESS_CODE,
    TrackAnnotator,
    TrackBatch,
)

BACKSCATTER_GLASS_MATERIAL_ID: Final[int] = 25

REQUIRED_POINT_DATA_KEYS: Final[tuple[str, ...]] = (
    "process_code",
    "volume_id",
    "material_id",
    "atomic_number",
    "is_crystal_node",
)

PROCESS_COUNTER_KEYS: Final[tuple[str, ...]] = (
    "phantom_compton",
    "phantom_coherent",
    "collimator_compton",
    "collimator_coherent",
    "backscatter_count",
    "crystal_compton",
    "crystal_coherent",
    "crystal_photoelectric",
)


class ProcessCounterAnnotator(TrackAnnotator):
    """Вычисляет поуровневые счётчики физических взаимодействий вдоль треков в `tracks.cell_data`.

    Агрегация выполняется полностью векторно через `np.add.reduceat` по CSR-сдвигам `tracks.offsets[:-1]`
    без использования циклов Python по отдельным трекам.
    """

    def __init__(
        self,
        phantom_volume_ids: tuple[int, ...] | None = None,
        collimator_volume_ids: tuple[int, ...] = (4, 5, 10, 11),
        backscatter_volume_ids: tuple[int, ...] = (2, 3, 7, 8, 9, 13),
    ) -> None:
        """Инициализирует аннотатор счётчиков процессов с идентификаторами зон геометрии.

        Args:
            phantom_volume_ids: Кортеж идентификаторов предколлиматорных объёмов фантома (`volume_id`),
                либо `None` для включения всех внешних предколлиматорных взаимодействий вне коллиматора и световода.
            collimator_volume_ids: Кортеж идентификаторов перегородок и каналов коллиматора (`volume_id`).
            backscatter_volume_ids: Кортеж идентификаторов задетекторных объёмов, коробов и световода (`volume_id`).
        """
        self.phantom_volume_ids: tuple[int, ...] | None = (
            tuple(int(vol_id) for vol_id in phantom_volume_ids) if phantom_volume_ids is not None else None
        )
        self.collimator_volume_ids: tuple[int, ...] = tuple(int(vol_id) for vol_id in collimator_volume_ids)
        self.backscatter_volume_ids: tuple[int, ...] = tuple(int(vol_id) for vol_id in backscatter_volume_ids)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> ProcessCounterAnnotator:
        """Создаёт аннотатор по таблице объёмов `metadata['geometry']['volumes']` (при наличии) или дефолтам."""
        if not isinstance(metadata, dict):
            return cls()
        geom_meta = metadata.get("geometry")
        if not isinstance(geom_meta, dict) or "volumes" not in geom_meta:
            return cls()

        volumes_spec = geom_meta["volumes"]
        if not isinstance(volumes_spec, dict):
            return cls()

        collimator_ids: list[int] = []
        backscatter_ids: list[int] = []
        for vol_name, vol_info in volumes_spec.items():
            if not isinstance(vol_info, dict) or "volume_id" not in vol_info:
                continue
            vol_id = int(vol_info["volume_id"])
            lower_name = str(vol_name).lower()
            if "collimator" in lower_name:
                collimator_ids.append(vol_id)
            elif any(tag in lower_name for tag in ("casing", "glass", "backend", "detector_box", "pmt", "shield")):
                backscatter_ids.append(vol_id)

        if not collimator_ids and not backscatter_ids:
            return cls()

        return cls(
            phantom_volume_ids=None,
            collimator_volume_ids=tuple(collimator_ids) if collimator_ids else (4, 5, 10, 11),
            backscatter_volume_ids=tuple(backscatter_ids) if backscatter_ids else (2, 3, 7, 8, 9, 13),
        )

    @staticmethod
    def _reduce_node_mask(
        node_mask: NDArray[np.bool_],
        track_starts: NDArray[np.int64],
    ) -> NDArray[np.uint8]:
        """Суммирует булеву маску узлов по сегментам треков и приводит результат к `uint8`."""
        segment_sums = np.add.reduceat(node_mask.astype(np.int32), track_starts)
        return np.clip(segment_sums, 0, 255).astype(np.uint8)

    def annotate(self, tracks: TrackBatch, metadata: dict[str, Any] | None = None) -> None:
        """Вычисляет и записывает в `tracks.cell_data` 8 счётчиков взаимодействий по зонам геометрии.

        Args:
            tracks: Пакет треков `TrackBatch` в CSR-представлении.
            metadata: Опциональный словарь метаданных симуляции (сохраняется для совместимости с протоколом).

        Raises:
            KeyError: Если в `tracks.point_data` отсутствует хотя бы один обязательный массив атрибутов узлов.
        """
        del metadata

        for required_key in REQUIRED_POINT_DATA_KEYS:
            if required_key not in tracks.point_data:
                raise KeyError(
                    f"В tracks.point_data отсутствует обязательный ключ '{required_key}' "
                    f"для расчёта счётчиков физических процессов."
                )

        if tracks.num_tracks == 0:
            for counter_key in PROCESS_COUNTER_KEYS:
                tracks.cell_data[counter_key] = np.empty(0, dtype=np.uint8)
            return

        process_code = tracks.point_data["process_code"]
        volume_id = tracks.point_data["volume_id"]
        material_id = tracks.point_data["material_id"]
        atomic_number = tracks.point_data["atomic_number"]
        is_crystal_node = tracks.point_data["is_crystal_node"]

        in_crystal: NDArray[np.bool_] = is_crystal_node == CRYSTAL_NODE_FLAG
        track_starts: NDArray[np.int64] = tracks.offsets[:-1]
        node_indices: NDArray[np.int64] = np.arange(tracks.num_points, dtype=np.int64)
        track_indices: NDArray[np.int64] = np.repeat(
            np.arange(tracks.num_tracks, dtype=np.int64),
            tracks.track_lengths,
        )
        last_crystal_indices: NDArray[np.int64] = np.maximum.reduceat(
            np.where(in_crystal, node_indices, -1),
            track_starts,
        )
        before_last_crystal: NDArray[np.bool_] = node_indices < last_crystal_indices[track_indices]
        outside_crystal: NDArray[np.bool_] = (
            (~in_crystal) & (process_code != EMISSION_PROCESS_CODE) & before_last_crystal
        )

        explicit_backscatter: NDArray[np.bool_] = outside_crystal & (
            np.isin(volume_id, self.backscatter_volume_ids) | (material_id == BACKSCATTER_GLASS_MATERIAL_ID)
        )
        in_collimator: NDArray[np.bool_] = (
            outside_crystal
            & (~explicit_backscatter)
            & (np.isin(volume_id, self.collimator_volume_ids) | (atomic_number == 82))
        )
        in_backscatter: NDArray[np.bool_] = explicit_backscatter & (~np.isin(volume_id, self.collimator_volume_ids))
        in_phantom_base: NDArray[np.bool_] = outside_crystal & (~in_collimator) & (~in_backscatter)
        in_phantom: NDArray[np.bool_] = (
            in_phantom_base & np.isin(volume_id, self.phantom_volume_ids)
            if self.phantom_volume_ids is not None
            else in_phantom_base
        )

        is_compton: NDArray[np.bool_] = process_code == COMPTON_PROCESS_CODE
        is_coherent: NDArray[np.bool_] = process_code == COHERENT_PROCESS_CODE
        is_photoelectric: NDArray[np.bool_] = process_code == PHOTOELECTRIC_PROCESS_CODE
        is_physical_interaction: NDArray[np.bool_] = process_code >= PHOTOELECTRIC_PROCESS_CODE

        tracks.cell_data["phantom_compton"] = self._reduce_node_mask(
            in_phantom & is_compton,
            track_starts,
        )
        tracks.cell_data["phantom_coherent"] = self._reduce_node_mask(
            in_phantom & is_coherent,
            track_starts,
        )
        tracks.cell_data["collimator_compton"] = self._reduce_node_mask(
            in_collimator & is_compton,
            track_starts,
        )
        tracks.cell_data["collimator_coherent"] = self._reduce_node_mask(
            in_collimator & is_coherent,
            track_starts,
        )
        tracks.cell_data["backscatter_count"] = self._reduce_node_mask(
            in_backscatter & is_physical_interaction,
            track_starts,
        )
        tracks.cell_data["crystal_compton"] = self._reduce_node_mask(
            in_crystal & is_compton,
            track_starts,
        )
        tracks.cell_data["crystal_coherent"] = self._reduce_node_mask(
            in_crystal & is_coherent,
            track_starts,
        )
        tracks.cell_data["crystal_photoelectric"] = self._reduce_node_mask(
            in_crystal & is_photoelectric,
            track_starts,
        )
