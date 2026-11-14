"""Векторный сборщик полилиний треков частиц из сырых файлов Монте-Карло симуляций HDF5."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from hepunits import MeV, cm, deg, eV, keV, m, mm, ms, ns, rad, s, us
from numpy.typing import NDArray

from nmspect.io.hdf5_reader import SimulationDataExtractor, parse_gantry_angle_from_filename
from nmspect.tracks.models import (
    CRYSTAL_NODE_FLAG,
    EMISSION_PROCESS_CODE,
    EMISSION_VOLUME_ID,
    TrackBatch,
)


class HdfTrackAssembler:
    """Собирает пакеты треков `TrackBatch` из групп `initial_states` и `interactions` файла HDF5."""

    def __init__(
        self,
        max_tracks_per_detector: int | None = None,
        random_seed: int | None = 42,
        rng: np.random.Generator | None = None,
        require_energy_deposit: bool = False,
    ) -> None:
        """Инициализирует сборщик треков из файлов HDF5.

        Args:
            max_tracks_per_detector: Максимальное число собираемых треков на один детектор
                в рамках одного файла (если `None` — собираются все зарегистрированные треки).
            random_seed: Зерно генератора случайных чисел для воспроизводимой выборки треков.
            rng: Опциональный готовый экземпляр генератора `np.random.Generator` (приоритетнее `random_seed`).
            require_energy_deposit: Если `True`, отбрасывает треки с нулевым суммарным энерговыделением в кристалле.
        """
        if max_tracks_per_detector is not None and max_tracks_per_detector <= 0:
            raise ValueError(
                f"Параметр max_tracks_per_detector должен быть положительным целым числом или None, "
                f"получено: {max_tracks_per_detector}."
            )
        self.max_tracks_per_detector: int | None = max_tracks_per_detector
        self.random_seed: int | None = random_seed
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(random_seed)
        self.require_energy_deposit: bool = bool(require_energy_deposit)

    @staticmethod
    def _resolve_dataset_unit_scale(dataset_obj: h5py.Dataset, default_scale: float) -> float:
        """Определяет множитель перевода датасета HDF5 в базовые единицы `hepunits` по атрибуту `@units`."""
        raw_attr = dataset_obj.attrs.get("units", "")
        if isinstance(raw_attr, bytes):
            unit_str = raw_attr.decode("utf-8", errors="ignore").strip().lower()
        else:
            unit_str = str(raw_attr).strip().lower()
        if not unit_str:
            return float(default_scale)

        unit_map: dict[str, float] = {
            "mm": float(mm),
            "millimeter": float(mm),
            "millimeters": float(mm),
            "cm": float(cm),
            "centimeter": float(cm),
            "m": float(m),
            "meter": float(m),
            "mev": float(MeV),
            "kev": float(keV),
            "ev": float(eV),
            "ns": float(ns),
            "nanosecond": float(ns),
            "us": float(us),
            "ms": float(ms),
            "s": float(s),
            "sec": float(s),
            "second": float(s),
        }
        return unit_map.get(unit_str, float(default_scale))

    @staticmethod
    def _resolve_gantry_angle(
        file_path: Path,
        metadata: dict[str, Any],
        explicit_angle: float | None,
    ) -> float:
        """Определяет угол поворота гантри в базовых единицах hepunits (радианах)."""
        if explicit_angle is not None:
            return float(explicit_angle)

        parsed_angle = parse_gantry_angle_from_filename(file_path.name)
        if parsed_angle is not None:
            return float(parsed_angle)

        gantry_meta = metadata.get("gantry", {})
        if "gantry_angle" in gantry_meta:
            raw_angle = float(np.asarray(gantry_meta["gantry_angle"]).item())
            units_label = str(gantry_meta.get("gantry_angle_units", "")).strip().lower()
            if units_label in ("deg", "degree", "degrees"):
                return float(raw_angle * deg)
            if units_label in ("rad", "radian", "radians"):
                return float(raw_angle * rad)
            if abs(raw_angle) > (2.0 * np.pi + 1e-2):
                return float(raw_angle * deg)
            return float(raw_angle * rad)

        return float(0.0 * rad)

    @staticmethod
    def _parse_detector_id(detector_name: str, fallback_id: int) -> int:
        """Извлекает целочисленный номер детектора из имени группы (например, 'Detector_1' -> 1)."""
        match_obj = re.search(r"Detector_(\d+)$", detector_name, re.IGNORECASE)
        if match_obj is not None:
            return int(match_obj.group(1))
        return fallback_id

    @staticmethod
    def _read_interaction_group(group_obj: h5py.Group) -> dict[str, NDArray[Any]]:
        """Считывает массивы шагов взаимодействия из группы HDF5 в типизированный словарь NumPy."""
        pid_key = "particle_ID" if "particle_ID" in group_obj else "particle_id"
        if pid_key not in group_obj:
            raise KeyError(f"В группе '{group_obj.name}' отсутствует обязательный датасет 'particle_ID'.")

        required_keys = (
            "global_position",
            "global_direction",
            "energy_deposit",
            "distance_traveled",
            "process_code",
            "volume_id",
            "material_id",
        )
        for req_key in required_keys:
            if req_key not in group_obj:
                raise KeyError(f"В группе '{group_obj.name}' отсутствует обязательный датасет '{req_key}'.")

        atomic_key = "Z" if "Z" in group_obj else "atomic_number"
        if atomic_key not in group_obj:
            raise KeyError(f"В группе '{group_obj.name}' отсутствует датасет атомного номера ('Z').")

        species_key = "species_code" if "species_code" in group_obj else "species"
        if species_key not in group_obj:
            raise KeyError(f"В группе '{group_obj.name}' отсутствует датасет типа частицы ('species_code').")

        pos_scale = HdfTrackAssembler._resolve_dataset_unit_scale(group_obj["global_position"], float(mm))
        edep_scale = HdfTrackAssembler._resolve_dataset_unit_scale(group_obj["energy_deposit"], float(MeV))
        dist_scale = HdfTrackAssembler._resolve_dataset_unit_scale(group_obj["distance_traveled"], float(mm))

        return {
            "particle_id": np.asarray(group_obj[pid_key][()], dtype=np.int64).ravel(),
            "global_position": (np.asarray(group_obj["global_position"][()], dtype=np.float32) * pos_scale).astype(
                np.float32, copy=False
            ),
            "global_direction": np.asarray(group_obj["global_direction"][()], dtype=np.float32),
            "energy_deposit": (
                np.asarray(group_obj["energy_deposit"][()], dtype=np.float32).ravel() * edep_scale
            ).astype(np.float32, copy=False),
            "distance_traveled": (
                np.asarray(group_obj["distance_traveled"][()], dtype=np.float32).ravel() * dist_scale
            ).astype(np.float32, copy=False),
            "process_code": np.asarray(group_obj["process_code"][()], dtype=np.int32).ravel(),
            "volume_id": np.asarray(group_obj["volume_id"][()], dtype=np.int32).ravel(),
            "material_id": np.asarray(group_obj["material_id"][()], dtype=np.int32).ravel(),
            "atomic_number": np.asarray(group_obj[atomic_key][()], dtype=np.int32).ravel(),
            "species": np.asarray(group_obj[species_key][()], dtype=np.int32).ravel(),
        }

    @staticmethod
    def _create_empty_batch() -> TrackBatch:
        """Создаёт пустой пакет `TrackBatch` с каноническим набором ключей и типов данных."""
        return TrackBatch(
            points=np.empty((0, 3), dtype=np.float32),
            offsets=np.zeros(1, dtype=np.int64),
            point_data={
                "global_direction": np.empty((0, 3), dtype=np.float32),
                "energy_deposit": np.empty(0, dtype=np.float32),
                "distance_traveled": np.empty(0, dtype=np.float32),
                "process_code": np.empty(0, dtype=np.int32),
                "volume_id": np.empty(0, dtype=np.int32),
                "material_id": np.empty(0, dtype=np.int32),
                "atomic_number": np.empty(0, dtype=np.int32),
                "species": np.empty(0, dtype=np.int32),
                "is_crystal_node": np.empty(0, dtype=np.uint8),
            },
            cell_data={
                "particle_id": np.empty(0, dtype=np.int64),
                "detector_id": np.empty(0, dtype=np.uint8),
                "projection_index": np.empty(0, dtype=np.int32),
                "gantry_angle": np.empty(0, dtype=np.float32),
                "emission_energy": np.empty(0, dtype=np.float32),
                "emission_time": np.empty(0, dtype=np.float32),
            },
        )

    def _assemble_detector_tracks(
        self,
        det_table: dict[str, NDArray[Any]],
        sim_table: dict[str, NDArray[Any]] | None,
        sorted_init_pids: NDArray[np.int64],
        sorted_init_positions: NDArray[np.float32],
        sorted_init_directions: NDArray[np.float32],
        sorted_init_energies: NDArray[np.float32],
        sorted_init_times: NDArray[np.float32],
        detector_id: int,
        projection_index: int,
        gantry_angle: float,
        rng: np.random.Generator,
    ) -> TrackBatch:
        """Векторно собирает `TrackBatch` для одного детектора без Python-циклов по трекам."""
        det_pids = det_table["particle_id"]
        if det_pids.size == 0:
            return self._create_empty_batch()

        if self.require_energy_deposit:
            sort_idx = np.argsort(det_pids, kind="mergesort")
            pids_sorted = det_pids[sort_idx]
            edep_sorted = np.asarray(det_table["energy_deposit"], dtype=np.float64)[sort_idx]
            unique_pids_all, group_starts_all = np.unique(pids_sorted, return_index=True)
            edep_sums = np.add.reduceat(edep_sorted, group_starts_all)
            unique_pids = unique_pids_all[edep_sums > 0.0]
            was_subsampled = True
        else:
            unique_pids = np.unique(det_pids)
            was_subsampled = False

        if unique_pids.size == 0:
            return self._create_empty_batch()

        if self.max_tracks_per_detector is not None and unique_pids.size > self.max_tracks_per_detector:
            sampled_positions = np.sort(
                rng.choice(unique_pids.size, size=self.max_tracks_per_detector, replace=False)
            )
            selected_pids = unique_pids[sampled_positions]
            was_subsampled = True
        else:
            selected_pids = unique_pids

        num_tracks = int(selected_pids.size)
        if num_tracks == 0:
            return self._create_empty_batch()

        # 1. Отбор и стабильная сортировка шагов в кристалле детектора по particle_ID
        if was_subsampled:
            det_valid_indices = np.flatnonzero(np.isin(det_pids, selected_pids))
            det_filtered_pids = det_pids[det_valid_indices]
            det_local_order = np.argsort(det_filtered_pids, kind="mergesort")
            det_step_indices = det_valid_indices[det_local_order]
            sorted_det_pids = det_filtered_pids[det_local_order]
        else:
            det_step_indices = np.argsort(det_pids, kind="mergesort")
            sorted_det_pids = det_pids[det_step_indices]

        det_group_starts = np.searchsorted(sorted_det_pids, selected_pids, side="left")
        det_group_ends = np.searchsorted(sorted_det_pids, selected_pids, side="right")
        det_counts = (det_group_ends - det_group_starts).astype(np.int64)

        det_track_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), det_counts)
        det_local_ranks = np.arange(sorted_det_pids.size, dtype=np.int64) - det_group_starts[det_track_indices]

        # 2. Отбор и стабильная сортировка шагов в объёме фантома (Simulation_volume)
        sim_step_indices: NDArray[np.intp] | None = None
        sim_track_indices: NDArray[np.int64] | None = None
        sim_local_ranks: NDArray[np.int64] | None = None
        sim_counts = np.zeros(num_tracks, dtype=np.int64)

        if sim_table is not None and sim_table["particle_id"].size > 0:
            sim_pids = sim_table["particle_id"]
            sim_valid_indices = np.flatnonzero(np.isin(sim_pids, selected_pids))
            if sim_valid_indices.size > 0:
                sim_filtered_pids = sim_pids[sim_valid_indices]
                sim_local_order = np.argsort(sim_filtered_pids, kind="mergesort")
                sim_step_indices = sim_valid_indices[sim_local_order]
                sorted_sim_pids = sim_filtered_pids[sim_local_order]

                sim_group_starts = np.searchsorted(sorted_sim_pids, selected_pids, side="left")
                sim_group_ends = np.searchsorted(sorted_sim_pids, selected_pids, side="right")
                sim_counts = (sim_group_ends - sim_group_starts).astype(np.int64)

                sim_track_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), sim_counts)
                sim_local_ranks = (
                    np.arange(sorted_sim_pids.size, dtype=np.int64) - sim_group_starts[sim_track_indices]
                )

        # 3. Сопоставление с отсортированной таблицей initial_states
        init_matched_indices = np.searchsorted(sorted_init_pids, selected_pids)
        if bool(np.any(init_matched_indices >= sorted_init_pids.size)) or bool(
            np.any(sorted_init_pids[init_matched_indices] != selected_pids)
        ):
            raise KeyError("Не все particle_ID из группы детектора найдены в таблице 'initial_states'.")

        # 4. Расчёт CSR-сдвигов и целевых позиций в плоских массивах узлов
        track_lengths = 1 + sim_counts + det_counts
        offsets = np.empty(num_tracks + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(track_lengths, out=offsets[1:])
        total_vertices = int(offsets[-1])

        emission_target_indices = offsets[:-1]
        det_target_indices = offsets[det_track_indices] + 1 + sim_counts[det_track_indices] + det_local_ranks

        # 5. Выделение и векторное заполнение массивов узлов
        points = np.empty((total_vertices, 3), dtype=np.float32)
        global_direction = np.empty((total_vertices, 3), dtype=np.float32)
        energy_deposit = np.zeros(total_vertices, dtype=np.float32)
        distance_traveled = np.zeros(total_vertices, dtype=np.float32)
        process_code = np.empty(total_vertices, dtype=np.int32)
        volume_id = np.zeros(total_vertices, dtype=np.int32)
        material_id = np.zeros(total_vertices, dtype=np.int32)
        atomic_number = np.zeros(total_vertices, dtype=np.int32)
        species = np.zeros(total_vertices, dtype=np.int32)
        is_crystal_node = np.zeros(total_vertices, dtype=np.uint8)

        # Заполнение узла 0 каждого трека (точка рождения)
        points[emission_target_indices] = sorted_init_positions[init_matched_indices]
        global_direction[emission_target_indices] = sorted_init_directions[init_matched_indices]
        process_code[emission_target_indices] = EMISSION_PROCESS_CODE
        volume_id[emission_target_indices] = EMISSION_VOLUME_ID

        # Заполнение узлов из Simulation_volume
        if (
            sim_table is not None
            and sim_step_indices is not None
            and sim_track_indices is not None
            and sim_local_ranks is not None
        ):
            sim_target_indices = offsets[sim_track_indices] + 1 + sim_local_ranks
            points[sim_target_indices] = sim_table["global_position"][sim_step_indices]
            global_direction[sim_target_indices] = sim_table["global_direction"][sim_step_indices]
            energy_deposit[sim_target_indices] = sim_table["energy_deposit"][sim_step_indices]
            distance_traveled[sim_target_indices] = sim_table["distance_traveled"][sim_step_indices]
            process_code[sim_target_indices] = sim_table["process_code"][sim_step_indices]
            volume_id[sim_target_indices] = sim_table["volume_id"][sim_step_indices]
            material_id[sim_target_indices] = sim_table["material_id"][sim_step_indices]
            atomic_number[sim_target_indices] = sim_table["atomic_number"][sim_step_indices]
            species[sim_target_indices] = sim_table["species"][sim_step_indices]

        # Заполнение узлов из Detector_k
        points[det_target_indices] = det_table["global_position"][det_step_indices]
        global_direction[det_target_indices] = det_table["global_direction"][det_step_indices]
        energy_deposit[det_target_indices] = det_table["energy_deposit"][det_step_indices]
        distance_traveled[det_target_indices] = det_table["distance_traveled"][det_step_indices]
        process_code[det_target_indices] = det_table["process_code"][det_step_indices]
        volume_id[det_target_indices] = det_table["volume_id"][det_step_indices]
        material_id[det_target_indices] = det_table["material_id"][det_step_indices]
        atomic_number[det_target_indices] = det_table["atomic_number"][det_step_indices]
        species[det_target_indices] = det_table["species"][det_step_indices]
        is_crystal_node[det_target_indices] = CRYSTAL_NODE_FLAG

        # 6. Хронологическая сортировка узлов внутри каждого трека по накопленному пути distance_traveled
        all_track_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), track_lengths)
        is_emission_rank = np.ones(total_vertices, dtype=np.int8)
        is_emission_rank[emission_target_indices] = 0
        chronological_order = np.lexsort((distance_traveled, is_emission_rank, all_track_indices))

        points = points[chronological_order]
        global_direction = global_direction[chronological_order]
        energy_deposit = energy_deposit[chronological_order]
        distance_traveled = distance_traveled[chronological_order]
        process_code = process_code[chronological_order]
        volume_id = volume_id[chronological_order]
        material_id = material_id[chronological_order]
        atomic_number = atomic_number[chronological_order]
        species = species[chronological_order]
        is_crystal_node = is_crystal_node[chronological_order]

        point_data: dict[str, NDArray[Any]] = {
            "global_direction": global_direction,
            "energy_deposit": energy_deposit,
            "distance_traveled": distance_traveled,
            "process_code": process_code,
            "volume_id": volume_id,
            "material_id": material_id,
            "atomic_number": atomic_number,
            "species": species,
            "is_crystal_node": is_crystal_node,
        }

        cell_data: dict[str, NDArray[Any]] = {
            "particle_id": selected_pids.astype(np.int64, copy=False),
            "detector_id": np.full(num_tracks, detector_id, dtype=np.uint8),
            "projection_index": np.full(num_tracks, projection_index, dtype=np.int32),
            "gantry_angle": np.full(num_tracks, gantry_angle, dtype=np.float32),
            "emission_energy": sorted_init_energies[init_matched_indices],
            "emission_time": sorted_init_times[init_matched_indices],
        }

        return TrackBatch(
            points=points,
            offsets=offsets,
            point_data=point_data,
            cell_data=cell_data,
        )

    @staticmethod
    def _merge_external_tables(
        sim_table: dict[str, NDArray[Any]] | None,
        other_detector_tables: list[dict[str, NDArray[Any]]],
        target_pids: NDArray[np.int64],
    ) -> dict[str, NDArray[Any]] | None:
        """Дополняет внешние взаимодействия `Simulation_volume` шагами из других детекторов для совпадающих `particle_id`."""
        if not other_detector_tables or target_pids.size == 0:
            return sim_table

        matched_tables: list[dict[str, NDArray[Any]]] = []
        for other_table in other_detector_tables:
            other_pids = other_table["particle_id"]
            if other_pids.size == 0:
                continue
            shared_mask = np.isin(other_pids, target_pids)
            if bool(np.any(shared_mask)):
                matched_tables.append(
                    {key_name: arr_val[shared_mask] for key_name, arr_val in other_table.items()}
                )

        if not matched_tables:
            return sim_table

        all_tables = ([sim_table] if sim_table is not None else []) + matched_tables
        return {
            key_name: np.concatenate([tbl_item[key_name] for tbl_item in all_tables], axis=0)
            for key_name in matched_tables[0]
        }

    def assemble_file(
        self,
        file_path: str | Path,
        projection_index: int = 0,
        gantry_angle: float | None = None,
    ) -> tuple[TrackBatch, dict[str, Any]]:
        """Собирает все зарегистрированные треки частиц и паспортные метаданные из одного HDF5-файла.

        Args:
            file_path: Путь к входному HDF5-файлу симуляции Монте-Карло.
            projection_index: Порядковый индекс проекции (углового шага гантри).
            gantry_angle: Явный угол поворота гантри в радианах (если `None` — определяется из имени файла или метаданных).

        Returns:
            Кортеж `(track_batch, metadata)` с собранным пакетом треков всех детекторов и словарём метаданных.
        """
        resolved_path = Path(file_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Файл симуляции HDF5 не найден: {resolved_path}")
        if projection_index < 0:
            raise ValueError(f"Индекс проекции projection_index должен быть неотрицательным: {projection_index}.")

        extractor = SimulationDataExtractor(resolved_path)
        metadata = extractor.get_simulation_metadata()
        resolved_gantry_angle = self._resolve_gantry_angle(resolved_path, metadata, gantry_angle)

        with h5py.File(resolved_path, "r") as h5_file:
            if "initial_states" not in h5_file or not isinstance(h5_file["initial_states"], h5py.Group):
                raise KeyError(f"В файле '{resolved_path}' отсутствует обязательная группа 'initial_states'.")
            if "interactions" not in h5_file or not isinstance(h5_file["interactions"], h5py.Group):
                raise KeyError(f"В файле '{resolved_path}' отсутствует обязательная группа 'interactions'.")

            init_group = h5_file["initial_states"]
            init_pid_key = "particle_ID" if "particle_ID" in init_group else "particle_id"
            for req_init_key in (init_pid_key, "emission_position", "emission_direction", "emission_energy", "emission_time"):
                if req_init_key not in init_group:
                    raise KeyError(f"В группе 'initial_states' отсутствует обязательный датасет '{req_init_key}'.")

            init_pos_scale = self._resolve_dataset_unit_scale(init_group["emission_position"], float(mm))
            init_energy_scale = self._resolve_dataset_unit_scale(init_group["emission_energy"], float(MeV))
            init_time_scale = self._resolve_dataset_unit_scale(init_group["emission_time"], float(ns))

            raw_init_pids = np.asarray(init_group[init_pid_key][()], dtype=np.int64).ravel()
            raw_init_positions = (
                np.asarray(init_group["emission_position"][()], dtype=np.float32) * init_pos_scale
            ).astype(np.float32, copy=False)
            raw_init_directions = np.asarray(init_group["emission_direction"][()], dtype=np.float32)
            raw_init_energies = (
                np.asarray(init_group["emission_energy"][()], dtype=np.float32).ravel() * init_energy_scale
            ).astype(np.float32, copy=False)
            raw_init_times = (
                np.asarray(init_group["emission_time"][()], dtype=np.float32).ravel() * init_time_scale
            ).astype(np.float32, copy=False)

            init_sort_order = np.argsort(raw_init_pids)
            sorted_init_pids = raw_init_pids[init_sort_order]
            sorted_init_positions = raw_init_positions[init_sort_order]
            sorted_init_directions = raw_init_directions[init_sort_order]
            sorted_init_energies = raw_init_energies[init_sort_order]
            sorted_init_times = raw_init_times[init_sort_order]

            interactions_group = h5_file["interactions"]
            sim_table: dict[str, NDArray[Any]] | None = None
            if "Simulation_volume" in interactions_group and isinstance(
                interactions_group["Simulation_volume"], h5py.Group
            ):
                sim_table = self._read_interaction_group(interactions_group["Simulation_volume"])

            detector_names = sorted(
                group_name
                for group_name, group_obj in interactions_group.items()
                if group_name != "Simulation_volume" and isinstance(group_obj, h5py.Group)
            )

            if not detector_names:
                return self._create_empty_batch(), metadata

            parsed_detectors: list[tuple[int, dict[str, NDArray[Any]]]] = []
            for det_idx, detector_name in enumerate(detector_names):
                det_group = interactions_group[detector_name]
                detector_id = self._parse_detector_id(detector_name, fallback_id=det_idx + 1)
                det_table = self._read_interaction_group(det_group)
                parsed_detectors.append((detector_id, det_table))

            detector_batches: list[TrackBatch] = []
            for det_idx, (detector_id, det_table) in enumerate(parsed_detectors):
                other_tables = [
                    other_tbl
                    for other_idx, (_, other_tbl) in enumerate(parsed_detectors)
                    if other_idx != det_idx
                ]
                merged_ext_table = self._merge_external_tables(
                    sim_table=sim_table,
                    other_detector_tables=other_tables,
                    target_pids=np.unique(det_table["particle_id"]),
                )
                det_batch = self._assemble_detector_tracks(
                    det_table=det_table,
                    sim_table=merged_ext_table,
                    sorted_init_pids=sorted_init_pids,
                    sorted_init_positions=sorted_init_positions,
                    sorted_init_directions=sorted_init_directions,
                    sorted_init_energies=sorted_init_energies,
                    sorted_init_times=sorted_init_times,
                    detector_id=detector_id,
                    projection_index=projection_index,
                    gantry_angle=resolved_gantry_angle,
                    rng=self.rng,
                )
                detector_batches.append(det_batch)

        combined_batch = TrackBatch.concat(detector_batches)
        return combined_batch, metadata
