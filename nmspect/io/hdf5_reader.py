"""Чтение сырых данных Монте-Карло симуляций ОФЭКТ из файлов HDF5."""

from __future__ import annotations

import ctypes
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from hepunits import c_light, deg, mm, ns, rad
from numpy.typing import NDArray


def clear_hdf5_status_flags_if_needed(file_path: str | Path) -> bool:
    """Проверяет заголовок файла HDF5 и сбрасывает статус-флаги незакрытой сессии записи.

    Args:
        file_path: Путь к проверяемому HDF5-файлу.

    Returns:
        True, если флаги были обнаружены и успешно сброшены, иначе False.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        return False

    with resolved_path.open("rb") as binary_stream:
        header_bytes = binary_stream.read(16)

    if len(header_bytes) < 12 or header_bytes[:8] != b"\x89HDF\r\n\x1a\n" or header_bytes[11] == 0:
        return False

    h5py_dir = Path(h5py.__file__).resolve().parent
    dll_path = h5py_dir / "hdf5.dll"
    if not dll_path.is_file():
        return False

    hdf5_lib = ctypes.CDLL(str(dll_path))
    hid_type = ctypes.c_int64
    hdf5_lib.H5Pcreate.argtypes = [hid_type]
    hdf5_lib.H5Pcreate.restype = hid_type
    hdf5_lib.H5Pset.argtypes = [hid_type, ctypes.c_char_p, ctypes.c_void_p]
    hdf5_lib.H5Pset.restype = ctypes.c_int
    hdf5_lib.H5Fopen.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_type]
    hdf5_lib.H5Fopen.restype = hid_type
    hdf5_lib.H5Fclose.argtypes = [hid_type]
    hdf5_lib.H5Fclose.restype = ctypes.c_int
    hdf5_lib.H5Pclose.argtypes = [hid_type]
    hdf5_lib.H5Pclose.restype = ctypes.c_int

    file_access_cls = hid_type.in_dll(hdf5_lib, "H5P_CLS_FILE_ACCESS_ID_g").value
    fapl_id = hdf5_lib.H5Pcreate(file_access_cls)
    flag_value = ctypes.c_bool(True)
    hdf5_lib.H5Pset(fapl_id, b"clear_status_flags", ctypes.byref(flag_value))
    file_id = hdf5_lib.H5Fopen(str(resolved_path).encode("utf-8"), 1, fapl_id)
    was_cleared = bool(file_id > 0)
    if was_cleared:
        hdf5_lib.H5Fclose(file_id)
    hdf5_lib.H5Pclose(fapl_id)
    return was_cleared


def parse_gantry_angle_from_filename(filename: str | Path) -> float | None:
    """Извлекает угол поворота гантри из имени файла симуляции в базовых единицах hepunits (радианах).

    Поддерживаемые форматы имён:
    - '0.0_deg.hdf', '3.0_deg.hdf'
    - 'gantry_0.0_deg.hdf', 'angle_45.0_deg.h5'
    - '0.0 deg.hdf5'
    - '0.0.hdf'

    Args:
        filename: Имя файла или полный путь к файлу симуляции.

    Returns:
        Угол поворота гантри в базовых единицах hepunits (умноженный на deg) или None.
    """
    base_name = Path(filename).name
    stem_name = Path(filename).stem

    match_underscore = re.search(r"(?:gantry_|angle_)?([+-]?[0-9]+(?:\.[0-9]+)?)_deg", base_name, re.IGNORECASE)
    if match_underscore is not None:
        return float(match_underscore.group(1)) * deg

    match_space = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*deg", base_name, re.IGNORECASE)
    if match_space is not None:
        return float(match_space.group(1)) * deg

    match_stem = re.fullmatch(r"(?:gantry_|angle_)?([+-]?[0-9]+(?:\.[0-9]+)?)", stem_name, re.IGNORECASE)
    if match_stem is not None:
        return float(match_stem.group(1)) * deg

    return None


def _convert_hdf5_value(raw_val: Any) -> Any:
    """Преобразует скалярные и строковые атрибуты HDF5 к стандартным типам Python и NumPy."""
    if isinstance(raw_val, bytes):
        return raw_val.decode("utf-8", errors="replace")
    if isinstance(raw_val, np.bytes_):
        return bytes(raw_val).decode("utf-8", errors="replace")
    if isinstance(raw_val, np.generic):
        return raw_val.item()
    return raw_val


class SimulationDataExtractor:
    """Извлекает сырые микроскопические акты взаимодействия ($N$), кратности рассеяния и метаданные из HDF5."""

    def __init__(self, file_path: str | Path) -> None:
        """Инициализирует экстрактор данных симуляции для указанного файла HDF5.

        Args:
            file_path: Путь к входному HDF5-файлу симуляции Монте-Карло.
        """
        resolved_path = Path(file_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Файл симуляции HDF5 не найден: {resolved_path}")

        self.file_path: Path = resolved_path
        self.scattering_orders: dict[str, NDArray[np.int32]] = {}
        clear_hdf5_status_flags_if_needed(self.file_path)

    @staticmethod
    def _read_dataset_array(dataset: h5py.Dataset) -> NDArray[Any]:
        """Считывает содержимое датасета HDF5 в массив NumPy."""
        return np.array(dataset[()], copy=True)

    @staticmethod
    def _find_pose_matrix(detector_name: str, metadata: dict[str, Any]) -> NDArray[np.float64] | None:
        """Находит аффинную матрицу трансформации 4x4 для указанного детектора."""
        gantry_meta = metadata.get("gantry", {})
        match_det = re.search(r"Detector_(\d+)$", detector_name, re.IGNORECASE)
        if match_det is not None:
            pose_key = f"pose_matrix_detector_{match_det.group(1)}"
            if pose_key in gantry_meta:
                return np.asarray(gantry_meta[pose_key], dtype=np.float64)

        volume_matrices = metadata.get("volume_matrices", {})
        if detector_name in volume_matrices:
            return np.asarray(volume_matrices[detector_name], dtype=np.float64)

        return None

    def get_detectors_names(self) -> list[str]:
        """Возвращает отсортированный список имён чувствительных объёмов детекторов в файле.

        Returns:
            Список строковых идентификаторов детекторов (например, ['Detector_1', 'Detector_2']).
        """
        excluded_names = {"Simulation_volume", "Phantom", "metadata", "initial_states", "interactions"}
        detector_names: list[str] = []

        with h5py.File(self.file_path, "r") as h5_file:
            target_group: h5py.Group = (
                h5_file["interactions"]
                if "interactions" in h5_file and isinstance(h5_file["interactions"], h5py.Group)
                else h5_file
            )
            for group_name, group_obj in target_group.items():
                if group_name in excluded_names:
                    continue
                if isinstance(group_obj, h5py.Group):
                    detector_names.append(str(group_name))

        return sorted(detector_names)

    def get_detectors_angles(self) -> list[float]:
        """Возвращает углы поворота всех детекторов в базовых единицах hepunits (радианах).

        Returns:
            Список углов проекций в радианах (hepunits), согласованный по порядку с `get_detectors_names()`.
        """
        detector_names = self.get_detectors_names()
        if not detector_names:
            return []

        metadata = self.get_simulation_metadata()
        gantry_meta = metadata.get("gantry", {})
        filename_angle = parse_gantry_angle_from_filename(self.file_path.name)

        base_angle: float | None = None
        if "gantry_angle" in gantry_meta:
            raw_angle = float(np.asarray(gantry_meta["gantry_angle"]).item())
            units_str = str(gantry_meta.get("gantry_angle_units", "")).strip().lower()
            if units_str in ("deg", "degree", "degrees"):
                base_angle = raw_angle * deg
            elif units_str in ("rad", "radian", "radians"):
                base_angle = raw_angle * rad
            elif filename_angle is not None and abs(raw_angle) > 0.0 and np.isclose(raw_angle * deg, filename_angle) or abs(raw_angle) > (2.0 * np.pi + 1e-2):
                base_angle = raw_angle * deg
            else:
                base_angle = raw_angle * rad

        if base_angle is None:
            base_angle = filename_angle if filename_angle is not None else 0.0 * rad

        full_turn = 360.0 * deg
        half_turn = 180.0 * deg
        angles: list[float] = []

        with h5py.File(self.file_path, "r") as h5_file:
            target_group: h5py.Group = (
                h5_file["interactions"]
                if "interactions" in h5_file and isinstance(h5_file["interactions"], h5py.Group)
                else h5_file
            )
            for det_idx, det_name in enumerate(detector_names):
                det_obj = target_group[det_name]
                if "angle" in det_obj.attrs:
                    raw_det_angle = float(np.asarray(det_obj.attrs["angle"]).item())
                    det_units = str(det_obj.attrs.get("angle_units", "deg")).strip().lower()
                    det_angle = raw_det_angle * rad if det_units.startswith("rad") else raw_det_angle * deg
                    angles.append(float(det_angle % full_turn))
                    continue

                if det_name.endswith("_2") or (det_idx == 1 and len(detector_names) == 2):
                    angles.append(float((base_angle + half_turn) % full_turn))
                else:
                    offset = float(det_idx) * (full_turn / float(max(len(detector_names), 1)))
                    angles.append(float((base_angle + offset) % full_turn))

        return angles

    def get_detector_size(self, detector_name: str | None = None) -> tuple[float, float, float]:
        """Возвращает физические габариты кристалла детектора (X, Y, Z) в базовых единицах hepunits (мм).

        Args:
            detector_name: Опциональное имя конкретного детектора.

        Returns:
            Кортеж (size_x, size_y, size_z) в базовых единицах hepunits (mm).
        """
        with h5py.File(self.file_path, "r") as h5_file:
            if detector_name is not None:
                target_group = (
                    h5_file["interactions"]
                    if "interactions" in h5_file and isinstance(h5_file["interactions"], h5py.Group)
                    else h5_file
                )
                if detector_name in target_group:
                    det_obj = target_group[detector_name]
                    for attr_key in ("size", "detector_size", "crystal_dimensions", "dimensions"):
                        if attr_key in det_obj.attrs:
                            dims_arr = np.asarray(det_obj.attrs[attr_key], dtype=np.float64).ravel()
                            if dims_arr.size >= 3:
                                return (float(dims_arr[0]) * mm, float(dims_arr[1]) * mm, float(dims_arr[2]) * mm)
                            if dims_arr.size == 2:
                                return (float(dims_arr[0]) * mm, float(dims_arr[1]) * mm, 9.5 * mm)

            if "metadata" in h5_file and isinstance(h5_file["metadata"], h5py.Group):
                meta_grp = h5_file["metadata"]
                if "geometry" in meta_grp and isinstance(meta_grp["geometry"], h5py.Group):
                    geom_grp = meta_grp["geometry"]
                    if "crystal_dimensions" in geom_grp.attrs:
                        dims_arr = np.asarray(geom_grp.attrs["crystal_dimensions"], dtype=np.float64).ravel()
                        if dims_arr.size >= 3:
                            return (float(dims_arr[0]) * mm, float(dims_arr[1]) * mm, float(dims_arr[2]) * mm)
                    if "crystal_dimensions" in geom_grp and isinstance(geom_grp["crystal_dimensions"], h5py.Dataset):
                        dims_arr = np.asarray(geom_grp["crystal_dimensions"][()], dtype=np.float64).ravel()
                        if dims_arr.size >= 3:
                            return (float(dims_arr[0]) * mm, float(dims_arr[1]) * mm, float(dims_arr[2]) * mm)

            for root_attr in ("detector_size", "crystal_dimensions", "size"):
                if root_attr in h5_file.attrs:
                    dims_arr = np.asarray(h5_file.attrs[root_attr], dtype=np.float64).ravel()
                    if dims_arr.size >= 3:
                        return (float(dims_arr[0]) * mm, float(dims_arr[1]) * mm, float(dims_arr[2]) * mm)

        return (540.0 * mm, 400.0 * mm, 9.5 * mm)

    def get_simulation_metadata(self) -> dict[str, Any]:
        """Извлекает паспортные метаданные симуляции (гантри, геометрия, источник, матрицы объёмов).

        Returns:
            Словарь с разделами 'gantry', 'geometry', 'source', 'simulation', 'protocol', 'volume_matrices'.
        """
        metadata: dict[str, Any] = {
            "gantry": {},
            "geometry": {},
            "source": {},
            "simulation": {},
            "protocol": {},
            "volume_matrices": {},
        }

        with h5py.File(self.file_path, "r") as h5_file:
            if "metadata" not in h5_file or not isinstance(h5_file["metadata"], h5py.Group):
                return metadata

            meta_group = h5_file["metadata"]
            for section_name in ("gantry", "geometry", "source", "simulation", "protocol"):
                if section_name in meta_group and isinstance(meta_group[section_name], h5py.Group):
                    sub_group = meta_group[section_name]
                    section_dict: dict[str, Any] = {
                        str(attr_key): _convert_hdf5_value(attr_val)
                        for attr_key, attr_val in sub_group.attrs.items()
                    }
                    for item_key, item_obj in sub_group.items():
                        if isinstance(item_obj, h5py.Dataset) and item_key not in section_dict:
                            section_dict[str(item_key)] = _convert_hdf5_value(item_obj[()])
                    metadata[section_name] = section_dict

            if "geometry" in meta_group and isinstance(meta_group["geometry"], h5py.Group):
                geom_group = meta_group["geometry"]
                volumes_container: h5py.Group = (
                    geom_group["volumes"]
                    if "volumes" in geom_group and isinstance(geom_group["volumes"], h5py.Group)
                    else geom_group
                )
                for vol_name, vol_obj in volumes_container.items():
                    if isinstance(vol_obj, h5py.Group):
                        if "global_matrix" in vol_obj.attrs:
                            metadata["volume_matrices"][str(vol_name)] = np.asarray(
                                vol_obj.attrs["global_matrix"], dtype=np.float64
                            )
                        elif "global_matrix" in vol_obj and isinstance(vol_obj["global_matrix"], h5py.Dataset):
                            metadata["volume_matrices"][str(vol_name)] = np.asarray(
                                vol_obj["global_matrix"][()], dtype=np.float64
                            )

        return metadata

    def extract_data(self, detector_name: str) -> NDArray[np.float64]:
        """Извлекает матрицу микроскопических актов взаимодействия `(N, 5)` и сохраняет кратности рассеяния.

        Столбцы возвращаемой матрицы `(N, 5)` в базовых единицах hepunits:
            `[particle_id, time, energy_deposit, local_x, local_y]`

        Args:
            detector_name: Имя детектора (например, 'Detector_1').

        Returns:
            Двумерный массив формы `(N, 5)` с типом `float64`.
        """
        metadata = self.get_simulation_metadata()

        with h5py.File(self.file_path, "r") as h5_file:
            interactions_grp: h5py.Group | None = (
                h5_file["interactions"]
                if "interactions" in h5_file and isinstance(h5_file["interactions"], h5py.Group)
                else None
            )
            if interactions_grp is not None and detector_name in interactions_grp:
                det_node = interactions_grp[detector_name]
            elif detector_name in h5_file:
                det_node = h5_file[detector_name]
            else:
                raise KeyError(f"Детектор '{detector_name}' отсутствует в файле '{self.file_path}'.")

            if not isinstance(det_node, h5py.Group):
                raise TypeError(
                    f"Узел детектора '{detector_name}' должен быть группой h5py.Group, "
                    f"получено: {type(det_node).__name__}"
                )

            det_group = det_node
            pid_key = "particle_ID" if "particle_ID" in det_group else "particle_id"
            if pid_key not in det_group:
                raise KeyError(f"В группе детектора '{detector_name}' отсутствует датасет 'particle_ID'.")
            if "energy_deposit" not in det_group:
                raise KeyError(f"В группе детектора '{detector_name}' отсутствует датасет 'energy_deposit'.")

            raw_pids = self._read_dataset_array(det_group[pid_key]).ravel()
            raw_energy = np.asarray(self._read_dataset_array(det_group["energy_deposit"]), dtype=np.float64).ravel()

            if "local_position" in det_group:
                local_pos = np.asarray(self._read_dataset_array(det_group["local_position"]), dtype=np.float64)
                local_x = local_pos[:, 0] * mm
                local_y = local_pos[:, 1] * mm
            elif "local_x" in det_group and "local_y" in det_group:
                local_x = np.asarray(self._read_dataset_array(det_group["local_x"]), dtype=np.float64).ravel() * mm
                local_y = np.asarray(self._read_dataset_array(det_group["local_y"]), dtype=np.float64).ravel() * mm
            elif "global_position" in det_group:
                global_pos = np.asarray(self._read_dataset_array(det_group["global_position"]), dtype=np.float64)
                pose_matrix = self._find_pose_matrix(detector_name, metadata)
                if pose_matrix is not None:
                    rotation = pose_matrix[:3, :3]
                    translation = pose_matrix[:3, 3]
                    local_pos = (global_pos - translation) @ rotation
                else:
                    local_pos = global_pos
                local_x = local_pos[:, 0] * mm
                local_y = local_pos[:, 1] * mm
            else:
                raise KeyError(
                    f"В группе детектора '{detector_name}' отсутствуют координатные датасеты "
                    "('local_position' или 'global_position')."
                )

            num_acts = min(raw_pids.size, raw_energy.size, local_x.size, local_y.size)
            if num_acts == 0:
                self.scattering_orders[detector_name] = np.empty(0, dtype=np.int32)
                return np.empty((0, 5), dtype=np.float64)

            raw_pids = raw_pids[:num_acts]
            raw_energy = raw_energy[:num_acts]
            local_x = local_x[:num_acts]
            local_y = local_y[:num_acts]

            # Определение времени актов взаимодействия (в базовых единицах hepunits: ns)
            if "time" in det_group:
                act_times = np.asarray(self._read_dataset_array(det_group["time"])[:num_acts], dtype=np.float64) * ns
            elif "registration_time" in det_group:
                act_times = (
                    np.asarray(self._read_dataset_array(det_group["registration_time"])[:num_acts], dtype=np.float64)
                    * ns
                )
            elif "emission_time" in det_group:
                act_times = (
                    np.asarray(self._read_dataset_array(det_group["emission_time"])[:num_acts], dtype=np.float64) * ns
                )
            elif "initial_states" in h5_file and isinstance(h5_file["initial_states"], h5py.Group):
                init_grp = h5_file["initial_states"]
                init_pid_key = "particle_ID" if "particle_ID" in init_grp else "particle_id"
                init_time_key = "emission_time" if "emission_time" in init_grp else "time"
                if init_pid_key in init_grp and init_time_key in init_grp:
                    init_pids = self._read_dataset_array(init_grp[init_pid_key]).ravel()
                    init_times = np.asarray(self._read_dataset_array(init_grp[init_time_key]), dtype=np.float64).ravel()
                    if init_pids.size > 0:
                        sort_order = np.argsort(init_pids)
                        sorted_init_pids = init_pids[sort_order]
                        found_idx = np.searchsorted(sorted_init_pids, raw_pids)
                        found_idx = np.clip(found_idx, 0, sorted_init_pids.size - 1)
                        matched_mask = sorted_init_pids[found_idx] == raw_pids
                        mapped_indices = sort_order[found_idx]
                        act_times = np.where(matched_mask, init_times[mapped_indices], 0.0) * ns
                    else:
                        act_times = np.zeros(num_acts, dtype=np.float64)
                else:
                    act_times = np.zeros(num_acts, dtype=np.float64)
            else:
                act_times = np.zeros(num_acts, dtype=np.float64)

            if "distance_traveled" in det_group:
                distances = np.asarray(
                    self._read_dataset_array(det_group["distance_traveled"])[:num_acts], dtype=np.float64
                )
                act_times = act_times + (distances * mm) / c_light

            # Расчёт кратности предварительного комптоновского рассеяния в фантоме (Simulation_volume)
            scatter_orders = np.zeros(num_acts, dtype=np.int32)
            for order_key in ("scattering_order", "scattering_orders", "scatter_order"):
                if order_key in det_group:
                    scatter_orders = np.asarray(
                        self._read_dataset_array(det_group[order_key])[:num_acts], dtype=np.int32
                    ).ravel()
                    break
            else:
                if interactions_grp is not None and "Simulation_volume" in interactions_grp:
                    sim_vol_grp = interactions_grp["Simulation_volume"]
                    sim_pid_key = "particle_ID" if "particle_ID" in sim_vol_grp else "particle_id"
                    if isinstance(sim_vol_grp, h5py.Group) and sim_pid_key in sim_vol_grp:
                        sim_pids = self._read_dataset_array(sim_vol_grp[sim_pid_key]).ravel()
                        if "process_code" in sim_vol_grp:
                            proc_ds = sim_vol_grp["process_code"]
                            proc_codes = self._read_dataset_array(proc_ds).ravel()
                            raw_mapping = _convert_hdf5_value(proc_ds.attrs.get("mapping", "{}"))
                            compton_codes: list[int] = []
                            if isinstance(raw_mapping, str) and raw_mapping.startswith("{"):
                                parsed_map = json.loads(raw_mapping)
                                for code_str, proc_name in parsed_map.items():
                                    if "compton" in str(proc_name).lower() or "scatter" in str(proc_name).lower():
                                        compton_codes.append(int(code_str))
                            if compton_codes:
                                min_sim_len = min(sim_pids.size, proc_codes.size)
                                valid_scatter = np.isin(proc_codes[:min_sim_len], compton_codes)
                                sim_pids = sim_pids[:min_sim_len][valid_scatter]

                        if sim_pids.size > 0:
                            unique_sim_pids, sim_counts = np.unique(sim_pids, return_counts=True)
                            pos_idx = np.searchsorted(unique_sim_pids, raw_pids)
                            pos_idx = np.clip(pos_idx, 0, unique_sim_pids.size - 1)
                            matched_sim = unique_sim_pids[pos_idx] == raw_pids
                            scatter_orders = np.where(matched_sim, sim_counts[pos_idx], 0).astype(np.int32)

        self.scattering_orders[detector_name] = scatter_orders
        acts_matrix = np.column_stack(
            (
                raw_pids.astype(np.float64),
                act_times,
                raw_energy,
                local_x,
                local_y,
            )
        )
        return acts_matrix

    def get_scattering_orders(self, detector_name: str) -> NDArray[np.int32]:
        """Возвращает массив кратностей рассеяния в фантоме для каждого акта в указанном детекторе.

        Args:
            detector_name: Имя детектора (например, 'Detector_1').

        Returns:
            Одномерный массив `int32` длины `N`.
        """
        if detector_name not in self.scattering_orders:
            self.extract_data(detector_name)
        return self.scattering_orders[detector_name]
