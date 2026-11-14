"""Векторный аннотатор прохождения фотонов через коллиматор и углов наклона траекторий."""

from __future__ import annotations

from typing import Any

import numpy as np
from hepunits import mm
from numpy.typing import NDArray

from nmspect.tracks.geometry import (
    MIN_VECTOR_NORM,
    extract_orthonormal_frame_from_pose,
    locate_hexagonal_channel_axial,
    locate_square_channel_grid,
    resolve_metadata_gantry_angle,
    rotate_vectors_around_z,
)
from nmspect.tracks.models import (
    CRYSTAL_NODE_FLAG,
    TrackAnnotator,
    TrackBatch,
)


class CollimatorTrackAnnotator(TrackAnnotator):
    """Аннотирует треки количеством пересечённых септ коллиматора (`septal_crossings`) и углами `tan_theta`, `tan_theta_axial`.

    Все геометрические расчёты (пересечение луча подлёта с плоскостями входа/выхода коллиматора,
    перевод в координаты каналов и расчёт поперечного и аксиального тангенсов угла наклона траектории)
    выполняются полностью векторно средствами NumPy без циклов по трекам.
    """

    def __init__(
        self,
        hole_diameter: float = 1.5 * mm,
        septa_thickness: float = 0.2 * mm,
        hole_length: float = 35.0 * mm,
        orbit_radius: float = 250.0 * mm,
        crystal_thickness: float = 9.5 * mm,
        hole_shape: str = "hexagonal",
    ) -> None:
        """Инициализирует геометрический аннотатор коллиматора параметрами по умолчанию.

        Args:
            hole_diameter: Диаметр (размер под ключ) канала коллиматора в единицах `hepunits` (`mm`).
            septa_thickness: Толщина свинцовой перегородки (септы) между каналами в единицах `hepunits` (`mm`).
            hole_length: Длина каналов (толщина коллиматора) вдоль нормали детектора в единицах `hepunits` (`mm`).
            orbit_radius: Радиус орбиты детектора по умолчанию в единицах `hepunits` (`mm`).
            crystal_thickness: Толщина сцинтилляционного кристалла по умолчанию в единицах `hepunits` (`mm`).
            hole_shape: Форма сечения канала (`'hexagonal'` или `'square'`).
        """
        if hole_diameter <= 0.0:
            raise ValueError(f"Параметр hole_diameter должен быть строго положительным, получено: {hole_diameter}.")
        if septa_thickness <= 0.0:
            raise ValueError(f"Параметр septa_thickness должен быть строго положительным, получено: {septa_thickness}.")
        if hole_length <= 0.0:
            raise ValueError(f"Параметр hole_length должен быть строго положительным, получено: {hole_length}.")
        if orbit_radius <= 0.0:
            raise ValueError(f"Параметр orbit_radius должен быть строго положительным, получено: {orbit_radius}.")
        if crystal_thickness <= 0.0:
            raise ValueError(
                f"Параметр crystal_thickness должен быть строго положительным, получено: {crystal_thickness}."
            )
        normalized_shape = hole_shape.strip().lower()
        if normalized_shape not in ("hexagonal", "square"):
            raise ValueError(f"Неподдерживаемая форма отверстий коллиматора: '{hole_shape}'.")

        self.hole_diameter: float = float(hole_diameter)
        self.septa_thickness: float = float(septa_thickness)
        self.hole_length: float = float(hole_length)
        self.orbit_radius: float = float(orbit_radius)
        self.crystal_thickness: float = float(crystal_thickness)
        self.hole_shape: str = normalized_shape

    def _resolve_collimator_geometry(
        self,
        metadata: dict[str, Any] | None,
    ) -> tuple[float, float, float, str]:
        """Извлекает положительные геометрические параметры коллиматора из `metadata` или возвращает значения `self`."""
        resolved_diameter = self.hole_diameter
        resolved_septa = self.septa_thickness
        resolved_length = self.hole_length
        resolved_shape = self.hole_shape

        if not isinstance(metadata, dict):
            return resolved_diameter, resolved_septa, resolved_length, resolved_shape

        collimator_meta = metadata.get("collimator")
        geometry_meta = metadata.get("geometry")

        for source_dict in (collimator_meta, geometry_meta):
            if not isinstance(source_dict, dict):
                continue

            for diameter_key in ("hole_diameter", "collimator_hole_diameter"):
                if diameter_key in source_dict and resolved_diameter == self.hole_diameter:
                    raw_val = float(np.asarray(source_dict[diameter_key]).item()) * float(mm)
                    if raw_val > 0.0:
                        resolved_diameter = raw_val
                        break

            for septa_key in ("septa_thickness", "collimator_septa_thickness"):
                if septa_key in source_dict and resolved_septa == self.septa_thickness:
                    raw_val = float(np.asarray(source_dict[septa_key]).item()) * float(mm)
                    if raw_val > 0.0:
                        resolved_septa = raw_val
                        break

            for length_key in ("hole_length", "collimator_hole_length"):
                if length_key in source_dict and resolved_length == self.hole_length:
                    raw_val = float(np.asarray(source_dict[length_key]).item()) * float(mm)
                    if raw_val > 0.0:
                        resolved_length = raw_val
                        break

            for shape_key in ("hole_shape", "collimator_hole_shape"):
                if shape_key in source_dict:
                    shape_cand = str(source_dict[shape_key]).strip().lower()
                    if "square" in shape_cand:
                        resolved_shape = "square"
                    elif "hex" in shape_cand:
                        resolved_shape = "hexagonal"

        return resolved_diameter, resolved_septa, resolved_length, resolved_shape

    @staticmethod
    def _find_detector_global_matrix(
        det_id: int,
        metadata: dict[str, Any] | None,
    ) -> tuple[NDArray[np.float64] | None, bool]:
        """Ищет 4x4 матрицу коллиматора или детектора в `metadata['volume_matrices']`."""
        if not isinstance(metadata, dict):
            return None, False

        volume_matrices = metadata.get("volume_matrices")
        if not isinstance(volume_matrices, dict):
            return None, False

        collimator_keys = (
            f"Collimator_{det_id}_channels",
            f"Collimator_{det_id}_body",
            f"Collimator_{det_id}",
        )
        for cand_key in collimator_keys:
            if cand_key in volume_matrices:
                mat_arr = np.asarray(volume_matrices[cand_key], dtype=np.float64)
                if mat_arr.shape == (4, 4):
                    return mat_arr, False

        detector_key = f"Detector_{det_id}"
        if detector_key in volume_matrices:
            mat_arr = np.asarray(volume_matrices[detector_key], dtype=np.float64)
            if mat_arr.shape == (4, 4):
                return mat_arr, True

        return None, False

    def _build_track_frames(
        self,
        detector_ids: NDArray[np.uint8],
        gantry_angles: NDArray[np.float64],
        crystal_entry_points: NDArray[np.float64],
        hole_length: float,
        metadata: dict[str, Any] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Формирует массивы `(collimator_centers, axis_normal, axis_transverse, axis_axial)` с учётом вращения гантри."""
        num_tracks = int(detector_ids.size)
        collimator_centers = np.zeros((num_tracks, 3), dtype=np.float64)
        axis_normal = np.zeros((num_tracks, 3), dtype=np.float64)
        axis_transverse = np.zeros((num_tracks, 3), dtype=np.float64)
        axis_axial = np.zeros((num_tracks, 3), dtype=np.float64)

        resolved_crystal_thickness = self.crystal_thickness
        if isinstance(metadata, dict):
            geom_meta = metadata.get("geometry")
            if isinstance(geom_meta, dict) and "crystal_dimensions" in geom_meta:
                raw_dims = np.asarray(geom_meta["crystal_dimensions"], dtype=np.float64).ravel()
                if raw_dims.size >= 3 and float(raw_dims[2]) > 0.0:
                    resolved_crystal_thickness = float(raw_dims[2]) * float(mm)

        meta_gantry_angle = resolve_metadata_gantry_angle(metadata)
        unique_detectors = np.unique(detector_ids)

        for raw_det_id in unique_detectors:
            det_id = int(raw_det_id)
            det_mask = detector_ids == raw_det_id
            mask_count = int(np.count_nonzero(det_mask))
            global_matrix, is_detector_only = self._find_detector_global_matrix(det_id, metadata)

            if global_matrix is not None:
                base_center, base_norm, base_trans, base_ax = extract_orthonormal_frame_from_pose(global_matrix)
                if is_detector_only:
                    base_center = base_center - base_norm * (0.5 * resolved_crystal_thickness + 0.5 * hole_length)

                delta_angles = gantry_angles[det_mask] - meta_gantry_angle
                collimator_centers[det_mask] = rotate_vectors_around_z(
                    np.broadcast_to(base_center, (mask_count, 3)),
                    delta_angles,
                )
                axis_normal[det_mask] = rotate_vectors_around_z(
                    np.broadcast_to(base_norm, (mask_count, 3)),
                    delta_angles,
                )
                axis_transverse[det_mask] = rotate_vectors_around_z(
                    np.broadcast_to(base_trans, (mask_count, 3)),
                    delta_angles,
                )
                axis_axial[det_mask] = rotate_vectors_around_z(
                    np.broadcast_to(base_ax, (mask_count, 3)),
                    delta_angles,
                )
            else:
                det_angles = gantry_angles[det_mask] + float(det_id - 1) * np.pi
                cos_ang = np.cos(det_angles)
                sin_ang = np.sin(det_angles)
                zeros_arr = np.zeros_like(cos_ang)
                ones_arr = np.ones_like(cos_ang)

                normal_y = np.column_stack((-sin_ang, cos_ang, zeros_arr))
                transverse_y = np.column_stack((cos_ang, sin_ang, zeros_arr))

                normal_x = np.column_stack((cos_ang, sin_ang, zeros_arr))
                transverse_x = np.column_stack((sin_ang, -cos_ang, zeros_arr))

                det_entry_pts = crystal_entry_points[det_mask]
                proj_y = float(np.mean(np.sum(det_entry_pts * normal_y, axis=1)))
                proj_x = float(np.mean(np.sum(det_entry_pts * normal_x, axis=1)))

                if abs(proj_x) > abs(proj_y):
                    sign_flip = 1.0 if proj_x >= 0.0 else -1.0
                    chosen_normal = sign_flip * normal_x
                    chosen_transverse = sign_flip * transverse_x
                else:
                    sign_flip = 1.0 if proj_y >= 0.0 else -1.0
                    chosen_normal = sign_flip * normal_y
                    chosen_transverse = sign_flip * transverse_y

                chosen_axial = np.column_stack((zeros_arr, zeros_arr, ones_arr))
                collimator_radius = max(
                    self.orbit_radius - 0.5 * resolved_crystal_thickness - 0.5 * hole_length,
                    0.5 * hole_length,
                )

                collimator_centers[det_mask] = chosen_normal * collimator_radius
                axis_normal[det_mask] = chosen_normal
                axis_transverse[det_mask] = chosen_transverse
                axis_axial[det_mask] = chosen_axial

        return collimator_centers, axis_normal, axis_transverse, axis_axial

    def annotate(self, tracks: TrackBatch, metadata: dict[str, Any] | None = None) -> None:
        """Вычисляет и записывает в `tracks.cell_data` поля `'septal_crossings'`, `'tan_theta'` и `'tan_theta_axial'`.

        Args:
            tracks: Пакет треков `TrackBatch` в CSR-представлении.
            metadata: Опциональный словарь метаданных симуляции с параметрами коллиматора и матрицами объёмов.

        Raises:
            KeyError: Если в `tracks.point_data` отсутствует обязательное поле `'is_crystal_node'`.
        """
        if "is_crystal_node" not in tracks.point_data:
            raise KeyError(
                "В tracks.point_data отсутствует обязательный ключ 'is_crystal_node' "
                "для геометрической аннотации коллиматора."
            )

        hole_diameter, septa_thickness, hole_length, hole_shape = self._resolve_collimator_geometry(metadata)

        if tracks.num_tracks == 0:
            tracks.cell_data["septal_crossings"] = np.empty(0, dtype=np.uint8)
            tracks.cell_data["tan_theta"] = np.empty(0, dtype=np.float32)
            tracks.cell_data["tan_theta_axial"] = np.empty(0, dtype=np.float32)
            return

        num_tracks = tracks.num_tracks
        num_points = tracks.num_points
        track_starts = tracks.offsets[:-1]
        track_ends = tracks.offsets[1:]

        emission_points = tracks.points[track_starts].astype(np.float64)

        crystal_mask = tracks.point_data["is_crystal_node"] == CRYSTAL_NODE_FLAG
        crystal_flags = crystal_mask.astype(np.int64)
        crystal_counts = np.add.reduceat(crystal_flags, track_starts)
        has_crystal = crystal_counts > 0

        node_indices = np.arange(num_points, dtype=np.int64)
        raw_first_crystal = np.minimum.reduceat(
            np.where(crystal_mask, node_indices, num_points),
            track_starts,
        )
        first_crystal_indices = np.where(has_crystal, raw_first_crystal, track_ends - 1)
        crystal_entry_points = tracks.points[first_crystal_indices].astype(np.float64)

        if "detector_id" in tracks.cell_data:
            detector_ids = np.asarray(tracks.cell_data["detector_id"], dtype=np.uint8)
        else:
            detector_ids = np.ones(num_tracks, dtype=np.uint8)

        if "gantry_angle" in tracks.cell_data:
            gantry_angles = np.asarray(tracks.cell_data["gantry_angle"], dtype=np.float64)
        else:
            gantry_angles = np.full(num_tracks, resolve_metadata_gantry_angle(metadata), dtype=np.float64)

        collimator_centers, axis_normal, axis_transverse, axis_axial = self._build_track_frames(
            detector_ids=detector_ids,
            gantry_angles=gantry_angles,
            crystal_entry_points=crystal_entry_points,
            hole_length=hole_length,
            metadata=metadata,
        )

        # Вычисление проекций кратчайшего вектора перемещения tan_theta и tan_theta_axial
        detection_points = crystal_entry_points
        if "scintillation_position" in tracks.cell_data:
            scint_pos = np.asarray(tracks.cell_data["scintillation_position"], dtype=np.float64)
            if scint_pos.shape == (num_tracks, 3):
                valid_scint = np.all(np.isfinite(scint_pos), axis=1)
                detection_points = np.where(valid_scint[:, np.newaxis], scint_pos, crystal_entry_points)

        shortest_vec = detection_points - emission_points
        vec_norm = np.linalg.norm(shortest_vec, axis=1)
        transverse_offset = np.sum(shortest_vec * axis_transverse, axis=1)
        axial_offset = np.sum(shortest_vec * axis_axial, axis=1)
        safe_vec_norm = np.where(vec_norm > MIN_VECTOR_NORM, vec_norm, 1.0)
        valid_tan_mask = has_crystal & (vec_norm > MIN_VECTOR_NORM)
        tan_theta = np.where(
            valid_tan_mask,
            transverse_offset / safe_vec_norm,
            0.0,
        ).astype(np.float32)
        tan_theta_axial = np.where(
            valid_tan_mask,
            axial_offset / safe_vec_norm,
            0.0,
        ).astype(np.float32)

        # Трассировка последнего предкристаллического сегмента сквозь входную и выходную плоскости коллиматора
        track_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), tracks.track_lengths)
        all_rel_coords = tracks.points.astype(np.float64) - collimator_centers[track_indices]
        all_depths = np.sum(all_rel_coords * axis_normal[track_indices], axis=1)

        depth_in = -0.5 * hole_length
        depth_out = 0.5 * hole_length

        before_exit_mask = (node_indices < first_crystal_indices[track_indices]) & (all_depths < depth_out)
        front_origin_indices = np.maximum.reduceat(
            np.where(before_exit_mask, node_indices, track_starts[track_indices]),
            track_starts,
        )
        past_collimator_indices = np.minimum(first_crystal_indices, front_origin_indices + 1)

        pre_crystal_points = tracks.points[front_origin_indices].astype(np.float64)
        post_collimator_points = tracks.points[past_collimator_indices].astype(np.float64)

        rel_pre = pre_crystal_points - collimator_centers
        rel_entry = post_collimator_points - collimator_centers

        depth_pre = np.sum(rel_pre * axis_normal, axis=1)
        coord_trans_pre = np.sum(rel_pre * axis_transverse, axis=1)
        coord_axial_pre = np.sum(rel_pre * axis_axial, axis=1)

        depth_entry = np.sum(rel_entry * axis_normal, axis=1)
        coord_trans_entry = np.sum(rel_entry * axis_transverse, axis=1)
        coord_axial_entry = np.sum(rel_entry * axis_axial, axis=1)

        delta_depth = depth_entry - depth_pre
        valid_ray = has_crystal & (np.abs(delta_depth) > MIN_VECTOR_NORM)
        safe_delta_depth = np.where(valid_ray, delta_depth, 1.0)

        param_in = (depth_in - depth_pre) / safe_delta_depth
        param_out = (depth_out - depth_pre) / safe_delta_depth

        coord_trans_in = np.where(valid_ray, coord_trans_pre + param_in * (coord_trans_entry - coord_trans_pre), 0.0)
        coord_axial_in = np.where(valid_ray, coord_axial_pre + param_in * (coord_axial_entry - coord_axial_pre), 0.0)

        coord_trans_out = np.where(valid_ray, coord_trans_pre + param_out * (coord_trans_entry - coord_trans_pre), 0.0)
        coord_axial_out = np.where(valid_ray, coord_axial_pre + param_out * (coord_axial_entry - coord_axial_pre), 0.0)

        if hole_shape == "square":
            in_hole_in, idx_first_in, idx_second_in = locate_square_channel_grid(
                coord_trans_in,
                coord_axial_in,
                hole_diameter,
                septa_thickness,
            )
            in_hole_out, idx_first_out, idx_second_out = locate_square_channel_grid(
                coord_trans_out,
                coord_axial_out,
                hole_diameter,
                septa_thickness,
            )
            channel_dist = np.abs(idx_first_out - idx_first_in) + np.abs(idx_second_out - idx_second_in)
        else:
            in_hole_in, idx_first_in, idx_second_in = locate_hexagonal_channel_axial(
                coord_trans_in,
                coord_axial_in,
                hole_diameter,
                septa_thickness,
            )
            in_hole_out, idx_first_out, idx_second_out = locate_hexagonal_channel_axial(
                coord_trans_out,
                coord_axial_out,
                hole_diameter,
                septa_thickness,
            )
            channel_dist = (
                np.abs(idx_first_out - idx_first_in)
                + np.abs(idx_second_out - idx_second_in)
                + np.abs((idx_first_out + idx_second_out) - (idx_first_in + idx_second_in))
            ) // 2

        same_channel = in_hole_in & in_hole_out & (channel_dist == 0)
        septa_count = np.where(same_channel, 0, np.maximum(channel_dist, 1))
        septa_count = np.where(valid_ray, septa_count, 0)

        tracks.cell_data["septal_crossings"] = np.clip(septa_count, 0, 255).astype(np.uint8)
        tracks.cell_data["tan_theta"] = tan_theta
        tracks.cell_data["tan_theta_axial"] = tan_theta_axial
