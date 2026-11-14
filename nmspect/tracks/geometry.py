"""Геометрические преобразования реперов детекторов и локализация каналов коллиматора."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from hepunits import deg, mm, rad
from numpy.typing import NDArray

MIN_VECTOR_NORM: Final[float] = 1e-6


def normalize_vector_3d(vector_arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Нормализует трёхмерный вектор до единичной длины."""
    vec_norm = float(np.linalg.norm(vector_arr))
    if vec_norm > MIN_VECTOR_NORM:
        return vector_arr / vec_norm
    return vector_arr.copy()


def resolve_metadata_gantry_angle(metadata: dict[str, Any] | None) -> float:
    """Извлекает базовый угол поворота гантри в радианах из словаря `metadata['gantry']`."""
    if not isinstance(metadata, dict) or "gantry" not in metadata:
        return float(0.0 * rad)

    gantry_meta = metadata["gantry"]
    if not isinstance(gantry_meta, dict) or "gantry_angle" not in gantry_meta:
        return float(0.0 * rad)

    raw_angle = float(np.asarray(gantry_meta["gantry_angle"]).item())
    units_label = str(gantry_meta.get("gantry_angle_units", "")).strip().lower()
    if units_label in ("deg", "degree", "degrees"):
        return float(raw_angle * deg)
    if units_label in ("rad", "radian", "radians"):
        return float(raw_angle * rad)
    if abs(raw_angle) > (2.0 * np.pi + 1e-2):
        return float(raw_angle * deg)
    return float(raw_angle * rad)


def rotate_vectors_around_z(
    vectors: NDArray[np.float64],
    delta_angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Векторно поворачивает массив 3D-векторов формы `(num_items, 3)` вокруг мировой оси Z на углы `delta_angles`."""
    if vectors.size == 0:
        return vectors.copy()
    if bool(np.all(np.abs(delta_angles) <= 1e-12)):
        return vectors.copy()

    cos_vals = np.cos(delta_angles)
    sin_vals = np.sin(delta_angles)
    vec_first = vectors[:, 0]
    vec_second = vectors[:, 1]
    vec_third = vectors[:, 2]

    rotated_first = cos_vals * vec_first - sin_vals * vec_second
    rotated_second = sin_vals * vec_first + cos_vals * vec_second
    return np.column_stack((rotated_first, rotated_second, vec_third))


def extract_orthonormal_frame_from_pose(
    pose_matrix: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Извлекает мировой центр и правый ортонормированный базис `(normal, transverse, axial)` из матрицы позы 4x4.

    Инварианты базиса:
    - `axis_normal` направлен наружу от изоцентра (`dot(axis_normal, center) >= 0`);
    - `axis_axial` выбирается по максимальной проекции на мировую ось Z и ориентирован вдоль `+Z`;
    - `axis_transverse` образует с `(axis_normal, axis_axial)` согласованную правую тройку
      (`dot(cross(axis_transverse, axis_normal), axis_axial) >= 0`).
    """
    mat_arr = np.asarray(pose_matrix, dtype=np.float64)
    if mat_arr.ndim != 2 or mat_arr.shape[0] < 3 or mat_arr.shape[1] < 3:
        raise ValueError(f"Матрица позы должна иметь размерность не менее (3, 3), получено: {mat_arr.shape}.")

    rot_matrix = mat_arr[:3, :3]
    if mat_arr.shape[1] >= 4:
        center_vec = np.asarray(mat_arr[:3, 3], dtype=np.float64).copy() * float(mm)
    else:
        center_vec = np.zeros(3, dtype=np.float64)

    center_norm = float(np.linalg.norm(center_vec))
    if center_norm > MIN_VECTOR_NORM:
        radial_projections = np.abs(rot_matrix.T @ center_vec)
        normal_col = int(np.argmax(radial_projections))
    else:
        normal_col = 2

    remaining_cols = [col_idx for col_idx in (0, 1, 2) if col_idx != normal_col]
    first_col, second_col = remaining_cols[0], remaining_cols[1]
    if abs(float(rot_matrix[2, second_col])) >= abs(float(rot_matrix[2, first_col])):
        trans_col = first_col
        axial_col = second_col
    else:
        trans_col = second_col
        axial_col = first_col

    axis_normal = normalize_vector_3d(rot_matrix[:, normal_col])
    axis_transverse = normalize_vector_3d(rot_matrix[:, trans_col])
    axis_axial = normalize_vector_3d(rot_matrix[:, axial_col])

    if center_norm > MIN_VECTOR_NORM and float(np.dot(axis_normal, center_vec)) < 0.0:
        axis_normal = -axis_normal

    if float(axis_axial[2]) < -MIN_VECTOR_NORM:
        axis_axial = -axis_axial

    cross_vec = np.cross(axis_transverse, axis_normal)
    if float(np.dot(cross_vec, axis_axial)) < 0.0:
        axis_transverse = -axis_transverse

    return center_vec, axis_normal, axis_transverse, axis_axial


def resolve_detector_frames(
    detector_ids: NDArray[np.integer[Any]],
    gantry_angles: NDArray[np.floating[Any]],
    metadata: dict[str, Any] | None,
    default_orbit_radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Определяет для каждого трека `(center, axis_normal, axis_transverse, axis_axial)` кристалла детектора.

    При наличии `metadata['volume_matrices']` извлекает точный базовый репер головки и поворачивает его
    на разность `gantry_angles - meta_gantry_angle` вокруг оси гантри Z (корректно поддерживая многоракурсные пакеты).
    """
    det_ids_arr = np.asarray(detector_ids)
    angles_arr = np.asarray(gantry_angles, dtype=np.float64)
    num_tracks = int(det_ids_arr.size)

    resolved_radius = float(default_orbit_radius)
    if isinstance(metadata, dict):
        gantry_meta = metadata.get("gantry")
        if isinstance(gantry_meta, dict) and "orbit_radius" in gantry_meta:
            raw_radius = float(np.asarray(gantry_meta["orbit_radius"]).item()) * float(mm)
            if raw_radius > 0.0:
                resolved_radius = raw_radius

    detector_angles = angles_arr + (det_ids_arr.astype(np.float64) - 1.0) * np.pi
    axis_normal = np.column_stack(
        (
            -np.sin(detector_angles),
            np.cos(detector_angles),
            np.zeros(num_tracks, dtype=np.float64),
        )
    )
    detector_centers = axis_normal * resolved_radius
    axis_transverse = np.column_stack(
        (
            np.cos(detector_angles),
            np.sin(detector_angles),
            np.zeros(num_tracks, dtype=np.float64),
        )
    )
    axis_axial = np.zeros((num_tracks, 3), dtype=np.float64)
    axis_axial[:, 2] = 1.0

    if not isinstance(metadata, dict) or "volume_matrices" not in metadata:
        return detector_centers, axis_normal, axis_transverse, axis_axial

    volume_matrices = metadata["volume_matrices"]
    if not isinstance(volume_matrices, dict):
        return detector_centers, axis_normal, axis_transverse, axis_axial

    meta_gantry_angle = resolve_metadata_gantry_angle(metadata)
    unique_detectors = np.unique(det_ids_arr)

    for raw_det_id in unique_detectors:
        det_id = int(raw_det_id)
        detector_key = f"Detector_{det_id}"
        if detector_key not in volume_matrices:
            continue

        pose_matrix = np.asarray(volume_matrices[detector_key], dtype=np.float64)
        if pose_matrix.ndim != 2 or pose_matrix.shape[0] < 3 or pose_matrix.shape[1] < 3:
            continue

        base_center, base_normal, base_trans, base_axial = extract_orthonormal_frame_from_pose(pose_matrix)
        det_mask = det_ids_arr == raw_det_id
        mask_count = int(np.count_nonzero(det_mask))
        delta_angles = angles_arr[det_mask] - meta_gantry_angle

        tiled_center = np.broadcast_to(base_center, (mask_count, 3))
        tiled_normal = np.broadcast_to(base_normal, (mask_count, 3))
        tiled_trans = np.broadcast_to(base_trans, (mask_count, 3))
        tiled_axial = np.broadcast_to(base_axial, (mask_count, 3))

        detector_centers[det_mask] = rotate_vectors_around_z(tiled_center, delta_angles)
        axis_normal[det_mask] = rotate_vectors_around_z(tiled_normal, delta_angles)
        axis_transverse[det_mask] = rotate_vectors_around_z(tiled_trans, delta_angles)
        axis_axial[det_mask] = rotate_vectors_around_z(tiled_axial, delta_angles)

    return detector_centers, axis_normal, axis_transverse, axis_axial


def locate_hexagonal_channel_axial(
    coord_transverse: NDArray[np.floating[Any]],
    coord_axial: NDArray[np.floating[Any]],
    hole_diameter: float,
    septa_thickness: float,
    margin_epsilon: float = 1e-7 * mm,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    """Векторно определяет принадлежность точек отверстию и аксиальные индексы `(axial_first, axial_second)`.

    Использует точное преобразование в аксиальные гексагональные координаты с кубическим округлением
    (`axial_first + axial_second + axial_third == 0`) и проверяет нахождение внутри шестиугольного канала.
    """
    if hole_diameter <= 0.0:
        raise ValueError(f"Диаметр отверстия коллиматора должен быть строго положительным: {hole_diameter}.")
    if septa_thickness <= 0.0:
        raise ValueError(f"Толщина септы коллиматора должна быть строго положительной: {septa_thickness}.")

    arr_trans = np.asarray(coord_transverse, dtype=np.float64)
    arr_axial = np.asarray(coord_axial, dtype=np.float64)

    sqrt_three = float(np.sqrt(3.0))
    pitch = float(hole_diameter + septa_thickness)
    hex_radius = pitch / sqrt_three

    frac_first = ((sqrt_three / 3.0) * arr_trans - (1.0 / 3.0) * arr_axial) / hex_radius
    frac_second = ((2.0 / 3.0) * arr_axial) / hex_radius
    frac_third = -frac_first - frac_second

    round_first = np.round(frac_first)
    round_second = np.round(frac_second)
    round_third = np.round(frac_third)

    diff_first = np.abs(round_first - frac_first)
    diff_second = np.abs(round_second - frac_second)
    diff_third = np.abs(round_third - frac_third)

    reset_first_mask = (diff_first > diff_second) & (diff_first > diff_third)
    reset_second_mask = (~reset_first_mask) & (diff_second > diff_third)

    axial_first = np.where(reset_first_mask, -round_second - round_third, round_first).astype(np.int64)
    axial_second = np.where(reset_second_mask, -axial_first - round_third, round_second).astype(np.int64)

    center_trans = pitch * (axial_first.astype(np.float64) + 0.5 * axial_second.astype(np.float64))
    center_axial = (0.5 * sqrt_three * pitch) * axial_second.astype(np.float64)
    abs_delta_trans = np.abs(arr_trans - center_trans)
    abs_delta_axial = np.abs(arr_axial - center_axial)

    eps_val = float(margin_epsilon)
    half_width = 0.5 * float(hole_diameter) + eps_val
    side_limit = float(hole_diameter) + 2.0 * eps_val
    is_in_hole = (abs_delta_trans <= half_width) & ((abs_delta_trans + sqrt_three * abs_delta_axial) <= side_limit)

    return np.asarray(is_in_hole, dtype=np.bool_), axial_first, axial_second


def locate_square_channel_grid(
    coord_transverse: NDArray[np.floating[Any]],
    coord_axial: NDArray[np.floating[Any]],
    hole_diameter: float,
    septa_thickness: float,
    margin_epsilon: float = 1e-7 * mm,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    """Векторно определяет попадание точек в отверстие квадратного коллиматора и индексы ячейки `(idx_trans, idx_axial)`."""
    if hole_diameter <= 0.0:
        raise ValueError(f"Ширина отверстия коллиматора должна быть строго положительной: {hole_diameter}.")
    if septa_thickness <= 0.0:
        raise ValueError(f"Толщина септы коллиматора должна быть строго положительной: {septa_thickness}.")

    arr_trans = np.asarray(coord_transverse, dtype=np.float64)
    arr_axial = np.asarray(coord_axial, dtype=np.float64)
    period = float(hole_diameter + septa_thickness)
    half_width = 0.5 * float(hole_diameter) + float(margin_epsilon)

    idx_trans = np.round(arr_trans / period).astype(np.int64)
    idx_axial = np.round(arr_axial / period).astype(np.int64)
    delta_trans = np.abs(arr_trans - idx_trans.astype(np.float64) * period)
    delta_axial = np.abs(arr_axial - idx_axial.astype(np.float64) * period)
    is_in_hole = (delta_trans <= half_width) & (delta_axial <= half_width)

    return np.asarray(is_in_hole, dtype=np.bool_), idx_trans, idx_axial
