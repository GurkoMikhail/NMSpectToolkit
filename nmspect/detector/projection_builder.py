"""Дискретизация зарегистрированных вспышек в планарные 2D-проекции с коррекцией рассеяния DEW/TEW."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nmspect.spectrometry.models import EnergyWindowConfig


def _accumulate_window_projection(
    energies_arr: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    y_arr: NDArray[np.float64],
    edges_x: NDArray[np.float64],
    edges_y: NDArray[np.float64],
    window_config: EnergyWindowConfig,
    clip_negative: bool,
) -> NDArray[np.float64]:
    """Внутренняя функция построения 2D-проекции по предвычисленным сеткам бинов `edges_x`, `edges_y`."""
    num_bins_x = edges_x.size - 1
    num_bins_y = edges_y.size - 1
    projection = np.zeros((num_bins_x, num_bins_y), dtype=np.float64)
    if energies_arr.size == 0:
        return projection

    main_weight = (
        float(window_config.scatter_weight)
        if window_config.is_scatter_window and window_config.scatter_weight != 0.0
        else 1.0
    )
    main_mask = (energies_arr >= window_config.lower_bound) & (energies_arr <= window_config.upper_bound)
    if np.any(main_mask):
        main_hist, _, _ = np.histogram2d(
            x_arr[main_mask],
            y_arr[main_mask],
            bins=(edges_x, edges_y),
        )
        projection += main_weight * main_hist

    for assoc_win in window_config.associated_windows:
        low_cond = (
            energies_arr > assoc_win.lower_bound
            if np.isclose(assoc_win.lower_bound, window_config.upper_bound, rtol=0.0, atol=1e-12)
            else energies_arr >= assoc_win.lower_bound
        )
        high_cond = (
            energies_arr < assoc_win.upper_bound
            if np.isclose(assoc_win.upper_bound, window_config.lower_bound, rtol=0.0, atol=1e-12)
            else energies_arr <= assoc_win.upper_bound
        )
        scatter_mask = low_cond & high_cond
        if np.any(scatter_mask):
            scatter_hist, _, _ = np.histogram2d(
                x_arr[scatter_mask],
                y_arr[scatter_mask],
                bins=(edges_x, edges_y),
            )
            projection += float(assoc_win.scatter_weight) * scatter_hist

    if clip_negative and not (window_config.is_scatter_window and main_weight < 0.0):
        np.clip(projection, 0.0, None, out=projection)

    return projection


def build_projection_for_energy_window(
    energies: NDArray[np.floating],
    coords_x: NDArray[np.floating],
    coords_y: NDArray[np.floating],
    detector_size: Sequence[float],
    pixel_size: float,
    window_config: EnergyWindowConfig,
    clip_negative: bool = True,
) -> NDArray[np.float64]:
    """Строит двумерную проекционную матрицу для заданного энергетического окна с учётом DEW/TEW.

    Args:
        energies: Вектор зарегистрированных энергий вспышек (размерность M, в hepunits).
        coords_x: Вектор локальных координат X вспышек на кристалле (размерность M, в hepunits).
        coords_y: Вектор локальных координат Y вспышек на кристалле (размерность M, в hepunits).
        detector_size: Габариты кристалла детектора `(size_x, size_y, ...)` в базовых единицах hepunits.
        pixel_size: Размер пикселя проекционной матрицы в базовых единицах hepunits.
        window_config: Конфигурация энергетического окна (включая `associated_windows` для DEW/TEW).
        clip_negative: Обнулять ли отрицательные значения в пикселях после вычитания рассеяния.

    Returns:
        Двумерный массив проекции формы `(num_bins_x, num_bins_y)` типа `float64`.
    """
    if pixel_size <= 0.0:
        raise ValueError(f"Размер пикселя должен быть строго положительным, получено: {pixel_size}")
    if len(detector_size) < 2 or detector_size[0] <= 0.0 or detector_size[1] <= 0.0:
        raise ValueError(f"Некорректные габариты детектора detector_size: {detector_size}")

    energies_arr = np.asarray(energies, dtype=np.float64).ravel()
    x_arr = np.asarray(coords_x, dtype=np.float64).ravel()
    y_arr = np.asarray(coords_y, dtype=np.float64).ravel()

    if not (energies_arr.size == x_arr.size == y_arr.size):
        raise ValueError(
            f"Размерности массивов energies ({energies_arr.size}), coords_x ({x_arr.size}) "
            f"и coords_y ({y_arr.size}) должны совпадать."
        )

    det_width = float(detector_size[0])
    det_height = float(detector_size[1])
    num_bins_x = max(1, round(det_width / float(pixel_size)))
    num_bins_y = max(1, round(det_height / float(pixel_size)))

    half_width = 0.5 * det_width
    half_height = 0.5 * det_height
    edges_x = np.linspace(-half_width, half_width, num_bins_x + 1, dtype=np.float64)
    edges_y = np.linspace(-half_height, half_height, num_bins_y + 1, dtype=np.float64)

    return _accumulate_window_projection(
        energies_arr=energies_arr,
        x_arr=x_arr,
        y_arr=y_arr,
        edges_x=edges_x,
        edges_y=edges_y,
        window_config=window_config,
        clip_negative=clip_negative,
    )


def generate_projections_from_events(
    events_list: Sequence[NDArray[np.floating]],
    detector_size: Sequence[float],
    pixel_size: float,
    energy_windows: Sequence[EnergyWindowConfig],
    clip_negative: bool = True,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Формирует 3D-стек проекций из списка матриц зарегистрированных вспышек для набора энергетических окон.

    Args:
        events_list: Последовательность матриц вспышек `(M_i, 5)` для каждого углового ракурса.
        detector_size: Физические размеры кристалла детектора `(size_x, size_y, ...)` в hepunits.
        pixel_size: Размер пикселя проекции в hepunits.
        energy_windows: Последовательность конфигураций энергетических окон `EnergyWindowConfig`.
        clip_negative: Обнулять ли отрицательные отсчёты после вычитания рассеяния.

    Returns:
        Кортеж `(combined_projections, per_window_projections)`, где `combined_projections` —
        суммарный 3D-стек формы `(num_views, num_bins_x, num_bins_y)`, а `per_window_projections` —
        словарь стеков по имени каждого энергетического окна.
    """
    if not energy_windows:
        raise ValueError("Список энергетических окон energy_windows не может быть пустым.")
    if pixel_size <= 0.0:
        raise ValueError(f"Размер пикселя должен быть строго положительным, получено: {pixel_size}")
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

    per_window_projections: dict[str, NDArray[np.float64]] = {}
    for win_cfg in energy_windows:
        per_window_projections[win_cfg.name] = np.zeros((num_views, num_bins_x, num_bins_y), dtype=np.float64)

    for view_idx, events_matrix in enumerate(events_list):
        events_arr = np.asarray(events_matrix, dtype=np.float64)
        if events_arr.size == 0:
            energies = np.empty(0, dtype=np.float64)
            coords_x = np.empty(0, dtype=np.float64)
            coords_y = np.empty(0, dtype=np.float64)
        else:
            if events_arr.ndim != 2 or events_arr.shape[1] < 5:
                raise ValueError(
                    f"Элемент events_list[{view_idx}] должен иметь форму (M, 5), получено: {events_arr.shape}"
                )
            energies = np.ascontiguousarray(events_arr[:, 2])
            coords_x = np.ascontiguousarray(events_arr[:, 3])
            coords_y = np.ascontiguousarray(events_arr[:, 4])

        for win_cfg in energy_windows:
            frame_proj = _accumulate_window_projection(
                energies_arr=energies,
                x_arr=coords_x,
                y_arr=coords_y,
                edges_x=edges_x,
                edges_y=edges_y,
                window_config=win_cfg,
                clip_negative=clip_negative,
            )
            per_window_projections[win_cfg.name][view_idx] = frame_proj

    combined_projections = np.zeros((num_views, num_bins_x, num_bins_y), dtype=np.float64)
    for win_stack in per_window_projections.values():
        combined_projections += win_stack

    if clip_negative:
        np.clip(combined_projections, 0.0, None, out=combined_projections)

    return combined_projections, per_window_projections


class ProjectionBuilder:
    """Высокоуровневый построитель планарных 2D-проекций ОФЭКТ из сцинтилляционных вспышек."""

    def __init__(
        self,
        pixel_size: float,
        energy_windows: Sequence[EnergyWindowConfig],
        detector_size: Sequence[float] | None = None,
        clip_negative: bool = True,
    ) -> None:
        """Инициализирует построитель проекций.

        Args:
            pixel_size: Физический размер пикселя в базовых единицах hepunits (mm).
            energy_windows: Набор энергетических окон для формирования проекций.
            detector_size: Опциональные габариты детектора `(size_x, size_y, ...)` по умолчанию.
            clip_negative: Обнулять ли отрицательные пиксели при коррекции рассеяния DEW/TEW.
        """
        if pixel_size <= 0.0:
            raise ValueError(f"Размер пикселя должен быть строго положительным: {pixel_size}")
        if not energy_windows:
            raise ValueError("Список энергетических окон energy_windows не может быть пустым.")

        self.pixel_size: float = float(pixel_size)
        self.energy_windows: tuple[EnergyWindowConfig, ...] = tuple(energy_windows)
        self.detector_size: tuple[float, ...] | None = (
            tuple(float(dim) for dim in detector_size) if detector_size is not None else None
        )
        self.clip_negative: bool = bool(clip_negative)

    def build_projections(
        self,
        events_list: Sequence[NDArray[np.floating]],
        detector_size: Sequence[float] | None = None,
        clip_negative: bool | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        """Строит суммарный и пооконные 3D-стеки проекций по списку матриц вспышек.

        Args:
            events_list: Список матриц вспышек `(M_i, 5)` по угловым ракурсам.
            detector_size: Габариты детектора `(size_x, size_y, ...)` (если не заданы в конструкторе).
            clip_negative: Переопределение флага обнуления отрицательных значений.

        Returns:
            Кортеж `(combined_projections, per_window_projections)`.
        """
        target_det_size = detector_size if detector_size is not None else self.detector_size
        if target_det_size is None:
            raise ValueError("Необходимо указать габариты детектора detector_size.")
        use_clip = self.clip_negative if clip_negative is None else bool(clip_negative)

        return generate_projections_from_events(
            events_list=events_list,
            detector_size=target_det_size,
            pixel_size=self.pixel_size,
            energy_windows=self.energy_windows,
            clip_negative=use_clip,
        )
