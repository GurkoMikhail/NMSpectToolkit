"""Моделирование физических процессов в сцинтилляционном кристалле и ФЭУ."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from hepunits import keV
from numpy.typing import NDArray

FWHM_TO_SIGMA_FACTOR: Final[float] = 1.0 / float(2.0 * np.sqrt(2.0 * np.log(2.0)))


def normalize_energy_resolution(energy_resolution: float) -> float:
    """Проверяет и нормализует относительное энергетическое разрешение FWHM к долям единицы.

    Если значение передано в процентах (`> 1.0`, например `9.9`), автоматически делит его на `100.0`.
    """
    if energy_resolution < 0.0:
        raise ValueError(f"Энергетическое разрешение не может быть отрицательным: {energy_resolution}")
    resolved_value = float(energy_resolution)
    return resolved_value / 100.0 if resolved_value > 1.0 else resolved_value


def aggregate_energy_weighted_segments(
    segment_starts: NDArray[np.integer[Any]],
    energy_deposits: NDArray[np.floating[Any]],
    coordinates: NDArray[np.floating[Any]],
    times: NDArray[np.floating[Any]],
    *,
    fallback_coordinates: NDArray[np.floating[Any]] | None = None,
    fallback_times: NDArray[np.floating[Any]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Векторно агрегирует энерговыделение, центроид координат и время по CSR-сегментам через `np.add.reduceat`.

    Args:
        segment_starts: Возрастающий массив начальных индексов сегментов формы `(K,)`.
        energy_deposits: Вектор энерговыделений в узлах/актах формы `(N,)` в `hepunits` (`MeV`).
        coordinates: Матрица координат узлов/актов формы `(N, D)` в `hepunits` (`mm`), где `D >= 1`.
        times: Вектор моментов времени узлов/актов формы `(N,)` в `hepunits` (`ns`).
        fallback_coordinates: Опциональные резервные координаты формы `(K, D)` для сегментов с нулевой энергией.
        fallback_times: Опциональные резервные времена формы `(K,)` для сегментов с нулевой энергией.

    Returns:
        Кортеж `(summed_energies, positive_energy_mask, weighted_centroids, weighted_times)`.
    """
    starts_arr = np.asarray(segment_starts, dtype=np.intp).ravel()
    edep_arr = np.asarray(energy_deposits, dtype=np.float64).ravel()
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64).ravel()

    if coords_arr.ndim != 2 or coords_arr.shape[1] < 1:
        raise ValueError(f"Массив coordinates должен иметь двумерную форму (N, D), получено: {coords_arr.shape}")

    num_dims = coords_arr.shape[1]
    num_segments = starts_arr.size
    if num_segments == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.bool_),
            np.empty((0, num_dims), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    num_nodes = edep_arr.size
    if num_nodes == 0 or coords_arr.shape[0] != num_nodes or times_arr.size != num_nodes:
        raise ValueError(
            f"Несогласованные размеры узловых массивов: energy_deposits={num_nodes}, "
            f"coordinates={coords_arr.shape}, times={times_arr.size}."
        )

    clean_edep = np.maximum(edep_arr, 0.0)
    summed_energies = np.add.reduceat(clean_edep, starts_arr)
    positive_mask: NDArray[np.bool_] = summed_energies > 0.0
    safe_denom = np.where(positive_mask, summed_energies, 1.0)

    sum_weighted_coords = np.add.reduceat(coords_arr * clean_edep[:, np.newaxis], starts_arr, axis=0)
    sum_weighted_times = np.add.reduceat(times_arr * clean_edep, starts_arr)

    if fallback_coordinates is not None:
        resolved_fallback_coords = np.asarray(fallback_coordinates, dtype=np.float64)
        if resolved_fallback_coords.shape != (num_segments, num_dims):
            raise ValueError(
                f"Массив fallback_coordinates должен иметь форму {(num_segments, num_dims)}, "
                f"получено: {resolved_fallback_coords.shape}."
            )
    else:
        resolved_fallback_coords = coords_arr[starts_arr]

    if fallback_times is not None:
        resolved_fallback_times = np.asarray(fallback_times, dtype=np.float64).ravel()
        if resolved_fallback_times.size != num_segments:
            raise ValueError(
                f"Массив fallback_times должен иметь длину {num_segments}, получено: {resolved_fallback_times.size}."
            )
    else:
        resolved_fallback_times = times_arr[starts_arr]

    weighted_centroids = np.where(
        positive_mask[:, np.newaxis],
        sum_weighted_coords / safe_denom[:, np.newaxis],
        resolved_fallback_coords,
    )
    weighted_times = np.where(
        positive_mask,
        sum_weighted_times / safe_denom,
        resolved_fallback_times,
    )
    return summed_energies, positive_mask, weighted_centroids, weighted_times


def smear_scintillation_energy(
    deposited_energies: NDArray[np.floating[Any]],
    energy_resolution: float,
    reference_energy: float,
    rng: np.random.Generator,
    *,
    active_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    r"""Моделирует гауссово энергетическое размытие ФЭУ по закону $\sigma_E \propto \sqrt{E_{\text{ref}} E}$.

    Args:
        deposited_energies: Вектор истинных энерговыделений вспышек формы `(K,)` в `hepunits` (`MeV`).
        energy_resolution: Относительное разрешение FWHM на референсной энергии (в долях или процентах).
        reference_energy: Референсная энергия калибровки в `hepunits` (`MeV`).
        rng: Генератор псевдослучайных чисел `np.random.Generator`.
        active_mask: Опциональная булева маска активных вспышек формы `(K,)`; неактивные элементы получают `0.0`.

    Returns:
        Вектор зарегистрированных энергий формы `(K,)` типа `float64` (неотрицательный).
    """
    if reference_energy <= 0.0:
        raise ValueError(f"Референсная энергия должна быть строго положительной: {reference_energy}")

    relative_fwhm_ref = normalize_energy_resolution(energy_resolution)
    energies_arr = np.clip(np.asarray(deposited_energies, dtype=np.float64).ravel(), 0.0, None)
    if energies_arr.size == 0:
        return np.empty(0, dtype=np.float64)

    if active_mask is not None:
        mask_arr = np.asarray(active_mask, dtype=np.bool_).ravel()
        if mask_arr.size != energies_arr.size:
            raise ValueError(
                f"Длина active_mask ({mask_arr.size}) не совпадает с числом вспышек ({energies_arr.size})."
            )
    else:
        mask_arr = None

    if relative_fwhm_ref == 0.0:
        return np.where(mask_arr, energies_arr, 0.0) if mask_arr is not None else energies_arr.copy()

    sigma_values = relative_fwhm_ref * FWHM_TO_SIGMA_FACTOR * np.sqrt(float(reference_energy) * energies_arr)
    smeared_values = np.clip(rng.normal(loc=energies_arr, scale=sigma_values), 0.0, None)
    if mask_arr is not None:
        return np.where(mask_arr, smeared_values, 0.0)
    return smeared_values


def compute_spatial_offsets_2d(
    num_events: int,
    intrinsic_spatial_resolution: float,
    rng: np.random.Generator,
    *,
    deposited_energies: NDArray[np.floating[Any]] | None = None,
    reference_energy: float = 140.5 * keV,
    scale_with_energy: bool = False,
    active_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Генерирует двумерные гауссовы смещения позиционирования ФЭУ `(Delta_u, Delta_v)` в плоскости кристалла.

    Args:
        num_events: Число моделируемых событий `K`.
        intrinsic_spatial_resolution: Собственное пространственное разрешение FWHM в `hepunits` (`mm`).
        rng: Генератор псевдослучайных чисел `np.random.Generator`.
        deposited_energies: Вектор энерговыделений формы `(K,)` (обязателен при `scale_with_energy=True`).
        reference_energy: Референсная энергия калибровки в `hepunits` (`MeV`).
        scale_with_energy: Масштабировать ли ширину отклика ФЭУ как `sqrt(E_ref / E_dep)`.
        active_mask: Булева маска активных вспышек `(K,)`; для неактивных элементов смещение равно `(0.0, 0.0)`.

    Returns:
        Матрица двумерных пространственных смещений формы `(K, 2)` типа `float64`.
    """
    if num_events < 0:
        raise ValueError(f"Число событий num_events не может быть отрицательным: {num_events}")
    if intrinsic_spatial_resolution < 0.0:
        raise ValueError(
            f"Собственное пространственное разрешение не может быть отрицательным: {intrinsic_spatial_resolution}"
        )
    if reference_energy <= 0.0:
        raise ValueError(f"Референсная энергия должна быть строго положительной: {reference_energy}")

    if num_events == 0 or intrinsic_spatial_resolution == 0.0:
        return np.zeros((num_events, 2), dtype=np.float64)

    if active_mask is not None:
        mask_arr = np.asarray(active_mask, dtype=np.bool_).ravel()
        if mask_arr.size != num_events:
            raise ValueError(f"Длина active_mask ({mask_arr.size}) не совпадает с num_events ({num_events}).")
    else:
        mask_arr = None

    base_sigma = float(intrinsic_spatial_resolution) * FWHM_TO_SIGMA_FACTOR
    if scale_with_energy:
        if deposited_energies is None:
            raise ValueError("При scale_with_energy=True необходимо передать массив deposited_energies.")
        edep_arr = np.asarray(deposited_energies, dtype=np.float64).ravel()
        if edep_arr.size != num_events:
            raise ValueError(
                f"Длина deposited_energies ({edep_arr.size}) не совпадает с num_events ({num_events})."
            )
        safe_edep = np.maximum(edep_arr, 1e-6 * float(keV))
        energy_scale = np.sqrt(float(reference_energy) / safe_edep)
        sigma_vec = np.where(mask_arr, base_sigma * energy_scale, 0.0) if mask_arr is not None else base_sigma * energy_scale
    elif mask_arr is not None:
        sigma_vec = np.where(mask_arr, base_sigma, 0.0)
    else:
        return rng.normal(loc=0.0, scale=base_sigma, size=(num_events, 2))

    unit_noise = rng.normal(loc=0.0, scale=1.0, size=(num_events, 2))
    return unit_noise * sigma_vec[:, np.newaxis]


class ScintillationProcessor:
    """Агрегирует микроскопические акты взаимодействия ($N$) в сцинтилляционные вспышки ($M$) и моделирует размытие."""

    def __init__(
        self,
        time_resolution: float,
        energy_resolution: float,
        intrinsic_spatial_resolution: float,
        reference_energy: float | None = None,
        rng: np.random.Generator | int | None = None,
        scale_spatial_with_energy: bool = False,
    ) -> None:
        """Инициализирует параметры аппаратного отклика сцинтилляционного детектора.

        Args:
            time_resolution: Временное разрешение (окно слияния импульсов / время высвечивания) в hepunits (ns).
            energy_resolution: Относительное энергетическое разрешение FWHM на референсной энергии (в долях или %).
            intrinsic_spatial_resolution: Собственное пространственное разрешение детектора FWHM в hepunits (mm).
            reference_energy: Референсная энергия калибровки разрешения в hepunits (по умолчанию 140.5 кэВ).
            rng: Генератор псевдослучайных чисел `np.random.Generator` или целочисленное зерно (`seed`).
            scale_spatial_with_energy: Масштабировать ли пространственное размытие ФЭУ как `sqrt(E_ref / E_dep)`.
        """
        if time_resolution < 0.0:
            raise ValueError(f"Временное разрешение не может быть отрицательным: {time_resolution}")
        if energy_resolution < 0.0:
            raise ValueError(f"Энергетическое разрешение не может быть отрицательным: {energy_resolution}")
        if intrinsic_spatial_resolution < 0.0:
            raise ValueError(
                f"Собственное пространственное разрешение не может быть отрицательным: {intrinsic_spatial_resolution}"
            )

        resolved_ref_energy = 140.5 * keV if reference_energy is None else float(reference_energy)
        if resolved_ref_energy <= 0.0:
            raise ValueError(f"Референсная энергия должна быть строго положительной: {resolved_ref_energy}")

        self.time_resolution: float = float(time_resolution)
        self.energy_resolution: float = float(energy_resolution)
        self.intrinsic_spatial_resolution: float = float(intrinsic_spatial_resolution)
        self.reference_energy: float = resolved_ref_energy
        self.scale_spatial_with_energy: bool = bool(scale_spatial_with_energy)
        self.rng: np.random.Generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )

    def unite_acts(
        self,
        raw_acts: NDArray[np.floating[Any]],
        scattering_orders: NDArray[np.integer[Any]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Векторно объединяет микроскопические акты взаимодействия ($N$) в сцинтилляционные вспышки ($M$).

        Использует `aggregate_energy_weighted_segments` и `np.maximum.reduceat` для расчёта суммарного
        энерговыделения, средневзвешенных по энергии координат центра тяжести светового пятна и кратности рассеяния.

        Args:
            raw_acts: Матрица сырых актов взаимодействия формы `(N, 5)`:
                `[particle_id, time, energy_deposit, local_x, local_y]`.
            scattering_orders: Опциональный массив кратностей рассеяния в фантоме длины `N`.

        Returns:
            Кортеж `(events, united_scattering_orders)`, где `events` имеет форму `(M, 5)`,
            а `united_scattering_orders` — форму `(M,)`.
        """
        acts_arr = np.asarray(raw_acts, dtype=np.float64)
        if acts_arr.size == 0:
            return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.int32)
        if acts_arr.ndim != 2 or acts_arr.shape[1] < 5:
            raise ValueError(f"Матрица raw_acts должна иметь форму (N, 5), получено: {acts_arr.shape}")

        num_acts = acts_arr.shape[0]
        if scattering_orders is not None:
            orders_arr = np.asarray(scattering_orders, dtype=np.int32).ravel()
            if orders_arr.size != num_acts:
                raise ValueError(
                    f"Несовпадение длины scattering_orders ({orders_arr.size}) и числа актов ({num_acts})."
                )
        else:
            orders_arr = np.zeros(num_acts, dtype=np.int32)

        # Отбрасывание микроскопических актов без энерговыделения (например, упругого когерентного рассеяния в кристалле)
        positive_energy_mask = acts_arr[:, 2] > 0.0
        if not np.any(positive_energy_mask):
            return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.int32)

        if np.all(positive_energy_mask):
            filtered_acts = acts_arr
            filtered_orders = orders_arr
        else:
            filtered_acts = acts_arr[positive_energy_mask]
            filtered_orders = orders_arr[positive_energy_mask]

        # Лексикографическая сортировка по (particle_id, time) с быстрой проверкой упорядоченности O(N)
        pids_col = filtered_acts[:, 0]
        times_col = filtered_acts[:, 1]
        is_already_sorted = bool(
            filtered_acts.shape[0] <= 1
            or np.all(
                (pids_col[1:] > pids_col[:-1])
                | ((pids_col[1:] == pids_col[:-1]) & (times_col[1:] >= times_col[:-1]))
            )
        )
        if is_already_sorted:
            sorted_acts = filtered_acts
            sorted_orders = filtered_orders
        else:
            sort_indices = np.lexsort((times_col, pids_col))
            sorted_acts = filtered_acts[sort_indices]
            sorted_orders = filtered_orders[sort_indices]

        pid_changed = sorted_acts[1:, 0] != sorted_acts[:-1, 0]
        if self.time_resolution > 0.0:
            time_gap_exceeded = (sorted_acts[1:, 1] - sorted_acts[:-1, 1]) > self.time_resolution
            new_group_mask = pid_changed | time_gap_exceeded
        else:
            new_group_mask = pid_changed

        group_starts = np.concatenate((np.array([0], dtype=np.intp), np.flatnonzero(new_group_mask) + 1))

        summed_energies, valid_mask, centroids_xy, mean_times = aggregate_energy_weighted_segments(
            segment_starts=group_starts,
            energy_deposits=sorted_acts[:, 2],
            coordinates=sorted_acts[:, 3:5],
            times=sorted_acts[:, 1],
        )
        if not np.any(valid_mask):
            return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.int32)

        united_orders = np.maximum.reduceat(sorted_orders, group_starts).astype(np.int32)
        event_pids = sorted_acts[group_starts[valid_mask], 0]
        events_matrix = np.column_stack(
            (
                event_pids,
                mean_times[valid_mask],
                summed_energies[valid_mask],
                centroids_xy[valid_mask, 0],
                centroids_xy[valid_mask, 1],
            )
        )
        return events_matrix, united_orders[valid_mask]

    def _smear_energy_inplace(
        self,
        events_matrix: NDArray[np.float64],
        reference_energies: NDArray[np.float64] | None = None,
    ) -> None:
        """Модифицирует столбец энергии (индекс 2) матрицы вспышек `(M, 5)` на месте."""
        if events_matrix.size == 0 or self.energy_resolution == 0.0:
            return
        if events_matrix.ndim != 2 or events_matrix.shape[1] < 5:
            raise ValueError(f"Матрица events должна иметь форму (M, 5), получено: {events_matrix.shape}")

        source_energies = events_matrix[:, 2] if reference_energies is None else reference_energies
        events_matrix[:, 2] = smear_scintillation_energy(
            deposited_energies=source_energies,
            energy_resolution=self.energy_resolution,
            reference_energy=self.reference_energy,
            rng=self.rng,
        )

    def _smear_spatial_inplace(
        self,
        events_matrix: NDArray[np.float64],
        deposited_energies: NDArray[np.float64] | None = None,
    ) -> None:
        """Модифицирует столбцы локальных координат (индексы 3 и 4) матрицы вспышек `(M, 5)` на месте."""
        if events_matrix.size == 0 or self.intrinsic_spatial_resolution == 0.0:
            return
        if events_matrix.ndim != 2 or events_matrix.shape[1] < 5:
            raise ValueError(f"Матрица events должна иметь форму (M, 5), получено: {events_matrix.shape}")

        source_edep = events_matrix[:, 2] if deposited_energies is None else deposited_energies
        offsets_xy = compute_spatial_offsets_2d(
            num_events=events_matrix.shape[0],
            intrinsic_spatial_resolution=self.intrinsic_spatial_resolution,
            rng=self.rng,
            deposited_energies=source_edep,
            reference_energy=self.reference_energy,
            scale_with_energy=self.scale_spatial_with_energy,
        )
        events_matrix[:, 3:5] += offsets_xy

    def apply_energy_resolution(self, events: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Накладывает стохастическое гауссово размытие энергии сцинтилляционных вспышек ($\sigma_E \propto \sqrt{E}$).

        Args:
            events: Матрица вспышек формы `(M, 5)`.

        Returns:
            Копия матрицы вспышек `(M, 5)` с размытыми значениями энергии в столбце 2.
        """
        result = np.array(events, dtype=np.float64, copy=True)
        self._smear_energy_inplace(result)
        return result

    def apply_spatial_resolution(self, events: NDArray[np.float64]) -> NDArray[np.float64]:
        """Накладывает аппаратное гауссово размытие локальных координат $(x, y)$ вспышек.

        Args:
            events: Матрица вспышек формы `(M, 5)`.

        Returns:
            Копия матрицы вспышек `(M, 5)` с размытыми координатами в столбцах 3 и 4.
        """
        result = np.array(events, dtype=np.float64, copy=True)
        self._smear_spatial_inplace(result)
        return result

    def process_detector_acts(
        self,
        raw_acts: NDArray[np.floating[Any]],
        scattering_orders: NDArray[np.integer[Any]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Выполняет полный цикл преобразования сырых актов ($N$) в зарегистрированные вспышки ($M$).

        Включает временное объединение актов (`unite_acts`), энергетическое размытие ФЭУ
        (`apply_energy_resolution`) и пространственное размытие позиционирования (`apply_spatial_resolution`).

        Args:
            raw_acts: Матрица сырых актов формы `(N, 5)`.
            scattering_orders: Опциональный вектор кратностей рассеяния длины `N`.

        Returns:
            Кортеж `(processed_events, united_scattering_orders)`.
        """
        united_events, united_orders = self.unite_acts(raw_acts, scattering_orders=scattering_orders)
        if united_events.shape[0] == 0:
            return united_events, united_orders

        true_deposited = united_events[:, 2].copy() if self.scale_spatial_with_energy else None
        self._smear_energy_inplace(united_events)
        self._smear_spatial_inplace(united_events, deposited_energies=true_deposited)
        return united_events, united_orders
