"""Моделирование физических процессов в сцинтилляционном кристалле и ФЭУ."""

from __future__ import annotations

import numpy as np
from hepunits import keV
from numpy.typing import NDArray

_FWHM_TO_SIGMA: float = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


class ScintillationProcessor:
    """Агрегирует микроскопические акты взаимодействия ($N$) в сцинтилляционные вспышки ($M$) и моделирует размытие."""

    def __init__(
        self,
        time_resolution: float,
        energy_resolution: float,
        intrinsic_spatial_resolution: float,
        reference_energy: float | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Инициализирует параметры аппаратного отклика сцинтилляционного детектора.

        Args:
            time_resolution: Временное разрешение (окно слияния импульсов / время высвечивания) в hepunits (ns).
            energy_resolution: Относительное энергетическое разрешение FWHM на референсной энергии (в долях или %).
            intrinsic_spatial_resolution: Собственное пространственное разрешение детектора FWHM в hepunits (mm).
            reference_energy: Референсная энергия калибровки разрешения в hepunits (по умолчанию 140.5 кэВ).
            rng: Генератор псевдослучайных чисел `np.random.Generator` или целочисленное зерно (`seed`).
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
        self.rng: np.random.Generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )

    def unite_acts(
        self,
        raw_acts: NDArray[np.floating],
        scattering_orders: NDArray[np.integer] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Векторно объединяет микроскопические акты взаимодействия ($N$) в сцинтилляционные вспышки ($M$).

        Использует `np.add.reduceat` и `np.maximum.reduceat` для расчёта суммарного энерговыделения,
        средневзвешенных по энергии координат центра тяжести светового пятна и кратности рассеяния.

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

        act_energies = sorted_acts[:, 2]
        summed_energies = np.add.reduceat(act_energies, group_starts)
        valid_mask = summed_energies > 0.0
        if not np.any(valid_mask):
            return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.int32)

        weighted_times = np.add.reduceat(sorted_acts[:, 1] * act_energies, group_starts)
        weighted_x = np.add.reduceat(sorted_acts[:, 3] * act_energies, group_starts)
        weighted_y = np.add.reduceat(sorted_acts[:, 4] * act_energies, group_starts)
        united_orders = np.maximum.reduceat(sorted_orders, group_starts).astype(np.int32)

        valid_energies = summed_energies[valid_mask]
        event_pids = sorted_acts[group_starts[valid_mask], 0]
        event_times = weighted_times[valid_mask] / valid_energies
        event_x = weighted_x[valid_mask] / valid_energies
        event_y = weighted_y[valid_mask] / valid_energies

        events_matrix = np.column_stack((event_pids, event_times, valid_energies, event_x, event_y))
        return events_matrix, united_orders[valid_mask]

    def _smear_energy_inplace(self, events_matrix: NDArray[np.float64]) -> None:
        """Модифицирует столбец энергии (индекс 2) матрицы вспышек `(M, 5)` на месте."""
        if events_matrix.size == 0 or self.energy_resolution == 0.0:
            return
        if events_matrix.ndim != 2 or events_matrix.shape[1] < 5:
            raise ValueError(f"Матрица events должна иметь форму (M, 5), получено: {events_matrix.shape}")

        relative_fwhm_ref = (
            self.energy_resolution / 100.0 if self.energy_resolution > 1.0 else self.energy_resolution
        )
        energies = np.clip(events_matrix[:, 2], 0.0, None)
        fwhm_values = relative_fwhm_ref * np.sqrt(self.reference_energy * energies)
        sigma_values = fwhm_values / _FWHM_TO_SIGMA

        smeared_energies = self.rng.normal(loc=energies, scale=sigma_values)
        events_matrix[:, 2] = np.clip(smeared_energies, 0.0, None)

    def _smear_spatial_inplace(self, events_matrix: NDArray[np.float64]) -> None:
        """Модифицирует столбцы локальных координат (индексы 3 и 4) матрицы вспышек `(M, 5)` на месте."""
        if events_matrix.size == 0 or self.intrinsic_spatial_resolution == 0.0:
            return
        if events_matrix.ndim != 2 or events_matrix.shape[1] < 5:
            raise ValueError(f"Матрица events должна иметь форму (M, 5), получено: {events_matrix.shape}")

        sigma_spatial = self.intrinsic_spatial_resolution / _FWHM_TO_SIGMA
        events_matrix[:, 3:5] = self.rng.normal(loc=events_matrix[:, 3:5], scale=sigma_spatial)

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
        raw_acts: NDArray[np.floating],
        scattering_orders: NDArray[np.integer] | None = None,
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

        self._smear_energy_inplace(united_events)
        self._smear_spatial_inplace(united_events)
        return united_events, united_orders
