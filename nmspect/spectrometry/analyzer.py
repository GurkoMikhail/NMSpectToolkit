"""Трёхуровневый анализатор энергетических спектров сцинтилляционного детектора."""

from __future__ import annotations

import numpy as np
from hepunits import keV
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from nmspect.spectrometry.models import EnergyWindowConfig, SpectrumResult


class SpectrumProcessor:
    """Выполняет гистограммирование спектров вспышек и расчёт 6 диагностических метрик качества."""

    def __init__(
        self,
        energy_range: tuple[float, float] = (0.0 * keV, 300.0 * keV),
        num_bins: int = 1024,
    ) -> None:
        """Инициализирует параметры сетки энергетического спектра.

        Args:
            energy_range: Диапазон энергий (min_energy, max_energy) в базовых единицах hepunits.
            num_bins: Количество энергетических каналов (бинов) гистограммы.
        """
        if energy_range[1] <= energy_range[0]:
            raise ValueError(f"Некорректный диапазон энергий спектра: {energy_range}")
        if num_bins <= 0:
            raise ValueError(f"Число бинов спектра должно быть положительным: {num_bins}")

        self.energy_range = energy_range
        self.num_bins = num_bins

    def analyze(
        self,
        energies: NDArray[np.floating],
        scatter_orders: NDArray[np.integer] | None = None,
        window_config: EnergyWindowConfig | None = None,
    ) -> SpectrumResult:
        """Рассчитывает трёхуровневый энергетический спектр и 6 метрик качества для заданного окна.

        Args:
            energies: Массив зарегистрированных энергий сцинтилляционных вспышек (размерность M, в hepunits).
            scatter_orders: Массив кратностей комптоновского рассеяния в фантоме для каждой вспышки (размерность M).
            window_config: Конфигурация целевого энергетического окна (если None — метрики считаются по всему диапазону).

        Returns:
            Объект SpectrumResult с массивами спектров и метриками качества.
        """
        energies_arr = np.asarray(energies, dtype=np.float64).ravel()
        total_events_count = energies_arr.size

        if scatter_orders is not None:
            orders_arr = np.asarray(scatter_orders, dtype=np.int32).ravel()
            if orders_arr.size != total_events_count:
                raise ValueError(
                    f"Несовпадение размерностей массивов energies ({total_events_count}) "
                    f"и scatter_orders ({orders_arr.size})."
                )
        else:
            orders_arr = np.zeros(total_events_count, dtype=np.int32)

        bin_edges = np.linspace(self.energy_range[0], self.energy_range[1], self.num_bins + 1, dtype=np.float64)
        energy_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        total_hist, _ = np.histogram(energies_arr, bins=bin_edges)
        total_spectrum = total_hist.astype(np.float64)

        primary_mask = orders_arr == 0
        scatter_mask = orders_arr > 0

        primary_hist, _ = np.histogram(energies_arr[primary_mask], bins=bin_edges)
        scatter_hist, _ = np.histogram(energies_arr[scatter_mask], bins=bin_edges)

        primary_spectrum = primary_hist.astype(np.float64)
        scatter_spectrum = scatter_hist.astype(np.float64)

        # Оценка гладкого комптоновского континуума методом морфологического огибающего минимума (SNIP-подобный фильтр)
        filter_width = max(5, self.num_bins // 32)
        eroded_spectrum = minimum_filter1d(total_spectrum, size=filter_width, mode="nearest")
        compton_continuum = np.minimum(total_spectrum, uniform_filter1d(eroded_spectrum, size=filter_width, mode="nearest"))

        if window_config is not None:
            lower_bound = window_config.lower_bound
            upper_bound = window_config.upper_bound
        else:
            lower_bound = self.energy_range[0]
            upper_bound = self.energy_range[1]

        in_window_mask = (energies_arr >= lower_bound) & (energies_arr <= upper_bound)
        window_total_count = float(np.count_nonzero(in_window_mask))
        window_primary_count = float(np.count_nonzero(in_window_mask & primary_mask))
        window_scatter_count = float(np.count_nonzero(in_window_mask & scatter_mask))

        bin_in_window_mask = (energy_bins >= lower_bound) & (energy_bins <= upper_bound)
        continuum_in_window = float(np.sum(compton_continuum[bin_in_window_mask]))
        net_photopeak_in_window = max(0.0, window_total_count - continuum_in_window)

        primary_fraction = window_primary_count / window_total_count if window_total_count > 0.0 else 0.0
        scatter_to_primary_ratio = (
            window_scatter_count / window_primary_count if window_primary_count > 0.0 else 0.0
        )
        compton_to_primary_ratio = (
            continuum_in_window / net_photopeak_in_window if net_photopeak_in_window > 0.0 else 0.0
        )
        photopeak_efficiency = window_total_count / float(total_events_count) if total_events_count > 0 else 0.0
        mean_scatter_order = (
            float(np.mean(orders_arr[in_window_mask])) if window_total_count > 0.0 else 0.0
        )
        window_purity = net_photopeak_in_window / window_total_count if window_total_count > 0.0 else 0.0

        return SpectrumResult(
            energy_bins=energy_bins,
            bin_edges=bin_edges,
            total_spectrum=total_spectrum,
            primary_spectrum=primary_spectrum,
            scatter_spectrum=scatter_spectrum,
            compton_continuum=compton_continuum,
            primary_fraction=primary_fraction,
            scatter_to_primary_ratio=scatter_to_primary_ratio,
            compton_to_primary_ratio=compton_to_primary_ratio,
            photopeak_efficiency=photopeak_efficiency,
            mean_scatter_order=mean_scatter_order,
            window_purity=window_purity,
            window_config=window_config,
        )
