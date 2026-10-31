"""Трёхуровневый анализатор энергетических спектров сцинтилляционного детектора."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
        self._cached_energies_ref: NDArray[Any] | None = None
        self._cached_orders_ref: NDArray[Any] | None = None
        self._cached_fingerprint: tuple[Any, ...] | None = None
        self._cached_bin_edges: NDArray[np.float64] | None = None
        self._cached_energy_bins: NDArray[np.float64] | None = None
        self._cached_total_spectrum: NDArray[np.float64] | None = None
        self._cached_primary_spectrum: NDArray[np.float64] | None = None
        self._cached_scatter_spectrum: NDArray[np.float64] | None = None
        self._cached_primary_mask: NDArray[np.bool_] | None = None
        self._cached_scatter_mask: NDArray[np.bool_] | None = None

    @staticmethod
    def _compute_input_fingerprint(
        energies_arr: NDArray[np.float64],
        orders_arr: NDArray[np.int32],
    ) -> tuple[Any, ...]:
        """Формирует компактный контрольный отпечаток входных массивов для защиты кэша от in-place мутаций."""
        if energies_arr.size == 0:
            return (0, 0)
        return (
            int(energies_arr.size),
            int(orders_arr.size),
            float(energies_arr[0]),
            float(energies_arr[-1]),
            float(np.sum(energies_arr)),
            int(orders_arr[0]),
            int(orders_arr[-1]),
            int(np.sum(orders_arr)),
        )

    def analyze(
        self,
        energies: NDArray[np.floating],
        scatter_orders: NDArray[np.integer] | None = None,
        window_config: EnergyWindowConfig | Sequence[EnergyWindowConfig] | None = None,
    ) -> SpectrumResult:
        """Рассчитывает трёхуровневый энергетический спектр и 6 метрик качества для заданного окна или набора окон.

        Args:
            energies: Массив зарегистрированных энергий сцинтилляционных вспышек (размерность M, в hepunits).
            scatter_orders: Массив кратностей комптоновского рассеяния в фантоме для каждой вспышки (размерность M).
            window_config: Конфигурация целевого энергетического окна или последовательность окон
                (если None — метрики считаются по всему диапазону).

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

        active_windows: tuple[EnergyWindowConfig, ...]
        primary_window_config: EnergyWindowConfig | None
        if isinstance(window_config, EnergyWindowConfig):
            active_windows = (window_config,)
            primary_window_config = window_config
        elif window_config is not None:
            active_windows = tuple(window_config)
            primary_window_config = active_windows[0] if len(active_windows) > 0 else None
        else:
            active_windows = ()
            primary_window_config = None

        bin_step = (self.energy_range[1] - self.energy_range[0]) / float(self.num_bins)
        current_fingerprint = self._compute_input_fingerprint(energies_arr, orders_arr)

        cache_hit = (
            energies is self._cached_energies_ref
            and scatter_orders is self._cached_orders_ref
            and current_fingerprint == self._cached_fingerprint
            and self._cached_bin_edges is not None
            and self._cached_energy_bins is not None
            and self._cached_total_spectrum is not None
            and self._cached_primary_spectrum is not None
            and self._cached_scatter_spectrum is not None
            and self._cached_primary_mask is not None
            and self._cached_scatter_mask is not None
        )

        if cache_hit:
            assert self._cached_bin_edges is not None
            assert self._cached_energy_bins is not None
            assert self._cached_total_spectrum is not None
            assert self._cached_primary_spectrum is not None
            assert self._cached_scatter_spectrum is not None
            assert self._cached_primary_mask is not None
            assert self._cached_scatter_mask is not None
            bin_edges = self._cached_bin_edges
            energy_bins = self._cached_energy_bins
            total_spectrum = self._cached_total_spectrum
            primary_spectrum = self._cached_primary_spectrum
            scatter_spectrum = self._cached_scatter_spectrum
            primary_mask = self._cached_primary_mask
            scatter_mask = self._cached_scatter_mask
        else:
            bin_edges = np.linspace(self.energy_range[0], self.energy_range[1], self.num_bins + 1, dtype=np.float64)
            energy_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            primary_mask = orders_arr == 0
            scatter_mask = orders_arr > 0

            primary_hist, _ = np.histogram(
                energies_arr[primary_mask],
                bins=self.num_bins,
                range=self.energy_range,
            )
            scatter_hist, _ = np.histogram(
                energies_arr[scatter_mask],
                bins=self.num_bins,
                range=self.energy_range,
            )

            primary_spectrum = primary_hist.astype(np.float64)
            scatter_spectrum = scatter_hist.astype(np.float64)

            if np.any(orders_arr < 0):
                total_hist, _ = np.histogram(
                    energies_arr,
                    bins=self.num_bins,
                    range=self.energy_range,
                )
                total_spectrum = total_hist.astype(np.float64)
            else:
                total_spectrum = primary_spectrum + scatter_spectrum

            self._cached_energies_ref = energies
            self._cached_orders_ref = scatter_orders
            self._cached_fingerprint = current_fingerprint
            self._cached_bin_edges = bin_edges
            self._cached_energy_bins = energy_bins
            self._cached_total_spectrum = total_spectrum
            self._cached_primary_spectrum = primary_spectrum
            self._cached_scatter_spectrum = scatter_spectrum
            self._cached_primary_mask = primary_mask
            self._cached_scatter_mask = scatter_mask

        # Оценка гладкого комптоновского континуума методом морфологического огибающего минимума:
        # ширина структурного элемента должна перекрывать полное основание фотопика (>= 1.8 ширины окна)
        if active_windows:
            max_window_width = max(win.width for win in active_windows)
            target_filter_span = max(1.8 * max_window_width, 30.0 * keV)
        else:
            target_filter_span = max(0.15 * (self.energy_range[1] - self.energy_range[0]), 30.0 * keV)
        filter_width = max(5, min(self.num_bins, int(np.ceil(target_filter_span / bin_step))))

        eroded_spectrum = minimum_filter1d(total_spectrum, size=filter_width, mode="nearest")
        compton_continuum = np.minimum(
            total_spectrum,
            uniform_filter1d(eroded_spectrum, size=filter_width, mode="nearest"),
        )

        if active_windows:
            in_window_mask = np.zeros(total_events_count, dtype=bool)
            bin_in_window_mask = np.zeros(self.num_bins, dtype=bool)
            for win_cfg in active_windows:
                in_window_mask |= (energies_arr >= win_cfg.lower_bound) & (energies_arr <= win_cfg.upper_bound)
                bin_in_window_mask |= (energy_bins >= win_cfg.lower_bound) & (energy_bins <= win_cfg.upper_bound)
        else:
            lower_bound = self.energy_range[0]
            upper_bound = self.energy_range[1]
            in_window_mask = (energies_arr >= lower_bound) & (energies_arr <= upper_bound)
            bin_in_window_mask = (energy_bins >= lower_bound) & (energy_bins <= upper_bound)

        raw_window_total = float(np.count_nonzero(in_window_mask))
        raw_window_primary = float(np.count_nonzero(in_window_mask & primary_mask))
        raw_window_scatter = float(np.count_nonzero(in_window_mask & scatter_mask))

        # Учёт вычитания рассеяния по связанным вспомогательным окнам DEW/TEW
        estimated_scatter_subtraction = 0.0
        for win_cfg in active_windows:
            for assoc_win in win_cfg.associated_windows:
                low_cond = (
                    energies_arr > assoc_win.lower_bound
                    if np.isclose(assoc_win.lower_bound, win_cfg.upper_bound, rtol=0.0, atol=1e-12)
                    else energies_arr >= assoc_win.lower_bound
                )
                high_cond = (
                    energies_arr < assoc_win.upper_bound
                    if np.isclose(assoc_win.upper_bound, win_cfg.lower_bound, rtol=0.0, atol=1e-12)
                    else energies_arr <= assoc_win.upper_bound
                )
                assoc_count = float(np.count_nonzero(low_cond & high_cond))
                if assoc_win.scatter_weight < 0.0:
                    estimated_scatter_subtraction += (-float(assoc_win.scatter_weight)) * assoc_count

        effective_scatter_count = max(0.0, raw_window_scatter - estimated_scatter_subtraction)
        effective_total_count = raw_window_primary + effective_scatter_count

        hist_total_in_window = float(np.sum(total_spectrum[bin_in_window_mask]))
        raw_continuum_in_window = float(np.sum(compton_continuum[bin_in_window_mask]))
        if hist_total_in_window > 0.0 and raw_window_total > 0.0:
            raw_continuum_in_window *= raw_window_total / hist_total_in_window

        continuum_in_window = max(0.0, raw_continuum_in_window - estimated_scatter_subtraction)
        net_photopeak_in_window = max(0.0, effective_total_count - continuum_in_window)

        primary_fraction = raw_window_primary / effective_total_count if effective_total_count > 0.0 else 0.0
        scatter_to_primary_ratio = (
            effective_scatter_count / raw_window_primary if raw_window_primary > 0.0 else 0.0
        )
        compton_to_primary_ratio = (
            continuum_in_window / net_photopeak_in_window if net_photopeak_in_window > 0.0 else 0.0
        )
        photopeak_efficiency = effective_total_count / float(total_events_count) if total_events_count > 0 else 0.0

        raw_mean_scatter_order = (
            float(np.sum(orders_arr, where=in_window_mask)) / raw_window_total
            if raw_window_total > 0.0
            else 0.0
        )
        scatter_retention_ratio = (
            effective_scatter_count / raw_window_scatter if raw_window_scatter > 0.0 else 1.0
        )
        mean_scatter_order = raw_mean_scatter_order * scatter_retention_ratio
        window_purity = net_photopeak_in_window / effective_total_count if effective_total_count > 0.0 else 0.0

        return SpectrumResult(
            energy_bins=energy_bins.copy(),
            bin_edges=bin_edges.copy(),
            total_spectrum=total_spectrum.copy(),
            primary_spectrum=primary_spectrum.copy(),
            scatter_spectrum=scatter_spectrum.copy(),
            compton_continuum=compton_continuum,
            primary_fraction=primary_fraction,
            scatter_to_primary_ratio=scatter_to_primary_ratio,
            compton_to_primary_ratio=compton_to_primary_ratio,
            photopeak_efficiency=photopeak_efficiency,
            mean_scatter_order=mean_scatter_order,
            window_purity=window_purity,
            window_config=primary_window_config,
        )
