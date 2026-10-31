"""Тесты конфигураций энергетических окон, расчёта весов DEW/TEW, реестра пресетов и анализатора спектров."""

from __future__ import annotations

import numpy as np
import pytest
from hepunits import keV

from nmspect.spectrometry import (
    TB161_ENERGY_WINDOW_PRESETS,
    TB161_PEAK_49,
    TB161_PEAK_75,
    TC99M_ENERGY_WINDOW_PRESETS,
    TC99M_PEAK_140,
    EnergyWindowConfig,
    SpectrumProcessor,
    calculate_dew_coefficient,
    calculate_tew_coefficients,
    format_presets_summary,
    get_energy_window_preset,
    list_available_presets,
)


def test_energy_window_config_properties_and_validation() -> None:
    """Проверяет расчёт ширины/центра окна и валидацию некорректных границ."""
    window = EnergyWindowConfig(
        name="test_peak",
        lower_bound=60.0 * keV,
        upper_bound=80.0 * keV,
        primary_peak=70.0 * keV,
    )
    assert np.isclose(window.width, 20.0 * keV)
    assert np.isclose(window.center, 70.0 * keV)
    assert window.primary_peak is not None
    assert np.isclose(window.primary_peak, 70.0 * keV)

    with pytest.raises(ValueError):
        EnergyWindowConfig(name="invalid", lower_bound=80.0 * keV, upper_bound=60.0 * keV)

    with pytest.raises(ValueError):
        EnergyWindowConfig(name="negative", lower_bound=-5.0 * keV, upper_bound=10.0 * keV)


def test_tew_and_dew_coefficients() -> None:
    """Проверяет математическую точность весовых коэффициентов DEW и TEW."""
    low_weight, high_weight = calculate_tew_coefficients(
        main_lower=67.1 * keV,
        main_upper=82.1 * keV,
        low_scatter_lower=61.1 * keV,
        low_scatter_upper=67.1 * keV,
        high_scatter_lower=82.1 * keV,
        high_scatter_upper=88.1 * keV,
    )
    assert np.isclose(low_weight, -0.5 * (15.0 / 6.0))
    assert np.isclose(high_weight, -0.5 * (15.0 / 6.0))

    dew_weight = calculate_dew_coefficient(
        main_lower=44.0 * keV,
        main_upper=53.8 * keV,
        scatter_lower=35.0 * keV,
        scatter_upper=43.0 * keV,
        k_factor=0.5,
    )
    assert np.isclose(dew_weight, -0.5 * (9.8 / 8.0))


def test_presets_registry_for_tb161_and_tc99m() -> None:
    """Проверяет реестр пресетов энергетических окон для 161Tb и 99mTc."""
    tb_keys = list_available_presets("161Tb")
    assert set(tb_keys) == set(TB161_ENERGY_WINDOW_PRESETS.keys())

    tc_keys = list_available_presets("99mTc")
    assert set(tc_keys) == set(TC99M_ENERGY_WINDOW_PRESETS.keys())

    dual_windows = get_energy_window_preset("dual_peak", isotope="tb161")
    assert len(dual_windows) == 2
    assert np.isclose(dual_windows[0].primary_peak or 0.0, TB161_PEAK_49)
    assert np.isclose(dual_windows[1].primary_peak or 0.0, TB161_PEAK_75)

    tc_tew = get_energy_window_preset("peak140_tew", isotope="tc99m")
    assert len(tc_tew) == 1
    assert len(tc_tew[0].associated_windows) == 2
    assert np.isclose(tc_tew[0].primary_peak or 0.0, TC99M_PEAK_140)

    summary_text = format_presets_summary()
    assert "161Tb" in summary_text
    assert "99mTc" in summary_text


def test_spectrum_processor_three_level_metrics() -> None:
    """Проверяет расчёт 3 компонент спектра и 6 метрик качества в SpectrumProcessor."""
    rng = np.random.default_rng(42)
    primary_energies = rng.normal(loc=74.6 * keV, scale=3.0 * keV, size=800)
    scatter_energies = rng.uniform(low=20.0 * keV, high=72.0 * keV, size=200)

    energies = np.concatenate([primary_energies, scatter_energies])
    scatter_orders = np.concatenate([
        np.zeros(800, dtype=np.int32),
        np.ones(200, dtype=np.int32),
    ])

    processor = SpectrumProcessor(energy_range=(0.0 * keV, 150.0 * keV), num_bins=256)
    window = get_energy_window_preset("peak75", isotope="161Tb")[0]
    result = processor.analyze(energies, scatter_orders=scatter_orders, window_config=window)

    assert result.energy_bins.shape == (256,)
    assert np.isclose(np.sum(result.total_spectrum), 1000.0)
    assert np.isclose(
        np.sum(result.primary_spectrum) + np.sum(result.scatter_spectrum),
        np.sum(result.total_spectrum),
    )
    assert 0.8 < result.primary_fraction <= 1.0
    assert result.scatter_to_primary_ratio >= 0.0
    assert 0.5 < result.photopeak_efficiency <= 1.0
    assert 0.0 <= result.window_purity <= 1.0
