"""Модульные тесты физики сцинтилляционного детектора (ScintillationProcessor) и биннинга проекций (ProjectionBuilder)."""

from __future__ import annotations

import numpy as np
import pytest
from hepunits import keV, mm, ns

from nmspect.detector import (
    ProjectionBuilder,
    ScintillationProcessor,
    build_projection_for_energy_window,
)
from nmspect.spectrometry import get_energy_window_preset


def test_scintillation_processor_validation_and_resolution() -> None:
    """Проверяет валидацию предусловий LBYL, энергетическое и пространственное размытие вспышек."""
    with pytest.raises(ValueError):
        ScintillationProcessor(
            time_resolution=-10.0 * ns,
            energy_resolution=0.095,
            intrinsic_spatial_resolution=3.8 * mm,
        )

    with pytest.raises(ValueError):
        ScintillationProcessor(
            time_resolution=200.0 * ns,
            energy_resolution=-0.05,
            intrinsic_spatial_resolution=3.8 * mm,
        )

    with pytest.raises(ValueError):
        ScintillationProcessor(
            time_resolution=200.0 * ns,
            energy_resolution=0.095,
            intrinsic_spatial_resolution=3.8 * mm,
            reference_energy=0.0 * keV,
        )

    processor = ScintillationProcessor(
        time_resolution=230.0 * ns,
        energy_resolution=0.095,
        intrinsic_spatial_resolution=3.8 * mm,
        reference_energy=74.6 * keV,
        rng=42,
    )

    raw_acts = np.array(
        [
            [1.0, 10.0 * ns, 25.0 * keV, -60.0 * mm, 40.0 * mm],
            [1.0, 25.0 * ns, 49.6 * keV, -30.0 * mm, 40.0 * mm],
            [1.0, 500.0 * ns, 74.6 * keV, 50.0 * mm, -20.0 * mm],
        ],
        dtype=np.float64,
    )
    scatter_orders = np.array([0, 2, 0], dtype=np.int32)

    united_events, united_orders = processor.unite_acts(raw_acts, scattering_orders=scatter_orders)
    assert united_events.shape == (2, 5)
    assert list(united_orders) == [2, 0]
    assert np.isclose(united_events[0, 2], 74.6 * keV)
    expected_x = (-60.0 * 25.0 + -30.0 * 49.6) / 74.6 * mm
    assert np.isclose(united_events[0, 3], expected_x)
    assert np.isclose(united_events[0, 4], 40.0 * mm)

    smeared_events, final_orders = processor.process_detector_acts(raw_acts, scattering_orders=scatter_orders)
    assert smeared_events.shape == (2, 5)
    assert np.array_equal(final_orders, united_orders)
    assert np.all(smeared_events[:, 2] > 0.0)


def test_projection_builder_validation_and_window_clipping() -> None:
    """Проверяет валидацию параметров ProjectionBuilder и режим clip_negative при вычитании DEW."""
    dew_window = get_energy_window_preset("peak75_dew", isotope="161Tb")[0]

    with pytest.raises(ValueError):
        ProjectionBuilder(pixel_size=0.0 * mm, energy_windows=(dew_window,))

    with pytest.raises(ValueError):
        ProjectionBuilder(pixel_size=4.2 * mm, energy_windows=())

    # Только рассеянное событие в окне DEW (60 кэВ) даёт отрицательный вклад до обрезки
    energies = np.array([60.0 * keV], dtype=np.float64)
    coords_x = np.array([0.0 * mm], dtype=np.float64)
    coords_y = np.array([0.0 * mm], dtype=np.float64)
    det_size = (32.0 * mm, 32.0 * mm, 9.5 * mm)

    proj_unclipped = build_projection_for_energy_window(
        energies=energies,
        coords_x=coords_x,
        coords_y=coords_y,
        detector_size=det_size,
        pixel_size=4.0 * mm,
        window_config=dew_window,
        clip_negative=False,
    )
    assert np.min(proj_unclipped) < 0.0

    proj_clipped = build_projection_for_energy_window(
        energies=energies,
        coords_x=coords_x,
        coords_y=coords_y,
        detector_size=det_size,
        pixel_size=4.0 * mm,
        window_config=dew_window,
        clip_negative=True,
    )
    assert np.min(proj_clipped) >= 0.0
