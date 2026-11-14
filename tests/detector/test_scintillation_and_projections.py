"""Модульные тесты физики сцинтилляционного детектора (ScintillationProcessor) и биннинга проекций (ProjectionBuilder)."""

from __future__ import annotations

import numpy as np
import pytest
from hepunits import keV, mm, ns

from nmspect.detector import (
    ProjectionBuilder,
    ScintillationProcessor,
    aggregate_energy_weighted_segments,
    build_projection_for_energy_window,
    compute_spatial_offsets_2d,
    generate_projections_from_events,
    normalize_energy_resolution,
    smear_scintillation_energy,
)
from nmspect.spectrometry import (
    TB161_PEAK_75_DEW_WINDOW,
    TB161_PEAK_75_TEW_WINDOW,
    TB161_PEAK_75_WINDOW,
    TB161_SCATTER_75_DEW,
    get_energy_window_preset,
)


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

    # Проверка фильтрации актов с нулевым энерговыделением (упругое когерентное рассеяние в кристалле)
    acts_with_zero_edep = np.array(
        [
            [7.0, 10.0 * ns, 74.6 * keV, 12.0 * mm, -8.0 * mm],
            [7.0, 12.0 * ns, 0.0 * keV, 99.0 * mm, 99.0 * mm],
        ],
        dtype=np.float64,
    )
    orders_with_zero_edep = np.array([0, 3], dtype=np.int32)
    clean_events, clean_orders = processor.unite_acts(acts_with_zero_edep, scattering_orders=orders_with_zero_edep)
    assert clean_events.shape == (1, 5)
    assert list(clean_orders) == [0]
    assert np.isclose(clean_events[0, 3], 12.0 * mm)
    assert np.isclose(clean_events[0, 4], -8.0 * mm)


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

    # Проверка сохранения отрицательного вклада отдельного окна рассеяния (is_scatter_window=True) до суммирования
    events_mix = np.array(
        [
            [1.0, 10.0 * ns, 74.6 * keV, 0.0 * mm, 0.0 * mm],
            [2.0, 20.0 * ns, 74.6 * keV, 0.0 * mm, 0.0 * mm],
            [3.0, 30.0 * ns, 60.0 * keV, 0.0 * mm, 0.0 * mm],
        ],
        dtype=np.float64,
    )
    comb_separate, per_win_separate = generate_projections_from_events(
        events_list=[events_mix],
        detector_size=det_size,
        pixel_size=4.0 * mm,
        energy_windows=(TB161_PEAK_75_WINDOW, TB161_SCATTER_75_DEW),
        clip_negative=True,
    )
    comb_assoc, _ = generate_projections_from_events(
        events_list=[events_mix],
        detector_size=det_size,
        pixel_size=4.0 * mm,
        energy_windows=(TB161_PEAK_75_DEW_WINDOW,),
        clip_negative=True,
    )
    assert np.min(per_win_separate["scatter_75_dew"]) < 0.0
    assert np.allclose(comb_separate, comb_assoc)

    # Проверка отсутствия двойного учёта событий ровно на смежных границах окон TEW (67.1 кэВ и 82.1 кэВ)
    boundary_energies = np.array([67.1 * keV, 82.1 * keV], dtype=np.float64)
    boundary_x = np.array([0.0 * mm, 0.0 * mm], dtype=np.float64)
    boundary_y = np.array([0.0 * mm, 0.0 * mm], dtype=np.float64)
    tew_boundary_proj = build_projection_for_energy_window(
        energies=boundary_energies,
        coords_x=boundary_x,
        coords_y=boundary_y,
        detector_size=det_size,
        pixel_size=4.0 * mm,
        window_config=TB161_PEAK_75_TEW_WINDOW,
        clip_negative=False,
    )
    assert np.isclose(np.sum(tew_boundary_proj), 2.0)


def test_shared_scintillation_kernel_functions() -> None:
    """Проверяет единые векторные функции сцинтилляционного отклика для 2D актов и 3D треков."""
    assert np.isclose(normalize_energy_resolution(9.9), 0.099)
    assert np.isclose(normalize_energy_resolution(0.099), 0.099)
    with pytest.raises(ValueError):
        normalize_energy_resolution(-0.01)

    starts = np.array([0, 2, 4], dtype=np.int64)
    edeps = np.array([25.0 * keV, 50.0 * keV, 0.0 * keV, 0.0 * keV, 100.0 * keV], dtype=np.float64)
    coords_3d = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 9.0],
            [10.0, 20.0, 30.0],
            [20.0, 40.0, 50.0],
            [-5.0, 5.0, 2.0],
        ],
        dtype=np.float64,
    ) * float(mm)
    times = np.array([1.0, 4.0, 10.0, 12.0, 20.0], dtype=np.float64) * float(ns)
    fb_coords = np.array([[0.0, 0.0, 0.0], [15.0, 30.0, 40.0], [0.0, 0.0, 0.0]], dtype=np.float64) * float(mm)
    fb_times = np.array([0.0, 11.0, 0.0], dtype=np.float64) * float(ns)

    summed_e, pos_mask, centroids, mean_t = aggregate_energy_weighted_segments(
        segment_starts=starts,
        energy_deposits=edeps,
        coordinates=coords_3d,
        times=times,
        fallback_coordinates=fb_coords,
        fallback_times=fb_times,
    )
    assert list(pos_mask) == [True, False, True]
    assert np.allclose(summed_e, [75.0 * keV, 0.0 * keV, 100.0 * keV])
    assert np.isclose(centroids[0, 2], 7.0 * mm)
    assert np.allclose(centroids[1], [15.0 * mm, 30.0 * mm, 40.0 * mm])
    assert np.isclose(mean_t[0], 3.0 * ns)
    assert np.isclose(mean_t[1], 11.0 * ns)

    smeared = smear_scintillation_energy(
        deposited_energies=summed_e,
        energy_resolution=9.9,
        reference_energy=140.5 * keV,
        rng=np.random.default_rng(7),
        active_mask=pos_mask,
    )
    assert smeared[0] > 0.0
    assert smeared[1] == 0.0
    assert smeared[2] > 0.0

    offsets_scaled = compute_spatial_offsets_2d(
        num_events=3,
        intrinsic_spatial_resolution=3.8 * mm,
        rng=np.random.default_rng(7),
        deposited_energies=summed_e,
        reference_energy=140.5 * keV,
        scale_with_energy=True,
        active_mask=pos_mask,
    )
    assert offsets_scaled.shape == (3, 2)
    assert np.allclose(offsets_scaled[1], [0.0, 0.0])

    # Энергетическое масштабирование пространственного размытия напрямую в ScintillationProcessor
    scaled_proc = ScintillationProcessor(
        time_resolution=200.0 * ns,
        energy_resolution=0.099,
        intrinsic_spatial_resolution=3.8 * mm,
        rng=123,
        scale_spatial_with_energy=True,
    )
    raw_pair = np.array(
        [
            [1.0, 10.0 * ns, 35.0 * keV, 0.0 * mm, 0.0 * mm],
            [2.0, 20.0 * ns, 140.5 * keV, 0.0 * mm, 0.0 * mm],
        ],
        dtype=np.float64,
    )
    proc_events, _ = scaled_proc.process_detector_acts(raw_pair)
    assert proc_events.shape == (2, 5)
