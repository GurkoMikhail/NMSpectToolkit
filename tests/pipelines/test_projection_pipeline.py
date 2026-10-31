"""Интеграционные тесты физики детектора (ScintillationProcessor, ProjectionBuilder) и сквозного конвейера ОФЭКТ."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from hepunits import deg, keV, mm, ns

from nmspect.detector import (
    ProjectionBuilder,
    ScintillationProcessor,
    generate_projections_from_events,
)
from nmspect.pipelines import DEFAULT_REFERENCE_ENERGIES, run_isotope_nema_pipeline
from nmspect.spectrometry import TC99M_PEAK_140, get_energy_window_preset


def test_scintillation_processor_vectorized_unite_acts() -> None:
    """Проверяет векторизованное слияние микроскопических актов во вспышки по particle_ID и времени."""
    processor = ScintillationProcessor(
        time_resolution=200.0 * ns,
        energy_resolution=0.095,
        intrinsic_spatial_resolution=3.8 * mm,
        reference_energy=74.6 * keV,
        rng=12345,
    )

    # Два акта от частицы ID=10 в пределах 200 нс должны слиться в одну вспышку с суммарной энергией
    raw_acts = np.array(
        [
            [10.0, 100.0 * ns, 30.0 * keV, -10.0 * mm, 20.0 * mm],
            [10.0, 120.0 * ns, 44.6 * keV, 10.0 * mm, 20.0 * mm],
            [20.0, 500.0 * ns, 48.9 * keV, 0.0 * mm, -15.0 * mm],
        ],
        dtype=np.float64,
    )
    raw_orders = np.array([0, 1, 0], dtype=np.int32)

    united_events, united_orders = processor.unite_acts(raw_acts, scattering_orders=raw_orders)
    assert united_events.shape == (2, 5)
    assert united_orders.shape == (2,)
    assert np.isclose(united_events[0, 2], 74.6 * keV)
    assert int(united_orders[0]) == 1
    assert np.isclose(united_events[1, 2], 48.9 * keV)
    assert int(united_orders[1]) == 0

    blurred_events, final_orders = processor.process_detector_acts(raw_acts, scattering_orders=raw_orders)
    assert blurred_events.shape == (2, 5)
    assert np.array_equal(final_orders, united_orders)


def test_projection_builder_and_scatter_subtraction() -> None:
    """Проверяет 2D-гистограммирование проекций и весовое вычитание рассеяния TEW."""
    events_view = np.array(
        [
            [1.0, 10.0 * ns, 74.6 * keV, 0.0 * mm, 0.0 * mm],
            [2.0, 20.0 * ns, 75.0 * keV, 0.0 * mm, 0.0 * mm],
            [3.0, 30.0 * ns, 64.0 * keV, 0.0 * mm, 0.0 * mm],
        ],
        dtype=np.float64,
    )
    detector_size = (64.0 * mm, 64.0 * mm, 9.5 * mm)
    pixel_size = 4.0 * mm

    tew_windows = get_energy_window_preset("peak75_tew", isotope="161Tb")
    builder = ProjectionBuilder(
        pixel_size=pixel_size,
        energy_windows=tew_windows,
        detector_size=detector_size,
    )
    combined, per_window = builder.build_projections([events_view])
    assert combined.shape == (1, 16, 16)
    assert "peak_75_tew" in per_window
    assert np.sum(combined) > 0.0
    assert np.sum(combined) < 2.0  # Из-за вычитания события в нижнем окне рассеяния 64 кэВ

    combined_direct, _ = generate_projections_from_events(
        events_list=[events_view],
        detector_size=detector_size,
        pixel_size=pixel_size,
        energy_windows=tew_windows,
    )
    assert np.allclose(combined, combined_direct)


def test_end_to_end_isotope_nema_pipeline(tmp_path: Path) -> None:
    """Проверяет полный цикл конвейера run_isotope_nema_pipeline на синтетической директории HDF5."""
    raw_dir = tmp_path / "raw_sim"
    raw_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"

    for angle_val in (0.0, 90.0):
        hdf_path = raw_dir / f"{angle_val}_deg.hdf"
        with h5py.File(hdf_path, "w") as h5_file:
            init_grp = h5_file.create_group("initial_states")
            init_grp.create_dataset("particle_ID", data=np.array([1, 2], dtype=np.uint64))
            init_grp.create_dataset("emission_time", data=np.array([5.0, 15.0], dtype=np.float32))

            inter_grp = h5_file.create_group("interactions")
            det_grp = inter_grp.create_group("Detector_1")
            det_grp.create_dataset("particle_ID", data=np.array([1, 2], dtype=np.uint64))
            det_grp.create_dataset("energy_deposit", data=np.array([0.0746, 0.1405], dtype=np.float32))
            det_grp.create_dataset(
                "global_position",
                data=np.array([[0.0, 200.0, 0.0], [10.0, 200.0, -10.0]], dtype=np.float32),
            )

            meta_grp = h5_file.create_group("metadata")
            gantry_grp = meta_grp.create_group("gantry")
            gantry_grp.attrs["gantry_angle"] = float(angle_val * deg)
            gantry_grp.attrs["gantry_angle_units"] = "rad"
            geom_grp = meta_grp.create_group("geometry")
            geom_grp.attrs["crystal_dimensions"] = np.array([128.0, 128.0, 9.5], dtype=np.float64)

    assert np.isclose(DEFAULT_REFERENCE_ENERGIES["161Tb"], TC99M_PEAK_140)

    exported_tb = run_isotope_nema_pipeline(
        isotope="161Tb",
        data_dir=raw_dir,
        preset_keys=["peak75", "dual_peak"],
        pixel_size=8.0 * mm,
        output_root=out_dir,
        export_legacy_formats=True,
        rng_seed=42,
    )
    assert "peak75" in exported_tb
    assert "dual_peak" in exported_tb
    assert exported_tb["peak75"]["nexus"].is_file()
    assert exported_tb["peak75"]["dicom"].is_file()
    assert exported_tb["peak75"]["paraview"].is_file()
    assert exported_tb["peak75"]["numpy_projections"].is_file()
