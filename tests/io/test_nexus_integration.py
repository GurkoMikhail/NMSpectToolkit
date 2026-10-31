"""Интеграционные тесты чтения HDF5-симуляций и экспорта в NeXus (.nxs), DICOM NM (.dcm) и VTK ParaView (.vtm)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pydicom
import pytest
import vtk
from hepunits import deg, keV, mm
from vtk.util import numpy_support

from nmspect.exporters import DicomNMExporter, NeXusExporter, VtkParaviewExporter
from nmspect.io import SimulationDataExtractor, parse_gantry_angle_from_filename
from nmspect.spectrometry import SpectrumProcessor, get_energy_window_preset
from nmspect.visualization import format_nexus_tree, inspect_nexus_file


def _create_synthetic_simulation_hdf5(file_path: Path) -> None:
    """Создаёт синтетический HDF5-файл симуляции с канонической иерархией метаданных и актов."""
    with h5py.File(file_path, "w") as h5_file:
        init_grp = h5_file.create_group("initial_states")
        init_grp.create_dataset("particle_ID", data=np.array([1, 2, 3], dtype=np.uint64))
        init_grp.create_dataset("emission_time", data=np.array([10.0, 20.0, 30.0], dtype=np.float32))

        inter_grp = h5_file.create_group("interactions")
        sim_vol = inter_grp.create_group("Simulation_volume")
        sim_vol.create_dataset("particle_ID", data=np.array([2], dtype=np.uint64))
        sim_vol.create_dataset("process_code", data=np.array([1], dtype=np.uint8))

        det_grp = inter_grp.create_group("Detector_1")
        det_grp.create_dataset("particle_ID", data=np.array([1, 1, 2, 3], dtype=np.uint64))
        det_grp.create_dataset(
            "energy_deposit",
            data=np.array([0.040, 0.0346, 0.0746, 0.0489], dtype=np.float32),
        )
        det_grp.create_dataset(
            "global_position",
            data=np.array(
                [
                    [10.0, 298.6, -5.0],
                    [10.2, 298.6, -5.0],
                    [-20.0, 298.6, 15.0],
                    [0.0, 298.6, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        meta_grp = h5_file.create_group("metadata")
        gantry_grp = meta_grp.create_group("gantry")
        gantry_grp.attrs["gantry_angle"] = 0.0
        gantry_grp.attrs["gantry_angle_units"] = "rad"
        gantry_grp.attrs["angular_step"] = 3.0
        gantry_grp.attrs["angular_step_units"] = "deg"
        pose_det1 = np.eye(4, dtype=np.float64)
        pose_det1[1, 3] = 298.6
        gantry_grp.attrs["pose_matrix_detector_1"] = pose_det1

        geom_grp = meta_grp.create_group("geometry")
        geom_grp.attrs["crystal_dimensions"] = np.array([540.0, 400.0, 9.5], dtype=np.float64)
        geom_grp.attrs["crystal_dimensions_units"] = "mm"

        src_grp = meta_grp.create_group("source")
        src_grp.attrs["radionuclide"] = "161Tb"
        src_grp.attrs["simulated_activity"] = 2.0e8


def test_simulation_data_extractor_and_exporters(tmp_path: Path) -> None:
    """Проверяет извлечение данных из HDF5 и экспорт в форматы NeXus, DICOM NM и VTK ParaView."""
    raw_hdf = tmp_path / "15.0_deg.hdf"
    _create_synthetic_simulation_hdf5(raw_hdf)

    parsed_angle = parse_gantry_angle_from_filename(raw_hdf.name)
    assert parsed_angle is not None
    assert np.isclose(parsed_angle, 15.0 * deg)

    extractor = SimulationDataExtractor(raw_hdf)
    detectors = extractor.get_detectors_names()
    assert "Detector_1" in detectors

    det_size = extractor.get_detector_size("Detector_1")
    assert np.allclose(det_size, (540.0 * mm, 400.0 * mm, 9.5 * mm))

    raw_acts = extractor.extract_data("Detector_1")
    scatter_orders = extractor.get_scattering_orders("Detector_1")
    assert raw_acts.shape == (4, 5)
    assert scatter_orders.shape == (4,)
    assert int(scatter_orders[2]) == 1

    # Подготовка синтетических проекций и спектра для проверки экспортеров
    projections = np.ones((6, 16, 16), dtype=np.float64) * 12.0
    angles = np.linspace(0.0, 150.0 * deg, 6)
    window_cfg = get_energy_window_preset("peak75", isotope="161Tb")[0]
    spec_proc = SpectrumProcessor(energy_range=(0.0 * keV, 150.0 * keV), num_bins=64)
    spec_result = spec_proc.analyze(raw_acts[:, 2], scatter_orders=scatter_orders, window_config=window_cfg)
    metadata = extractor.get_simulation_metadata()

    # 1. Экспорт в NeXus (.nxs)
    nxs_path = tmp_path / "study.nxs"
    NeXusExporter(nxs_path).export(
        projections=projections,
        angles=angles,
        spectrum_result=spec_result,
        pixel_size=4.2 * mm,
        detector_size=det_size,
        window_config=window_cfg,
        metadata=metadata,
    )
    assert nxs_path.is_file()
    with h5py.File(nxs_path, "r") as nxs_file:
        x_coords = np.asarray(nxs_file["/entry/data/x"][()], dtype=np.float64)
        y_coords = np.asarray(nxs_file["/entry/data/y"][()], dtype=np.float64)
        expected_coords = (np.arange(16, dtype=np.float64) - 0.5 * 15.0) * 4.2
        assert np.allclose(x_coords, expected_coords, atol=1e-4)
        assert np.allclose(y_coords, expected_coords, atol=1e-4)
        assert np.isclose(float(np.mean(x_coords)), 0.0, atol=1e-5)

    summary = inspect_nexus_file(nxs_path)
    assert summary["projections_shape"] == (6, 16, 16)
    assert "primary_fraction" in summary["metrics"]
    tree_str = format_nexus_tree(nxs_path)
    assert "NXtomo" in tree_str or "entry" in tree_str

    # 2. Экспорт в DICOM NM (.dcm)
    dcm_path = tmp_path / "study.dcm"
    DicomNMExporter(dcm_path).export(
        projections=projections,
        angles=angles,
        pixel_size=4.2 * mm,
        window_config=window_cfg,
        orbit_radius=298.6 * mm,
        metadata=metadata,
    )
    assert dcm_path.is_file()
    dcm_ds = pydicom.dcmread(dcm_path)
    assert dcm_ds.Modality == "NM"
    assert int(dcm_ds.NumberOfFrames) == 6
    assert list(dcm_ds.DetectorVector) == [1, 1, 1, 2, 2, 2]
    assert dcm_ds.RotationInformationSequence[0].RotationDirection == "CCW"

    # 3. Экспорт в VTK ParaView (.vtm) с проверкой выбора противоположного ракурса (+180 град) для Detector_2
    projections[0] = 10.0
    projections[1] = 20.0
    projections[3] = 99.0  # Вектор angles=[0, 30, 60, 90, 120, 150] -> ближайший к 180 град ракурс имеет индекс 5 (150 град)
    projections[5] = 77.0
    projections[5, 0, :] = 15.0
    projections[5, -1, :] = 85.0
    vtm_path = tmp_path / "study.vtm"
    VtkParaviewExporter(vtm_path).export(
        projections=projections,
        angles=angles,
        pixel_size=4.2 * mm,
        detector_size=det_size,
        orbit_radius=298.6 * mm,
        window_config=window_cfg,
        metadata=metadata,
    )
    assert vtm_path.is_file()
    det1_vti_path = tmp_path / "study" / "Detector_1_Projection.vti"
    det2_vti_path = tmp_path / "study" / "Detector_2_Projection.vti"
    col1_vti_path = tmp_path / "study" / "Collimator_1.vti"
    assert det1_vti_path.is_file()
    assert det2_vti_path.is_file()
    assert col1_vti_path.is_file()

    reader_det1 = vtk.vtkXMLImageDataReader()
    reader_det1.SetFileName(str(det1_vti_path))
    reader_det1.Update()
    det1_bounds = reader_det1.GetOutput().GetBounds()
    assert np.isclose(0.5 * (det1_bounds[2] + det1_bounds[3]), 298.6, atol=1e-3)

    reader_col1 = vtk.vtkXMLImageDataReader()
    reader_col1.SetFileName(str(col1_vti_path))
    reader_col1.Update()
    col1_bounds = reader_col1.GetOutput().GetBounds()
    # Верхняя грань коллиматора 1 должна примыкать к нижней (входной) грани кристалла 1 без пересечения
    assert np.isclose(col1_bounds[3], det1_bounds[2], atol=1e-3)

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(det2_vti_path))
    reader.Update()
    det2_bounds = reader.GetOutput().GetBounds()
    assert np.isclose(0.5 * (det2_bounds[2] + det2_bounds[3]), -298.6, atol=1e-3)
    det2_scalars = numpy_support.vtk_to_numpy(reader.GetOutput().GetCellData().GetScalars()).reshape((16, 16), order="F")
    # Проверка разворота оси X для противоположного детектора (индекс 0 в мировых X соответствует концу локальной оси X_2)
    assert np.allclose(det2_scalars[0, :], 85.0)
    assert np.allclose(det2_scalars[-1, :], 15.0)


def test_detector_pose_matrix_and_fail_fast_validation(tmp_path: Path) -> None:
    """Проверяет проекцию глобальных координат через реалистичную pose_matrix и отказ на легаси-датасете."""
    hdf_path = tmp_path / "0.0_deg.hdf"
    with h5py.File(hdf_path, "w") as h5_file:
        inter_grp = h5_file.create_group("interactions")
        det1_grp = inter_grp.create_group("Detector_1")
        det1_grp.create_dataset("particle_ID", data=np.array([1, 2], dtype=np.uint64))
        det1_grp.create_dataset("energy_deposit", data=np.array([0.0746, 0.0746], dtype=np.float32))
        det1_grp.create_dataset(
            "global_position",
            data=np.array([[-260.0, 298.6, -190.0], [260.0, 298.6, 190.0]], dtype=np.float32),
        )

        det2_grp = inter_grp.create_group("Detector_2")
        det2_grp.create_dataset("particle_ID", data=np.array([3], dtype=np.uint64))
        det2_grp.create_dataset("energy_deposit", data=np.array([0.0746], dtype=np.float32))
        det2_grp.create_dataset(
            "global_position",
            data=np.array([[150.0, -298.6, -120.0]], dtype=np.float32),
        )

        # Некорректный плоский датасет вместо группы детектора
        inter_grp.create_dataset("Invalid_Detector", data=np.zeros((2, 5), dtype=np.float64))

        meta_grp = h5_file.create_group("metadata")
        gantry_grp = meta_grp.create_group("gantry")
        pose_det1 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 298.6],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pose_det2 = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -298.6],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        gantry_grp.attrs["pose_matrix_detector_1"] = pose_det1
        gantry_grp.attrs["pose_matrix_detector_2"] = pose_det2

    extractor = SimulationDataExtractor(hdf_path)
    assert extractor.get_detectors_names() == ["Detector_1", "Detector_2"]

    acts_det1 = extractor.extract_data("Detector_1")
    assert np.allclose(acts_det1[:, 3], [-260.0 * mm, 260.0 * mm])
    assert np.allclose(acts_det1[:, 4], [-190.0 * mm, 190.0 * mm])

    acts_det2 = extractor.extract_data("Detector_2")
    assert np.allclose(acts_det2[:, 3], [-150.0 * mm])
    assert np.allclose(acts_det2[:, 4], [-120.0 * mm])

    with pytest.raises(TypeError):
        extractor.extract_data("Invalid_Detector")

    real_file = Path("data/raw/output_tb161_dual_cam/0.0_deg.hdf")
    if real_file.is_file():
        real_extractor = SimulationDataExtractor(real_file)
        for det_name in real_extractor.get_detectors_names():
            real_acts = real_extractor.extract_data(det_name)
            assert real_acts.shape[0] > 0
            assert float(np.min(real_acts[:, 3])) >= -270.1 * mm
            assert float(np.max(real_acts[:, 3])) <= 270.1 * mm
            assert float(np.min(real_acts[:, 4])) >= -200.1 * mm
            assert float(np.max(real_acts[:, 4])) <= 200.1 * mm


def test_causal_scattering_order_by_distance_traveled(tmp_path: Path) -> None:
    """Проверяет учёт только предшествующих рассеяний (sim_dist < det_dist) при расчёте scattering_order."""
    hdf_path = tmp_path / "causal_scatter.hdf"
    sim_pids = np.array([1, 2, 3, 3, 10, 10, 20, 20, 20, 20, 20, 30, 30, 30], dtype=np.uint64)
    sim_dists = np.array(
        [100.0, 260.0, 250.0, 270.0, 50.0, 250.0, 30.0, 80.0, 120.0, 310.0, 400.0, 10.0, 90.0, 180.0],
        dtype=np.float32,
    )
    det_pids = np.array([1, 2, 3, 3, 10, 20, 30, 99], dtype=np.uint64)
    det_dists = np.array([210.0, 200.0, 220.0, 300.0, 150.0, 200.0, 90.0, 100.0], dtype=np.float32)

    with h5py.File(hdf_path, "w") as h5_file:
        inter_grp = h5_file.create_group("interactions")
        sim_vol = inter_grp.create_group("Simulation_volume")
        sim_vol.create_dataset("particle_ID", data=sim_pids)
        sim_vol.create_dataset("distance_traveled", data=sim_dists)
        proc_ds = sim_vol.create_dataset("process_code", data=np.ones_like(sim_pids, dtype=np.uint8))
        proc_ds.attrs["mapping"] = '{"0": "PhotoelectricEffect", "1": "ComptonScattering"}'

        det_grp = inter_grp.create_group("Detector_1")
        det_grp.create_dataset("particle_ID", data=det_pids)
        det_grp.create_dataset("distance_traveled", data=det_dists)
        det_grp.create_dataset(
            "energy_deposit",
            data=np.full(det_pids.size, 0.050, dtype=np.float32),
        )
        det_grp.create_dataset(
            "local_position",
            data=np.zeros((det_pids.size, 3), dtype=np.float32),
        )

    extractor = SimulationDataExtractor(hdf_path)
    extractor.extract_data("Detector_1")
    orders = extractor.get_scattering_orders("Detector_1")
    expected_orders = [
        int(np.sum((sim_pids == pid_val) & (sim_dists < dist_val)))
        for pid_val, dist_val in zip(det_pids, det_dists, strict=True)
    ]
    assert list(orders) == expected_orders
    assert list(orders[:4]) == [1, 0, 0, 2]

    # Проверка DICOM NM при num_frames == 1 и при направлении вращения CW
    single_dcm_path = tmp_path / "single_frame.dcm"
    DicomNMExporter(single_dcm_path).export(
        projections=np.ones((1, 8, 8), dtype=np.float64) * 5.0,
        angles=np.array([0.0 * deg], dtype=np.float64),
        pixel_size=4.2 * mm,
        orbit_radius=250.0 * mm,
    )
    single_ds = pydicom.dcmread(single_dcm_path)
    assert int(single_ds.NumberOfDetectors) == 1
    assert len(single_ds.DetectorInformationSequence) == 1
    assert np.atleast_1d(single_ds.DetectorVector).tolist() == [1]

    cw_dcm_path = tmp_path / "cw_frames.dcm"
    cw_angles = np.array([0.0, 330.0, 300.0, 270.0], dtype=np.float64) * deg
    DicomNMExporter(cw_dcm_path).export(
        projections=np.ones((4, 8, 8), dtype=np.float64) * 10.0,
        angles=cw_angles,
        pixel_size=4.2 * mm,
        orbit_radius=250.0 * mm,
    )
    cw_ds = pydicom.dcmread(cw_dcm_path)
    assert int(cw_ds.NumberOfDetectors) == 2
    assert len(cw_ds.DetectorInformationSequence) == 2
    assert cw_ds.RotationInformationSequence[0].RotationDirection == "CW"
