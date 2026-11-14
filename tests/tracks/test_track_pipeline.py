"""Модульные и интеграционные тесты конвейера 3D-треков частиц (nmspect.tracks) и экспорта VTK (vtk_tracks)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import vtk
from hepunits import cm, deg, keV, mm, ns, rad

from nmspect.detector import ProjectionBuilder
from nmspect.exporters.vtk_tracks import VtkTrackExporter
from nmspect.reporting import TrackStudyReportGenerator
from nmspect.spectrometry import TB161_PEAK_75_WINDOW, PhotonComponent
from nmspect.tracks import (
    COHERENT_PROCESS_CODE,
    COMPTON_PROCESS_CODE,
    CRYSTAL_NODE_FLAG,
    EMISSION_PROCESS_CODE,
    EMISSION_VOLUME_ID,
    EXCLUSIVE_TOPOLOGY_NAMES,
    PHOTOELECTRIC_PROCESS_CODE,
    SPEED_OF_LIGHT,
    AuxiliaryMaskRule,
    CollimatorTrackAnnotator,
    FieldCondition,
    HdfTrackAssembler,
    ProcessCounterAnnotator,
    ScintillationTrackAnnotator,
    StreamingTrackStudyAnalyzer,
    TaxonomyScheme,
    TrackBatch,
    TrackClassRule,
    TrackProcessingPipeline,
    TrackStudyConfig,
    TrackStudySummary,
    classify_tracks,
    compute_angular_psf_metrics,
    compute_azimuthal_anisotropy,
    compute_doi_roc_curve,
    compute_interdetector_crosstalk,
    extract_detector_events_from_tracks,
)


def _build_sample_track_batch() -> TrackBatch:
    """Создаёт тестовый пакет из трёх треков разной длины (2, 3 и 4 узла) для проверки CSR-операций."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 15.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 7.0, 0.0],
            [2.0, 12.0, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 5, 9], dtype=np.int64)
    point_data = {
        "energy_deposit": np.arange(9, dtype=np.float32) * float(keV),
        "process_code": np.array([-1, 0, -1, 1, 0, -1, 1, 2, 0], dtype=np.int32),
    }
    cell_data = {
        "particle_id": np.array([101, 202, 303], dtype=np.int64),
        "detector_id": np.array([1, 2, 1], dtype=np.uint8),
    }
    return TrackBatch(
        points=points,
        offsets=offsets,
        point_data=point_data,
        cell_data=cell_data,
    )


def test_track_batch_invariants_slice_and_concat() -> None:
    """Проверяет структурные инварианты TrackBatch, векторную фильтрацию slice_tracks и объединение concat."""
    # 1. Валидация форм координат и CSR-сдвигов
    with pytest.raises(ValueError, match="points"):
        TrackBatch(
            points=np.zeros((4, 2), dtype=np.float32),
            offsets=np.array([0, 2, 4], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="offsets"):
        TrackBatch(
            points=np.zeros((2, 3), dtype=np.float32),
            offsets=np.array([], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="offsets\\[0\\]"):
        TrackBatch(
            points=np.zeros((2, 3), dtype=np.float32),
            offsets=np.array([1, 3], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="offsets\\[-1\\]"):
        TrackBatch(
            points=np.zeros((3, 3), dtype=np.float32),
            offsets=np.array([0, 2], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="не менее 2 узлов"):
        TrackBatch(
            points=np.zeros((3, 3), dtype=np.float32),
            offsets=np.array([0, 1, 3], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="point_data"):
        TrackBatch(
            points=np.zeros((2, 3), dtype=np.float32),
            offsets=np.array([0, 2], dtype=np.int64),
            point_data={"invalid_field": np.zeros(5, dtype=np.float32)},
        )

    with pytest.raises(ValueError, match="cell_data"):
        TrackBatch(
            points=np.zeros((2, 3), dtype=np.float32),
            offsets=np.array([0, 2], dtype=np.int64),
            cell_data={"invalid_cell": np.zeros(3, dtype=np.int32)},
        )

    # 2. Проверка свойств и slice_tracks по индексам, булевой маске и пустой выборке
    batch = _build_sample_track_batch()
    assert batch.num_tracks == 3
    assert batch.num_points == 9
    assert np.array_equal(batch.track_lengths, np.array([2, 3, 4], dtype=np.int64))

    sliced_by_idx = batch.slice_tracks([2, 0])
    assert sliced_by_idx.num_tracks == 2
    assert sliced_by_idx.num_points == 6
    assert np.array_equal(sliced_by_idx.offsets, np.array([0, 4, 6], dtype=np.int64))
    assert np.array_equal(sliced_by_idx.cell_data["particle_id"], np.array([303, 101], dtype=np.int64))
    assert np.allclose(sliced_by_idx.points[:4], batch.points[5:9])
    assert np.allclose(sliced_by_idx.points[4:], batch.points[0:2])

    bool_mask = np.array([False, True, True], dtype=np.bool_)
    sliced_by_mask = batch.slice_tracks(bool_mask)
    assert sliced_by_mask.num_tracks == 2
    assert sliced_by_mask.num_points == 7
    assert np.array_equal(sliced_by_mask.offsets, np.array([0, 3, 7], dtype=np.int64))
    assert np.array_equal(sliced_by_mask.cell_data["particle_id"], np.array([202, 303], dtype=np.int64))

    empty_slice = batch.slice_tracks([])
    assert empty_slice.num_tracks == 0
    assert empty_slice.num_points == 0
    assert np.array_equal(empty_slice.offsets, np.array([0], dtype=np.int64))
    assert set(empty_slice.point_data.keys()) == set(batch.point_data.keys())
    assert set(empty_slice.cell_data.keys()) == set(batch.cell_data.keys())

    empty_mask_slice = batch.slice_tracks(np.array([False, False, False], dtype=np.bool_))
    assert empty_mask_slice.num_tracks == 0
    assert empty_mask_slice.num_points == 0

    with pytest.raises(IndexError):
        batch.slice_tracks([3])

    with pytest.raises(ValueError, match="Булева маска"):
        batch.slice_tracks(np.array([True, False], dtype=np.bool_))

    # 3. Проверка объединения пакетов TrackBatch.concat
    part_first = batch.slice_tracks([0, 1])
    part_second = batch.slice_tracks([2])
    reconstructed = TrackBatch.concat([part_first, empty_slice, part_second])
    assert reconstructed.num_tracks == batch.num_tracks
    assert reconstructed.num_points == batch.num_points
    assert np.allclose(reconstructed.points, batch.points)
    assert np.array_equal(reconstructed.offsets, batch.offsets)
    assert np.array_equal(reconstructed.cell_data["particle_id"], batch.cell_data["particle_id"])
    assert np.allclose(reconstructed.point_data["energy_deposit"], batch.point_data["energy_deposit"])

    with pytest.raises(ValueError, match="не может быть пустой"):
        TrackBatch.concat([])

    mismatched_batch = TrackBatch(
        points=np.zeros((2, 3), dtype=np.float32),
        offsets=np.array([0, 2], dtype=np.int64),
        point_data={"other_key": np.zeros(2, dtype=np.float32)},
        cell_data=part_second.cell_data,
    )
    with pytest.raises(ValueError, match="point_data"):
        TrackBatch.concat([part_first, mismatched_batch])


def test_annotators_on_deterministic_synthetic_batch() -> None:
    """Проверяет ProcessCounterAnnotator, ScintillationTrackAnnotator и CollimatorTrackAnnotator на известных треках."""
    # Трек 0 (7 узлов): строго перпендикулярен плоскости кристалла Detector_1 вдоль оси Y (x = 0, z = 0)
    # Трек 1 (6 узлов): наклонный трек, пересекающий перегородки гексагонального коллиматора
    points = (
        np.array(
            [
                # Трек 0: рождение -> фантом (комптон, когерент) -> коллиматор (когерент) -> кристалл (комптон, когерент, фото)
                [0.0, 0.0, 0.0],
                [0.0, 50.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 200.0, 0.0],
                [0.0, 246.0, 0.0],
                [0.0, 248.0, 0.0],
                [0.0, 250.0, 0.0],
                # Трек 1: рождение -> фантом (комптон) -> коллиматор (комптон) -> световод (комптон) -> кристалл (комптон, фото)
                [-30.0, 0.0, 0.0],
                [-25.0, 100.0, 0.0],
                [-20.0, 210.0, 0.0],
                [-15.0, 200.0, 0.0],
                [20.0, 246.0, 10.0],
                [20.0, 249.0, 10.0],
            ],
            dtype=np.float32,
        )
        * float(mm)
    ).astype(np.float32)

    offsets = np.array([0, 7, 13], dtype=np.int64)

    process_code = np.array(
        [
            EMISSION_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            COHERENT_PROCESS_CODE,
            COHERENT_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            COHERENT_PROCESS_CODE,
            PHOTOELECTRIC_PROCESS_CODE,
            EMISSION_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            COMPTON_PROCESS_CODE,
            PHOTOELECTRIC_PROCESS_CODE,
        ],
        dtype=np.int32,
    )

    volume_id = np.array(
        [
            EMISSION_VOLUME_ID,
            1,
            1,
            4,
            5,
            5,
            5,
            EMISSION_VOLUME_ID,
            1,
            10,
            7,
            5,
            5,
        ],
        dtype=np.int32,
    )

    material_id = np.array(
        [0, 2, 2, 10, 15, 15, 15, 0, 2, 10, 25, 15, 15],
        dtype=np.int32,
    )
    atomic_number = np.array(
        [0, 8, 8, 82, 53, 53, 53, 0, 8, 82, 14, 53, 53],
        dtype=np.int32,
    )
    is_crystal_node = np.array(
        [
            0,
            0,
            0,
            0,
            CRYSTAL_NODE_FLAG,
            CRYSTAL_NODE_FLAG,
            CRYSTAL_NODE_FLAG,
            0,
            0,
            0,
            0,
            CRYSTAL_NODE_FLAG,
            CRYSTAL_NODE_FLAG,
        ],
        dtype=np.uint8,
    )

    energy_deposit = (
        np.array(
            [
                0.0,
                10.0,
                0.0,
                0.0,
                40.0,
                0.0,
                100.0,
                0.0,
                5.0,
                15.0,
                10.0,
                30.0,
                60.0,
            ],
            dtype=np.float32,
        )
        * float(keV)
    ).astype(np.float32)

    distance_traveled = (
        np.array(
            [
                0.0,
                50.0,
                100.0,
                200.0,
                246.0,
                248.0,
                250.0,
                0.0,
                100.0,
                210.0,
                260.0,
                310.0,
                314.0,
            ],
            dtype=np.float32,
        )
        * float(mm)
    ).astype(np.float32)

    emission_time = (np.array([10.0, 25.0], dtype=np.float32) * float(ns)).astype(np.float32)

    batch = TrackBatch(
        points=points,
        offsets=offsets,
        point_data={
            "process_code": process_code,
            "volume_id": volume_id,
            "material_id": material_id,
            "atomic_number": atomic_number,
            "is_crystal_node": is_crystal_node,
            "energy_deposit": energy_deposit,
            "distance_traveled": distance_traveled,
        },
        cell_data={
            "particle_id": np.array([1, 2], dtype=np.int64),
            "detector_id": np.array([1, 1], dtype=np.uint8),
            "projection_index": np.array([0, 0], dtype=np.int32),
            "gantry_angle": np.array([0.0 * rad, 0.0 * rad], dtype=np.float32),
            "emission_time": emission_time,
        },
    )

    # 1. Проверка ProcessCounterAnnotator (все 8 счётчиков uint8)
    counter_annotator = ProcessCounterAnnotator()
    counter_annotator.annotate(batch)

    expected_counters = {
        "phantom_compton": [1, 1],
        "phantom_coherent": [1, 0],
        "collimator_compton": [0, 1],
        "collimator_coherent": [1, 0],
        "backscatter_count": [0, 1],
        "crystal_compton": [1, 1],
        "crystal_coherent": [1, 0],
        "crystal_photoelectric": [1, 1],
    }
    for counter_name, expected_values in expected_counters.items():
        actual_array = batch.cell_data[counter_name]
        assert actual_array.dtype == np.uint8
        assert list(actual_array) == expected_values

    # 2. Проверка ScintillationTrackAnnotator без шума (точное совпадение физических формул)
    noiseless_scint = ScintillationTrackAnnotator(
        energy_resolution=0.0,
        reference_energy=140.5 * keV,
        intrinsic_resolution=0.0 * mm,
        random_seed=42,
    )
    noiseless_scint.annotate(batch)

    expected_deposited = np.array([140.0 * keV, 90.0 * keV], dtype=np.float32)
    assert np.allclose(batch.cell_data["deposited_energy"], expected_deposited, rtol=1e-5)
    assert np.allclose(batch.cell_data["measured_energy"], expected_deposited, rtol=1e-5)

    expected_y_first = (40.0 * 246.0 + 100.0 * 250.0) / 140.0 * float(mm)
    expected_y_second = (30.0 * 246.0 + 60.0 * 249.0) / 90.0 * float(mm)
    expected_scint_pos = np.array(
        [
            [0.0, expected_y_first, 0.0],
            [20.0 * float(mm), expected_y_second, 10.0 * float(mm)],
        ],
        dtype=np.float32,
    )
    assert np.allclose(batch.cell_data["scintillation_position"], expected_scint_pos, atol=1e-4)
    assert np.allclose(batch.cell_data["registered_position"], expected_scint_pos, atol=1e-4)
    assert batch.cell_data["crystal_depth"].dtype == np.float32
    assert np.all((batch.cell_data["crystal_depth"] >= 0.0) & (batch.cell_data["crystal_depth"] <= 9.5 * float(mm)))
    assert batch.cell_data["detector_local_position"].shape == (2, 2)
    assert np.allclose(
        batch.cell_data["detector_local_position"],
        np.array([[0.0, 0.0], [20.0 * float(mm), 10.0 * float(mm)]], dtype=np.float32),
        atol=1e-4,
    )

    expected_path_first = (40.0 * 246.0 + 100.0 * 250.0) / 140.0 * float(mm)
    expected_path_second = (30.0 * 310.0 + 60.0 * 314.0) / 90.0 * float(mm)
    expected_scint_time = np.array(
        [
            10.0 * float(ns) + expected_path_first / SPEED_OF_LIGHT,
            25.0 * float(ns) + expected_path_second / SPEED_OF_LIGHT,
        ],
        dtype=np.float32,
    )
    assert np.allclose(batch.cell_data["scintillation_time"], expected_scint_time, rtol=1e-5)

    # 3. Проверка CollimatorTrackAnnotator (septal_crossings == 0 для прямого трека и > 0 для наклонного)
    collimator_annotator = CollimatorTrackAnnotator(
        hole_diameter=1.5 * mm,
        septa_thickness=0.2 * mm,
        hole_length=35.0 * mm,
        orbit_radius=250.0 * mm,
        crystal_thickness=9.5 * mm,
    )
    collimator_annotator.annotate(batch)

    septal_crossings = batch.cell_data["septal_crossings"]
    tan_theta = batch.cell_data["tan_theta"]
    tan_theta_axial = batch.cell_data["tan_theta_axial"]
    assert septal_crossings.dtype == np.uint8
    assert tan_theta.dtype == np.float32
    assert tan_theta_axial.dtype == np.float32
    assert int(septal_crossings[0]) == 0
    assert int(septal_crossings[1]) > 0
    assert np.isclose(float(tan_theta[0]), 0.0, atol=1e-6)
    assert np.isclose(float(tan_theta_axial[0]), 0.0, atol=1e-6)
    assert float(tan_theta[1]) > 0.0
    assert float(tan_theta_axial[1]) > 0.0

    # Проверка сдвига центра коллиматора, если передана только матрица Detector_1 без Collimator_1
    detector_only_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 250.0 * float(mm)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    collimator_annotator.annotate(batch, metadata={"volume_matrices": {"Detector_1": detector_only_pose}})
    assert int(batch.cell_data["septal_crossings"][0]) == 0
    assert int(batch.cell_data["septal_crossings"][1]) > 0

    # 4. Проверка стохастического размытия в ScintillationTrackAnnotator
    detector_pose = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 250.0 * float(mm)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    smeared_scint = ScintillationTrackAnnotator(
        energy_resolution=0.099,
        reference_energy=140.5 * keV,
        intrinsic_resolution=3.8 * mm,
        random_seed=123,
    )
    smeared_scint.annotate(batch, metadata={"volume_matrices": {"Detector_1": detector_pose}})
    assert np.all(batch.cell_data["measured_energy"] > 0.0)
    assert not np.allclose(batch.cell_data["measured_energy"], batch.cell_data["deposited_energy"])
    assert not np.allclose(batch.cell_data["registered_position"], batch.cell_data["scintillation_position"])
    assert np.allclose(
        batch.cell_data["registered_position"][:, 1],
        batch.cell_data["scintillation_position"][:, 1],
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "relative_hdf_path",
    [
        "data/raw/output_tc99m_dual_cam/0.0_deg.hdf",
        "data/raw/output_tb161_dual_cam/0.0_deg.hdf",
    ],
)
def test_end_to_end_track_pipeline_and_vtk_export_on_real_hdf(
    relative_hdf_path: str,
    tmp_path: Path,
) -> None:
    """Проверяет сквозной конвейер TrackProcessingPipeline и VtkTrackExporter на реальных файлах Tc-99m и Tb-161."""
    hdf_path = Path(relative_hdf_path)
    if not hdf_path.is_file():
        pytest.skip(f"Реальный файл симуляции отсутствует на диске: {hdf_path}")

    pipeline = TrackProcessingPipeline(max_tracks_per_detector=200, random_seed=42)
    tracks, metadata = pipeline.process_file(hdf_path)

    assert tracks.num_tracks > 0
    assert tracks.num_tracks <= 400
    assert tracks.num_points >= 2 * tracks.num_tracks
    assert isinstance(metadata, dict)

    expected_cell_fields = (
        "particle_id",
        "detector_id",
        "projection_index",
        "gantry_angle",
        "emission_energy",
        "emission_time",
        "phantom_compton",
        "phantom_coherent",
        "collimator_compton",
        "collimator_coherent",
        "backscatter_count",
        "crystal_compton",
        "crystal_coherent",
        "crystal_photoelectric",
        "deposited_energy",
        "measured_energy",
        "has_scintillation",
        "scintillation_time",
        "scintillation_position",
        "registered_position",
        "crystal_depth",
        "detector_local_position",
        "septal_crossings",
        "tan_theta",
        "tan_theta_axial",
        "topology_class",
        "photon_component",
    )
    for field_name in expected_cell_fields:
        assert field_name in tracks.cell_data
        assert tracks.cell_data[field_name].shape[0] == tracks.num_tracks

    exporter = VtkTrackExporter()

    # 1. Сквозная конвертация в файл полилиний .vtp и обратное чтение через vtkXMLPolyDataReader
    vtp_output_path = tmp_path / f"{hdf_path.parent.name}_tracks.vtp"
    saved_vtp_path = exporter.convert_hdf_to_vtk(
        input_path=hdf_path,
        output_path=vtp_output_path,
        pipeline=TrackProcessingPipeline(max_tracks_per_detector=200, random_seed=42),
        as_scene_vtm=False,
    )
    assert saved_vtp_path.is_file()

    poly_reader = vtk.vtkXMLPolyDataReader()
    poly_reader.SetFileName(str(saved_vtp_path))
    poly_reader.Update()
    read_poly = poly_reader.GetOutput()

    assert read_poly.GetNumberOfPoints() == tracks.num_points
    assert read_poly.GetNumberOfCells() == tracks.num_tracks
    assert read_poly.GetPointData().GetArray("energy_deposit") is not None
    assert read_poly.GetPointData().GetArray("process_code") is not None
    assert read_poly.GetCellData().GetArray("measured_energy") is not None
    assert read_poly.GetCellData().GetArray("septal_crossings") is not None

    # 2. Сквозная конвертация в мультиблочную 3D-сцену .vtm и обратное чтение через vtkXMLMultiBlockDataReader
    vtm_output_path = tmp_path / f"{hdf_path.parent.name}_scene.vtm"
    saved_vtm_path = exporter.convert_hdf_to_vtk(
        input_path=hdf_path,
        output_path=vtm_output_path,
        pipeline=TrackProcessingPipeline(max_tracks_per_detector=200, random_seed=42),
        as_scene_vtm=True,
    )
    assert saved_vtm_path.is_file()

    multiblock_reader = vtk.vtkXMLMultiBlockDataReader()
    multiblock_reader.SetFileName(str(saved_vtm_path))
    multiblock_reader.Update()
    read_multiblock = multiblock_reader.GetOutput()

    assert read_multiblock.GetNumberOfBlocks() == 9

    tracks_block = read_multiblock.GetBlock(0)
    assert isinstance(tracks_block, vtk.vtkPolyData)
    assert tracks_block.GetNumberOfPoints() == tracks.num_points
    assert tracks_block.GetNumberOfCells() == tracks.num_tracks

    hits_block = read_multiblock.GetBlock(1)
    assert isinstance(hits_block, vtk.vtkPolyData)
    assert hits_block.GetNumberOfPoints() == tracks.num_tracks
    assert hits_block.GetPointData().GetArray("measured_energy") is not None


def test_process_counter_volume_routing_and_post_escape_cutoff() -> None:
    """Проверяет маршрутизацию воздушных каналов коллиматора (vol 5, 11), бокса (vol 3, 9) и отсечение послевылетных узлов."""
    points = np.zeros((6, 3), dtype=np.float32)
    offsets = np.array([0, 6], dtype=np.int64)
    point_data = {
        "process_code": np.array(
            [
                EMISSION_PROCESS_CODE,
                COMPTON_PROCESS_CODE,
                COHERENT_PROCESS_CODE,
                COMPTON_PROCESS_CODE,
                COMPTON_PROCESS_CODE,
                PHOTOELECTRIC_PROCESS_CODE,
            ],
            dtype=np.int32,
        ),
        "volume_id": np.array([EMISSION_VOLUME_ID, 0, 5, 3, 6, 8], dtype=np.int32),
        "material_id": np.array([0, 4, 4, 4, 45, 108], dtype=np.int32),
        "atomic_number": np.array([0, 7, 7, 7, 53, 82], dtype=np.int32),
        "is_crystal_node": np.array([0, 0, 0, 0, CRYSTAL_NODE_FLAG, 0], dtype=np.uint8),
    }
    batch = TrackBatch(points=points, offsets=offsets, point_data=point_data)
    annotator = ProcessCounterAnnotator()
    annotator.annotate(batch)

    assert int(batch.cell_data["phantom_compton"][0]) == 1
    assert int(batch.cell_data["collimator_coherent"][0]) == 1
    assert int(batch.cell_data["backscatter_count"][0]) == 1
    assert int(batch.cell_data["crystal_compton"][0]) == 1
    assert int(batch.cell_data["crystal_photoelectric"][0]) == 0

    strict_phantom_annotator = ProcessCounterAnnotator(phantom_volume_ids=(1,))
    strict_phantom_annotator.annotate(batch)
    assert int(batch.cell_data["phantom_compton"][0]) == 0



def _create_synthetic_track_hdf5(
    file_path: Path,
    *,
    gantry_angle_value: float = 0.0,
    position_units: str = "mm",
    energy_units: str = "keV",
    include_second_head: bool = True,
) -> Path:
    """Формирует компактный синтетический HDF5-файл с треками, междетекторным перелётом и нулевым вкладом."""
    pos_div = float(cm / mm) if position_units == "cm" else 1.0
    ene_scale = 1.0 if energy_units == "keV" else 1e-3

    with h5py.File(file_path, "w") as hdf_file:
        meta_grp = hdf_file.create_group("metadata")
        gantry_grp = meta_grp.create_group("gantry")
        gantry_grp.create_dataset("gantry_angle", data=gantry_angle_value)
        gantry_grp.create_dataset("gantry_angle_units", data="deg")
        gantry_grp.create_dataset("orbit_radius", data=250.0)

        angle_val = gantry_angle_value * float(deg)
        cos_a = float(np.cos(angle_val))
        sin_a = float(np.sin(angle_val))
        pose_det1 = np.array(
            [
                [cos_a, 0.0, -sin_a, -250.0 * sin_a],
                [sin_a, 0.0, cos_a, 250.0 * cos_a],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        vol_grp = meta_grp.create_group("volume_matrices")
        vol_grp.create_dataset("Detector_1", data=pose_det1)
        if include_second_head:
            pose_det2 = pose_det1.copy()
            pose_det2[:3, :3] *= -1.0
            pose_det2[2, 1] = 1.0
            pose_det2[:3, 3] = -pose_det1[:3, 3]
            vol_grp.create_dataset("Detector_2", data=pose_det2)

        init_grp = hdf_file.create_group("initial_states")
        init_ids = np.array([10, 20, 30], dtype=np.int64)
        init_pos = (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-5.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            / pos_div
        )
        init_dir = np.array(
            [
                [-sin_a, cos_a, 0.0],
                [-sin_a, cos_a, 0.0],
                [-sin_a, cos_a, 0.0],
            ],
            dtype=np.float64,
        )
        init_ene = np.array([74.6, 48.9, 300.0], dtype=np.float64) * ene_scale
        init_time = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        init_grp.create_dataset("particle_ID", data=init_ids)
        ds_ip = init_grp.create_dataset("emission_position", data=init_pos)
        ds_ip.attrs["units"] = position_units
        init_grp.create_dataset("emission_direction", data=init_dir)
        ds_ie = init_grp.create_dataset("emission_energy", data=init_ene)
        ds_ie.attrs["units"] = energy_units
        ds_it = init_grp.create_dataset("emission_time", data=init_time)
        ds_it.attrs["units"] = "ns"

        inter_grp = hdf_file.create_group("interactions")
        det1_grp = inter_grp.create_group("Detector_1")
        # Частица 10: прямое поглощение; частица 20: упругий Рэлей в кристалле (0 кэВ); частица 30: перелёт в Det_2
        d1_ids = np.array([10, 20, 30], dtype=np.int64)
        d1_pos = (
            np.array(
                [
                    [-247.0 * sin_a, 247.0 * cos_a, 0.0],
                    [-248.0 * sin_a, 248.0 * cos_a, 0.0],
                    [-252.0 * sin_a, 252.0 * cos_a, 0.0],
                ],
                dtype=np.float64,
            )
            / pos_div
        )
        d1_dir = np.array(
            [
                [-sin_a, cos_a, 0.0],
                [-sin_a, cos_a, 0.0],
                [sin_a, -cos_a, 0.0],
            ],
            dtype=np.float64,
        )
        d1_edep = np.array([74.6, 0.0, 45.0], dtype=np.float64) * ene_scale
        d1_dist = np.array([247.0, 248.0, 252.0], dtype=np.float64) / pos_div
        det1_grp.create_dataset("particle_ID", data=d1_ids)
        ds_p1 = det1_grp.create_dataset("global_position", data=d1_pos)
        ds_p1.attrs["units"] = position_units
        det1_grp.create_dataset("global_direction", data=d1_dir)
        ds_e1 = det1_grp.create_dataset("energy_deposit", data=d1_edep)
        ds_e1.attrs["units"] = energy_units
        ds_d1 = det1_grp.create_dataset("distance_traveled", data=d1_dist)
        ds_d1.attrs["units"] = position_units
        det1_grp.create_dataset("process_code", data=np.array([3, 1, 2], dtype=np.int32))
        det1_grp.create_dataset("volume_id", data=np.array([6, 6, 6], dtype=np.int32))
        det1_grp.create_dataset("material_id", data=np.array([45, 45, 45], dtype=np.int32))
        det1_grp.create_dataset("Z", data=np.array([53, 53, 53], dtype=np.int32))
        det1_grp.create_dataset("species_code", data=np.array([0, 0, 0], dtype=np.int32))

        if include_second_head:
            det2_grp = inter_grp.create_group("Detector_2")
            det2_grp.create_dataset("particle_ID", data=np.array([30], dtype=np.int64))
            ds_p2 = det2_grp.create_dataset(
                "global_position",
                data=np.array([[249.0 * sin_a, -249.0 * cos_a, 0.0]], dtype=np.float64) / pos_div,
            )
            ds_p2.attrs["units"] = position_units
            det2_grp.create_dataset("global_direction", data=np.array([[sin_a, -cos_a, 0.0]], dtype=np.float64))
            ds_e2 = det2_grp.create_dataset("energy_deposit", data=np.array([75.0], dtype=np.float64) * ene_scale)
            ds_e2.attrs["units"] = energy_units
            ds_d2 = det2_grp.create_dataset("distance_traveled", data=np.array([750.0], dtype=np.float64) / pos_div)
            ds_d2.attrs["units"] = position_units
            det2_grp.create_dataset("process_code", data=np.array([3], dtype=np.int32))
            det2_grp.create_dataset("volume_id", data=np.array([12], dtype=np.int32))
            det2_grp.create_dataset("material_id", data=np.array([45], dtype=np.int32))
            det2_grp.create_dataset("Z", data=np.array([53], dtype=np.int32))
            det2_grp.create_dataset("species_code", data=np.array([0], dtype=np.int32))

    return file_path


def test_assembler_units_crosstalk_zero_deposit_and_single_head_vtk(tmp_path: Path) -> None:
    """Проверяет чтение @units (cm, MeV), учёт междетекторного перелёта, нулевой вклад и single-head VTK."""
    hdf_dual = _create_synthetic_track_hdf5(
        tmp_path / "0.0_deg.hdf",
        gantry_angle_value=0.0,
        position_units="cm",
        energy_units="MeV",
        include_second_head=True,
    )
    pipeline = TrackProcessingPipeline(random_seed=7)
    tracks, _meta = pipeline.process_file(hdf_dual)
    assert tracks.num_tracks == 4

    # Проверка трека с нулевым энерговыделением (particle_id == 20): нет ложного размытия ФЭУ и энергии
    zero_mask = tracks.cell_data["particle_id"] == 20
    assert int(np.count_nonzero(zero_mask)) == 1
    assert int(tracks.cell_data["has_scintillation"][zero_mask][0]) == 0
    assert float(tracks.cell_data["measured_energy"][zero_mask][0]) == 0.0
    assert np.allclose(
        tracks.cell_data["registered_position"][zero_mask],
        tracks.cell_data["scintillation_position"][zero_mask],
    )

    # Частица 30 должна иметь во втором детекторе полный маршрут из обоих кристаллов (3 узла: эмиссия + Det1 + Det2)
    crosstalk = compute_interdetector_crosstalk(tracks)
    assert crosstalk.shared_particles == 1
    assert crosstalk.shared_track_count == 2
    assert int(crosstalk.pair_matrix[0, 1]) == 1

    # Проверка фильтрации нулевого вклада require_energy_deposit=True
    strict_assembler = HdfTrackAssembler(require_energy_deposit=True, random_seed=7)
    positive_tracks, _ = strict_assembler.assemble_file(hdf_dual)
    assert positive_tracks.num_tracks == 3

    # Проверка одноголовочного экспорта VtkTrackExporter (ровно 6 блоков в .vtm вместо 9)
    hdf_single = _create_synthetic_track_hdf5(
        tmp_path / "single_0.0_deg.hdf",
        gantry_angle_value=0.0,
        include_second_head=False,
    )
    single_tracks, single_meta = pipeline.process_file(hdf_single)
    exporter = VtkTrackExporter()
    vtm_single = exporter.export_scene_vtm(
        tracks=single_tracks,
        metadata=single_meta,
        output_path=tmp_path / "single_scene.vtm",
    )
    mb_reader = vtk.vtkXMLMultiBlockDataReader()
    mb_reader.SetFileName(str(vtm_single))
    mb_reader.Update()
    assert mb_reader.GetOutput().GetNumberOfBlocks() == 6

    filtered_hits = exporter.build_registered_hits_polydata(tracks, only_positive_deposit=True)
    assert filtered_hits.GetNumberOfPoints() == 3


def test_multiview_frame_rotation_and_shared_rng(tmp_path: Path) -> None:
    """Проверяет поворот репера R_z(Delta_phi) на многоракурсном пакете и отсутствие повторения ГСЧ между файлами."""
    _create_synthetic_track_hdf5(tmp_path / "0.0_deg.hdf", gantry_angle_value=0.0)
    _create_synthetic_track_hdf5(tmp_path / "90.0_deg.hdf", gantry_angle_value=90.0)

    pipeline = TrackProcessingPipeline(random_seed=99)
    combined_tracks, primary_meta = pipeline.process_directory(tmp_path)

    # Повторное аннотирование склеенного пакета с базовыми метаданными 0 град обязано сохранить нулевой наклон ракурса 90 град
    CollimatorTrackAnnotator().annotate(combined_tracks, metadata=primary_meta)
    p10_mask = combined_tracks.cell_data["particle_id"] == 10
    assert np.allclose(combined_tracks.cell_data["tan_theta"][p10_mask], 0.0, atol=1e-5)

    # Сравнение размытых энергий на двух идентичных файлах: общий генератор не должен дублировать случайный шум
    sub_dir = tmp_path / "rng_pair"
    sub_dir.mkdir()
    hdf_a = _create_synthetic_track_hdf5(sub_dir / "copy_a.hdf", gantry_angle_value=0.0)
    hdf_b = _create_synthetic_track_hdf5(sub_dir / "copy_b.hdf", gantry_angle_value=0.0)
    seq_pipeline = TrackProcessingPipeline(random_seed=12345)
    batch_a, _ = seq_pipeline.process_file(hdf_a)
    batch_b, _ = seq_pipeline.process_file(hdf_b)
    pos_a = batch_a.cell_data["measured_energy"][batch_a.cell_data["deposited_energy"] > 0.0]
    pos_b = batch_b.cell_data["measured_energy"][batch_b.cell_data["deposited_energy"] > 0.0]
    assert not np.allclose(pos_a, pos_b)


def test_track_analytics_and_streaming_study_analyzer(tmp_path: Path) -> None:
    """Проверяет метрики ФРТ, 6-лучевую анизотропию, монотонность AUC по DOI и потоковый анализатор."""
    rng = np.random.default_rng(42)
    tan_u = rng.normal(0.0, 0.02, size=4000)
    tan_v = rng.normal(0.0, 0.02, size=4000)
    psf_metrics = compute_angular_psf_metrics(tan_u, tan_v)
    assert psf_metrics.count == 4000.0
    assert 0.035 <= psf_metrics.fwhm_tan <= 0.060
    assert psf_metrics.fwtm_tan > psf_metrics.fwhm_tan

    # Синтетическая 6-лучевая звезда септ
    angles = rng.uniform(0.0, 2.0 * np.pi, size=12000)
    keep_prob = 0.5 + 0.45 * np.cos(6.0 * angles)
    chosen = rng.uniform(0.0, 1.0, size=12000) < keep_prob
    star_r = np.full(int(np.count_nonzero(chosen)), 0.12)
    star_u = star_r * np.cos(angles[chosen])
    star_v = star_r * np.sin(angles[chosen])
    aniso = compute_azimuthal_anisotropy(star_u, star_v, radial_min=0.05, radial_max=0.20)
    assert aniso.dominant_harmonic_order == 6
    assert aniso.peak_to_valley_ratio > 3.0

    # Проверка строгой математической монотонности AUC(FWHM=0) >= AUC(1.5 мм) >= AUC(3.0 мм)
    depth_edges = np.linspace(0.0, 9.5, 191, dtype=np.float64) * float(mm)
    centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    sig_hist = np.exp(-centers / (0.8 * float(mm))) * 10000.0
    bkg_hist = np.exp(-(9.5 * float(mm) - centers) / (1.2 * float(mm))) * 5000.0
    roc_0 = compute_doi_roc_curve(sig_hist, bkg_hist, depth_edges, depth_resolution_fwhm=0.0 * mm)
    roc_15 = compute_doi_roc_curve(sig_hist, bkg_hist, depth_edges, depth_resolution_fwhm=1.5 * mm)
    roc_30 = compute_doi_roc_curve(sig_hist, bkg_hist, depth_edges, depth_resolution_fwhm=3.0 * mm)
    assert roc_0.auc > roc_15.auc > roc_30.auc > 0.90

    # Потоковый анализатор StreamingTrackStudyAnalyzer + сериализация бандла
    study_dir = tmp_path / "study_views"
    study_dir.mkdir()
    _create_synthetic_track_hdf5(study_dir / "0.0_deg.hdf", gantry_angle_value=0.0)
    _create_synthetic_track_hdf5(study_dir / "45.0_deg.hdf", gantry_angle_value=45.0)

    cfg = TrackStudyConfig.for_isotope("161Tb", vtk_sample_per_view=10, random_seed=11)
    analyzer = StreamingTrackStudyAnalyzer(config=cfg)
    summary = analyzer.analyze_directory(study_dir)
    assert summary.num_files == 2
    assert summary.num_detectors == 2
    assert summary.num_views == 4
    assert summary.total_tracks == 8

    fracs = summary.global_topology_fractions("full_spectrum")
    assert len(fracs) == len(EXCLUSIVE_TOPOLOGY_NAMES)
    assert np.isclose(sum(fracs.values()), 1.0)

    json_file, npz_file = summary.save_bundle(tmp_path / "summary.json", tmp_path / "arrays.npz")
    restored = TrackStudySummary.load_bundle(json_file, npz_file)
    assert restored.total_tracks == summary.total_tracks
    assert restored.window_totals == summary.window_totals
    assert np.array_equal(restored.arrays["spec_total"], summary.arrays["spec_total"])


def test_extract_detector_events_from_tracks_and_projection_builder(tmp_path: Path) -> None:
    """Проверяет бесшовную конвертацию аннотированных треков в события (M, 5) для ProjectionBuilder."""
    hdf_file = _create_synthetic_track_hdf5(tmp_path / "0.0_deg.hdf", gantry_angle_value=0.0)
    pipeline = TrackProcessingPipeline(energy_resolution=0.0, intrinsic_resolution=0.0 * mm, random_seed=42)
    tracks, _ = pipeline.process_file(hdf_file)

    # Из 4 треков один имеет нулевое энерговыделение (particle_id == 20) и отфильтровывается при only_scintillating=True
    all_scint_events = extract_detector_events_from_tracks(tracks, only_scintillating=True)
    assert all_scint_events.shape == (3, 5)
    assert 20.0 not in set(all_scint_events[:, 0].tolist())

    det1_events = extract_detector_events_from_tracks(tracks, detector_id=1, only_scintillating=True)
    det2_events = extract_detector_events_from_tracks(tracks, detector_id=2, only_scintillating=True)
    assert det1_events.shape == (2, 5)
    assert det2_events.shape == (1, 5)

    # Построение проекции только по геометрическому каналу детектора 1 через общий ProjectionBuilder
    classification = classify_tracks(tracks)
    det1_geom_events = extract_detector_events_from_tracks(
        tracks,
        detector_id=1,
        selection_mask=classification.component_codes == int(PhotonComponent.GEOMETRIC),
    )
    builder = ProjectionBuilder(
        pixel_size=4.0 * mm,
        energy_windows=(TB161_PEAK_75_WINDOW,),
        detector_size=(540.0 * mm, 400.0 * mm),
    )
    combined_stack, _ = builder.build_projections([det1_geom_events, det2_events])
    assert combined_stack.shape == (2, 135, 100)
    assert np.isclose(np.sum(combined_stack[0]), 1.0)
    assert np.isclose(np.sum(combined_stack[1]), 1.0)


def test_custom_taxonomy_schemes_strict_priority_and_reporting(tmp_path: Path) -> None:
    """Проверяет пользовательские схемы TaxonomyScheme (strict_partition, priority, JSON и генерацию отчёта)."""
    study_dir = tmp_path / "custom_study"
    study_dir.mkdir()
    _create_synthetic_track_hdf5(study_dir / "0.0_deg.hdf", gantry_angle_value=0.0)
    _create_synthetic_track_hdf5(study_dir / "60.0_deg.hdf", gantry_angle_value=60.0)

    pipeline = TrackProcessingPipeline(random_seed=21)
    sample_tracks, _ = pipeline.process_file(study_dir / "0.0_deg.hdf")

    # 1. Декларативная схема strict_partition (3 класса по глубине кристалла и фотоэффекту) + сериализация JSON
    declarative_scheme = TaxonomyScheme(
        name="depth_and_escape_3class",
        mode="strict_partition",
        rules=(
            TrackClassRule(
                name="shallow_photo",
                label="Входной слой кристалла (фотоэффект, z <= 4 мм)",
                photon_component=PhotonComponent.GEOMETRIC,
                conditions=(
                    FieldCondition("crystal_photoelectric", min_value=1),
                    FieldCondition("crystal_depth", max_value=4.0 * mm, include_max=True),
                ),
                color="#1b9e77",
                is_primary_signal=True,
            ),
            TrackClassRule(
                name="deep_photo",
                label="Глубокий слой кристалла (фотоэффект, z > 4 мм)",
                photon_component=PhotonComponent.BACKSCATTER,
                conditions=(
                    FieldCondition("crystal_photoelectric", min_value=1),
                    FieldCondition("crystal_depth", min_value=4.0 * mm, include_min=False),
                ),
                color="#d95f02",
            ),
            TrackClassRule(
                name="crystal_escape_any",
                label="Неполное поглощение / вылет из кристалла",
                photon_component=PhotonComponent.PHANTOM_SCATTER,
                conditions=(FieldCondition("crystal_photoelectric", equals=0),),
                color="#377eb8",
            ),
        ),
        auxiliary_rules=(
            AuxiliaryMaskRule(
                name="high_energy_birth",
                label="Рождение с энергией >= 100 кэВ",
                conditions=(FieldCondition("emission_energy", min_value=100.0 * keV),),
            ),
        ),
    )

    scheme_dict = declarative_scheme.to_dict()
    roundtrip_scheme = TaxonomyScheme.from_dict(scheme_dict)
    res_decl = classify_tracks(sample_tracks, scheme=roundtrip_scheme)
    assert res_decl.scheme_name == "depth_and_escape_3class"
    assert sum(int(np.count_nonzero(mask_v)) for mask_v in res_decl.exclusive_masks.values()) == sample_tracks.num_tracks
    assert "high_energy_birth" in res_decl.auxiliary_masks
    assert "non_geometric" in res_decl.auxiliary_masks

    # Проверка мутаторов with_added_auxiliary и with_replaced_rule
    extended_scheme = roundtrip_scheme.with_added_auxiliary(
        AuxiliaryMaskRule(
            name="first_head_only",
            conditions=(FieldCondition("detector_id", equals=1),),
        )
    )
    assert "first_head_only" in extended_scheme.auxiliary_names

    # 2. Негативная проверка Fail-Fast на пересечение правил в mode='strict_partition'
    overlapping_scheme = TaxonomyScheme(
        name="bad_overlap",
        mode="strict_partition",
        rules=(
            TrackClassRule(
                name="rule_alpha",
                label="Альфа",
                photon_component=PhotonComponent.GEOMETRIC,
                conditions=(FieldCondition("detector_id", min_value=1),),
            ),
            TrackClassRule(
                name="rule_beta",
                label="Бета",
                photon_component=PhotonComponent.BACKSCATTER,
                conditions=(FieldCondition("crystal_depth", min_value=0.0 * mm),),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="rule_alpha.*rule_beta"):
        overlapping_scheme.classify(sample_tracks)

    # 3. Каскадная схема mode='priority' с замыкающим правилом fallback_rule
    priority_scheme = TaxonomyScheme(
        name="priority_cascade",
        mode="priority",
        rules=(
            TrackClassRule(
                name="high_birth_priority",
                label="Жёсткие кванты >= 200 кэВ",
                photon_component=PhotonComponent.SEPTAL_PENETRATION,
                conditions=(FieldCondition("emission_energy", min_value=200.0 * keV),),
                color="#e41a1c",
            ),
            TrackClassRule(
                name="clean_primary",
                label="Прямой фотоэффект",
                photon_component=PhotonComponent.GEOMETRIC,
                conditions=(
                    FieldCondition("crystal_photoelectric", min_value=1),
                    FieldCondition("septal_crossings", equals=0),
                ),
                color="#1b9e77",
                is_primary_signal=True,
            ),
        ),
        fallback_rule=TrackClassRule(
            name="residual_background",
            label="Остаточный фон",
            photon_component=PhotonComponent.PHANTOM_SCATTER,
            color="#7570b3",
        ),
    )
    res_prio = priority_scheme.classify(sample_tracks)
    assert res_prio.exclusive_masks.keys() == {"high_birth_priority", "clean_primary", "residual_background"}
    assert sum(int(np.count_nonzero(mask_v)) for mask_v in res_prio.exclusive_masks.values()) == sample_tracks.num_tracks

    # 4. Сквозной прогон StreamingTrackStudyAnalyzer + сохранение бандла + генерация DOCX-отчёта на кастомной схеме
    custom_cfg = TrackStudyConfig.for_isotope(
        "161Tb",
        taxonomy_scheme=roundtrip_scheme,
        vtk_sample_per_view=10,
        random_seed=5,
    )
    custom_summary = StreamingTrackStudyAnalyzer(config=custom_cfg).analyze_directory(study_dir)
    assert custom_summary.taxonomy_scheme_name == "depth_and_escape_3class"
    assert custom_summary.exclusive_class_names == ("shallow_photo", "deep_photo", "crystal_escape_any")
    custom_fracs = custom_summary.global_topology_fractions("full_spectrum")
    assert len(custom_fracs) == 3
    assert np.isclose(sum(custom_fracs.values()), 1.0)

    c_json, c_npz = custom_summary.save_bundle(tmp_path / "custom_sum.json", tmp_path / "custom_arr.npz")
    loaded_custom = TrackStudySummary.load_bundle(c_json, c_npz)
    assert loaded_custom.exclusive_class_names == ("shallow_photo", "deep_photo", "crystal_escape_any")
    assert loaded_custom.primary_class_names == ("shallow_photo",)

    report_gen = TrackStudyReportGenerator()
    report_result = report_gen.generate_report(
        loaded_custom,
        tmp_path / "custom_taxonomy_report.docx",
        export_pdf=False,
        export_legacy_doc=False,
    )
    assert report_result.docx_path.is_file()
    assert len(report_result.figure_paths) == 8
