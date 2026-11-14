"""Векторный экспорт полилиний 3D-треков частиц и мультиблочных сцен в форматы VTK ParaView (.vtp и .vtm)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import vtk
from hepunits import mm
from numpy.typing import NDArray
from vtk.util import numpy_support

from nmspect.tracks.models import TrackBatch
from nmspect.tracks.pipeline import TrackProcessingPipeline


class VtkTrackExporter:
    """Экспортирует пакеты полилиний треков `TrackBatch` и 3D-сцены гантри в форматы `.vtp` и `.vtm`.

    Геометрические координаты узлов и точек записываются в миллиметрах (`mm`), а скалярные физические поля
    `PointData`/`CellData` сохраняются в базовых единицах `hepunits` (`MeV`, `ns`, `rad`, `mm`).
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Инициализирует экспортер треков VTK ParaView.

        Args:
            output_path: Опциональный базовый путь сохранения файла `.vtp` или `.vtm`.
        """
        self.output_path: Path | None = Path(output_path) if output_path is not None else None

    @staticmethod
    def _numpy_array_to_vtk(field_name: str, field_array: NDArray[Any]) -> vtk.vtkDataArray:
        """Конвертирует 1D скалярный или 2D векторный массив NumPy в `vtkDataArray` без циклов."""
        arr_obj = np.asarray(field_array)
        if arr_obj.dtype == np.bool_:
            arr_obj = arr_obj.astype(np.uint8)

        if arr_obj.ndim not in (1, 2):
            raise ValueError(
                f"Массив атрибута '{field_name}' должен быть 1D или 2D, получена форма: {arr_obj.shape}."
            )

        contiguous_arr = np.ascontiguousarray(arr_obj)
        vtk_arr = numpy_support.numpy_to_vtk(contiguous_arr, deep=True)
        vtk_arr.SetName(field_name)
        return vtk_arr

    @staticmethod
    def build_polydata(tracks: TrackBatch) -> vtk.vtkPolyData:
        """Формирует объект `vtk.vtkPolyData` с полилиниями треков, `PointData` и `CellData` без циклов.

        Args:
            tracks: Пакет треков `TrackBatch` в CSR-представлении.

        Returns:
            Готовый экземпляр `vtk.vtkPolyData` с линиями и привязанными атрибутами узлов и ячеек.
        """
        poly_data = vtk.vtkPolyData()

        points_contiguous = np.ascontiguousarray(tracks.points / float(mm), dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(points_contiguous, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_points_array)
        poly_data.SetPoints(vtk_points)

        connectivity = np.arange(tracks.num_points, dtype=np.int64)
        vtk_offsets = numpy_support.numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(tracks.offsets, dtype=np.int64),
            deep=True,
        )
        vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)

        cell_array = vtk.vtkCellArray()
        cell_array.SetData(vtk_offsets, vtk_conn)
        poly_data.SetLines(cell_array)

        point_data_container = poly_data.GetPointData()
        for field_name, field_array in tracks.point_data.items():
            vtk_arr = VtkTrackExporter._numpy_array_to_vtk(field_name, field_array)
            point_data_container.AddArray(vtk_arr)

        cell_data_container = poly_data.GetCellData()
        for field_name, field_array in tracks.cell_data.items():
            vtk_arr = VtkTrackExporter._numpy_array_to_vtk(field_name, field_array)
            cell_data_container.AddArray(vtk_arr)

        return poly_data

    @staticmethod
    def build_registered_hits_polydata(
        tracks: TrackBatch,
        only_positive_deposit: bool = False,
    ) -> vtk.vtkPolyData:
        """Строит облако точек `vtk.vtkPolyData` для зарегистрированных вспышек на кристаллах детекторов.

        Все атрибуты из `tracks.cell_data` записываются в `PointData` облака точек, что позволяет
        в ParaView окрашивать и фильтровать вспышки по энергии, классу топологии и окнам.

        Args:
            tracks: Пакет треков `TrackBatch`, содержащий поле `'registered_position'` в `cell_data`.
            only_positive_deposit: Если `True`, исключает треки с нулевым энерговыделением в кристалле.

        Returns:
            Экземпляр `vtk.vtkPolyData` с точечными ячейками (`Verts`) и атрибутами в `PointData`.
        """
        if "registered_position" not in tracks.cell_data:
            raise KeyError("В tracks.cell_data отсутствует поле 'registered_position'.")

        raw_positions = np.asarray(tracks.cell_data["registered_position"], dtype=np.float32)
        if raw_positions.ndim != 2 or raw_positions.shape[1] != 3:
            raise ValueError(
                f"Поле 'registered_position' должно иметь форму (num_tracks, 3), получено: {raw_positions.shape}."
            )

        valid_mask = np.all(np.isfinite(raw_positions), axis=1)
        if only_positive_deposit and "deposited_energy" in tracks.cell_data:
            valid_mask = valid_mask & (np.asarray(tracks.cell_data["deposited_energy"], dtype=np.float64) > 0.0)
        all_valid = bool(np.all(valid_mask))
        filtered_positions = raw_positions if all_valid else raw_positions[valid_mask]
        num_hits = int(filtered_positions.shape[0])

        hits_poly_data = vtk.vtkPolyData()
        points_contiguous = np.ascontiguousarray(filtered_positions / float(mm), dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(points_contiguous, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_points_array)
        hits_poly_data.SetPoints(vtk_points)

        hit_offsets = np.arange(num_hits + 1, dtype=np.int64)
        hit_connectivity = np.arange(num_hits, dtype=np.int64)
        vtk_offsets = numpy_support.numpy_to_vtkIdTypeArray(hit_offsets, deep=True)
        vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(hit_connectivity, deep=True)

        verts_array = vtk.vtkCellArray()
        verts_array.SetData(vtk_offsets, vtk_conn)
        hits_poly_data.SetVerts(verts_array)

        point_data_container = hits_poly_data.GetPointData()
        for field_name, field_array in tracks.cell_data.items():
            filtered_field = field_array if all_valid else field_array[valid_mask]
            vtk_arr = VtkTrackExporter._numpy_array_to_vtk(field_name, filtered_field)
            point_data_container.AddArray(vtk_arr)

        return hits_poly_data

    @staticmethod
    def _write_polydata_vtp(poly_data: vtk.vtkPolyData, file_path: Path) -> None:
        """Записывает `vtk.vtkPolyData` в бинарный XML-файл `.vtp`."""
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(poly_data)
        writer.SetDataModeToBinary()
        if writer.Write() != 1:
            raise OSError(f"Не удалось записать файл VTK PolyData: {file_path}")

    @staticmethod
    def _write_structural_box_vti(
        file_path: Path,
        center: tuple[float, float, float],
        lengths: tuple[float, float, float],
        direction_matrix: NDArray[np.float64] | None = None,
    ) -> None:
        """Создаёт и сохраняет ориентированный структурный 3D-бокс (`vtkImageData` без скаляров) в файл `.vti`."""
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(2, 2, 2)
        lengths_arr = np.asarray(
            (float(lengths[0]), float(lengths[1]), float(lengths[2])),
            dtype=np.float64,
        )
        image_data.SetSpacing(float(lengths_arr[0]), float(lengths_arr[1]), float(lengths_arr[2]))

        if direction_matrix is not None:
            rot_arr = np.asarray(direction_matrix, dtype=np.float64).copy()
            if float(np.linalg.det(rot_arr)) < 0.0:
                rot_arr[:, 2] = -rot_arr[:, 2]
            vtk_dir_mat = vtk.vtkMatrix3x3()
            for row_idx in range(3):
                for col_idx in range(3):
                    vtk_dir_mat.SetElement(row_idx, col_idx, float(rot_arr[row_idx, col_idx]))
            image_data.SetDirectionMatrix(vtk_dir_mat)
            origin_vec = np.asarray(center, dtype=np.float64) - 0.5 * (rot_arr @ lengths_arr)
            image_data.SetOrigin(float(origin_vec[0]), float(origin_vec[1]), float(origin_vec[2]))
        else:
            image_data.SetOrigin(
                float(center[0] - 0.5 * lengths_arr[0]),
                float(center[1] - 0.5 * lengths_arr[1]),
                float(center[2] - 0.5 * lengths_arr[2]),
            )

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(image_data)
        if writer.Write() != 1:
            raise OSError(f"Не удалось записать структурный блок VTK ImageData: {file_path}")

    @staticmethod
    def _resolve_box_from_matrix(
        pose_matrix: NDArray[np.floating[Any]] | None,
        local_lengths: tuple[float, float, float],
        fallback_center: tuple[float, float, float],
        fallback_lengths: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], NDArray[np.float64] | None]:
        """Вычисляет мировой центр, локальные габариты и ортонормированную матрицу ориентации 3D-бокса."""
        if pose_matrix is None:
            return fallback_center, fallback_lengths, None

        mat_arr = np.asarray(pose_matrix, dtype=np.float64)
        if mat_arr.shape != (4, 4):
            return fallback_center, fallback_lengths, None

        center_coords = (
            float(mat_arr[0, 3]),
            float(mat_arr[1, 3]),
            float(mat_arr[2, 3]),
        )
        rot_raw = mat_arr[:3, :3].copy()
        col_norms = np.linalg.norm(rot_raw, axis=0)
        safe_norms = np.where(col_norms > 1e-9, col_norms, 1.0)
        rot_unit = rot_raw / safe_norms[np.newaxis, :]
        scaled_lengths = (
            max(float(local_lengths[0] * safe_norms[0]), 1.0),
            max(float(local_lengths[1] * safe_norms[1]), 1.0),
            max(float(local_lengths[2] * safe_norms[2]), 1.0),
        )
        return center_coords, scaled_lengths, rot_unit

    @staticmethod
    def _find_volume_matrix(
        metadata: dict[str, Any] | None,
        candidate_names: Sequence[str],
        gantry_pose_key: str | None = None,
    ) -> NDArray[np.float64] | None:
        """Ищет матрицу трансформации 4x4 в словаре `metadata` по списку возможных ключей."""
        if metadata is None:
            return None

        volume_matrices = metadata.get("volume_matrices")
        if isinstance(volume_matrices, dict):
            for vol_name in candidate_names:
                if vol_name in volume_matrices:
                    return np.asarray(volume_matrices[vol_name], dtype=np.float64)

        if gantry_pose_key is not None:
            gantry_meta = metadata.get("gantry")
            if isinstance(gantry_meta, dict) and gantry_pose_key in gantry_meta:
                return np.asarray(gantry_meta[gantry_pose_key], dtype=np.float64)

        return None

    def export_vtp(
        self,
        tracks: TrackBatch,
        output_path: str | Path | None = None,
    ) -> Path:
        """Экспортирует пакет полилиний треков `TrackBatch` в бинарный XML-файл `.vtp`.

        Args:
            tracks: Пакет полилиний треков `TrackBatch`.
            output_path: Целевой путь сохранения файла `.vtp` (переопределяет путь из конструктора).

        Returns:
            Полный путь к сохранённому файлу `.vtp`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения файла VTK PolyData (output_path).")

        if target_path.suffix.lower() != ".vtp":
            target_path = target_path.with_suffix(".vtp")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        poly_data = self.build_polydata(tracks)
        self._write_polydata_vtp(poly_data, target_path)
        return target_path

    def export_scene_vtm(
        self,
        tracks: TrackBatch,
        metadata: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
        orbit_radius: float = 250.0 * mm,
        detector_size: Sequence[float] | None = None,
    ) -> Path:
        """Экспортирует мультиблочную 3D-сцену ParaView `.vtm` с треками, вспышками и геометрией сканера.

        Args:
            tracks: Пакет полилиний треков `TrackBatch`.
            metadata: Опциональный словарь метаданных симуляции с реальными матрицами и габаритами.
            output_path: Целевой путь сохранения файла `.vtm` (переопределяет путь из конструктора).
            orbit_radius: Радиус орбиты вращения детекторов в базовых единицах hepunits (mm).
            detector_size: Физические габариты кристалла `(size_x, size_y, size_z)` в hepunits (mm).

        Returns:
            Полный путь к созданному файлу `.vtm`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения файла VTK MultiBlock (output_path).")
        if orbit_radius <= 0.0:
            raise ValueError(f"Радиус орбиты должен быть строго положительным: {orbit_radius}")

        if target_path.suffix.lower() != ".vtm":
            target_path = target_path.with_suffix(".vtm")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        blocks_dir = target_path.parent / target_path.stem
        blocks_dir.mkdir(parents=True, exist_ok=True)

        geom_meta = metadata.get("geometry", {}) if isinstance(metadata, dict) else {}
        gantry_meta = metadata.get("gantry", {}) if isinstance(metadata, dict) else {}

        if detector_size is not None and len(detector_size) >= 3:
            det_dims = (
                float(detector_size[0] / mm),
                float(detector_size[1] / mm),
                float(detector_size[2] / mm),
            )
        elif isinstance(geom_meta, dict) and "crystal_dimensions" in geom_meta:
            raw_dims = np.asarray(geom_meta["crystal_dimensions"], dtype=np.float64).ravel()
            if raw_dims.size >= 3:
                det_dims = (float(raw_dims[0]), float(raw_dims[1]), float(raw_dims[2]))
            else:
                det_dims = (540.0, 400.0, 9.5)
        else:
            det_dims = (540.0, 400.0, 9.5)

        collimator_thick = (
            float(np.asarray(geom_meta["collimator_hole_length"]).item())
            if isinstance(geom_meta, dict) and "collimator_hole_length" in geom_meta
            else 35.0
        )
        casing_dims: tuple[float, float, float]
        if isinstance(geom_meta, dict) and "casing_dimensions" in geom_meta:
            raw_cas = np.asarray(geom_meta["casing_dimensions"], dtype=np.float64).ravel()
            if raw_cas.size >= 3:
                casing_dims = (float(raw_cas[0]), float(raw_cas[1]), float(raw_cas[2]))
            else:
                casing_dims = (det_dims[0] + 40.0, det_dims[1] + 40.0, 96.0)
        else:
            casing_dims = (det_dims[0] + 40.0, det_dims[1] + 40.0, 96.0)
        casing_thick = casing_dims[2]
        crystal_thick = det_dims[2]

        radius_val = (
            float(np.asarray(gantry_meta["orbit_radius"]).item())
            if isinstance(gantry_meta, dict) and "orbit_radius" in gantry_meta
            else float(orbit_radius / mm)
        )
        collimator_center_y = radius_val - 0.5 * crystal_thick - 0.5 * collimator_thick
        casing_center_y = radius_val + 0.5 * crystal_thick + 0.5 * casing_thick

        active_heads: set[int] = set()
        if isinstance(gantry_meta, dict) and "detector_heads_count" in gantry_meta:
            heads_count = int(np.asarray(gantry_meta["detector_heads_count"]).item())
            for head_num in range(1, max(1, heads_count) + 1):
                active_heads.add(head_num)
        volume_matrices = metadata.get("volume_matrices") if isinstance(metadata, dict) else None
        if isinstance(volume_matrices, dict):
            for vol_key in volume_matrices:
                for head_cand in (1, 2, 3, 4):
                    if f"Detector_{head_cand}" in vol_key or f"Collimator_{head_cand}" in vol_key:
                        active_heads.add(head_cand)
        if not active_heads and "detector_id" in tracks.cell_data and tracks.num_tracks > 0:
            for det_val in np.unique(tracks.cell_data["detector_id"]):
                active_heads.add(int(det_val))
        if not active_heads:
            active_heads = {1, 2}
        sorted_heads = sorted(active_heads)

        block_entries: list[tuple[str, str]] = []

        # 1. Блок полилиний треков
        tracks_vtp = blocks_dir / "Tracks.vtp"
        tracks_poly_data = self.build_polydata(tracks)
        self._write_polydata_vtp(tracks_poly_data, tracks_vtp)
        block_entries.append(("Tracks", f"{blocks_dir.name}/{tracks_vtp.name}"))

        # 2. Опциональный блок облака зарегистрированных вспышек на кристаллах
        if "registered_position" in tracks.cell_data:
            hits_vtp = blocks_dir / "Registered_Hits.vtp"
            hits_poly_data = self.build_registered_hits_polydata(tracks)
            self._write_polydata_vtp(hits_poly_data, hits_vtp)
            block_entries.append(("Registered_Hits", f"{blocks_dir.name}/{hits_vtp.name}"))

        # 3. Структурные 3D-боксы геометрии сцены
        phantom_mat = self._find_volume_matrix(metadata, ("Phantom",))
        phantom_center, phantom_lengths, phantom_dir = self._resolve_box_from_matrix(
            pose_matrix=phantom_mat,
            local_lengths=(300.0, 230.0, 220.0),
            fallback_center=(0.0, 0.0, 0.0),
            fallback_lengths=(300.0, 230.0, 220.0),
        )
        phantom_vti = blocks_dir / "Phantom.vti"
        self._write_structural_box_vti(phantom_vti, phantom_center, phantom_lengths, phantom_dir)
        block_entries.append(("Phantom", f"{blocks_dir.name}/{phantom_vti.name}"))

        for head_idx in sorted_heads:
            head_sign = 1.0 if head_idx % 2 == 1 else -1.0
            coll_mat = self._find_volume_matrix(metadata, (f"Collimator_{head_idx}_body", f"Collimator_{head_idx}"))
            coll_center, coll_lengths, coll_dir = self._resolve_box_from_matrix(
                pose_matrix=coll_mat,
                local_lengths=(det_dims[0], det_dims[1], collimator_thick),
                fallback_center=(0.0, head_sign * collimator_center_y, 0.0),
                fallback_lengths=(det_dims[0], collimator_thick, det_dims[1]),
            )
            coll_vti = blocks_dir / f"Collimator_{head_idx}.vti"
            self._write_structural_box_vti(coll_vti, coll_center, coll_lengths, coll_dir)
            block_entries.append((f"Collimator_{head_idx}", f"{blocks_dir.name}/{coll_vti.name}"))

        for head_idx in sorted_heads:
            head_sign = 1.0 if head_idx % 2 == 1 else -1.0
            cas_mat = self._find_volume_matrix(metadata, (f"Casing_{head_idx}",))
            cas_center, cas_lengths, cas_dir = self._resolve_box_from_matrix(
                pose_matrix=cas_mat,
                local_lengths=casing_dims,
                fallback_center=(0.0, head_sign * casing_center_y, 0.0),
                fallback_lengths=(casing_dims[0], casing_thick, casing_dims[1]),
            )
            cas_vti = blocks_dir / f"Casing_{head_idx}.vti"
            self._write_structural_box_vti(cas_vti, cas_center, cas_lengths, cas_dir)
            block_entries.append((f"Casing_{head_idx}", f"{blocks_dir.name}/{cas_vti.name}"))

        for head_idx in sorted_heads:
            head_sign = 1.0 if head_idx % 2 == 1 else -1.0
            det_mat = self._find_volume_matrix(
                metadata,
                (f"Detector_{head_idx}",),
                gantry_pose_key=f"pose_matrix_detector_{head_idx}",
            )
            det_center, det_lengths, det_dir = self._resolve_box_from_matrix(
                pose_matrix=det_mat,
                local_lengths=(det_dims[0], det_dims[1], crystal_thick),
                fallback_center=(0.0, head_sign * radius_val, 0.0),
                fallback_lengths=(det_dims[0], crystal_thick, det_dims[1]),
            )
            det_vti = blocks_dir / f"Detector_{head_idx}.vti"
            self._write_structural_box_vti(det_vti, det_center, det_lengths, det_dir)
            block_entries.append((f"Detector_{head_idx}", f"{blocks_dir.name}/{det_vti.name}"))

        # 4. Формирование XML-манифеста .vtm
        vtk_file_elem = ET.Element(
            "VTKFile",
            attrib={
                "type": "vtkMultiBlockDataSet",
                "version": "1.0",
                "byte_order": "LittleEndian",
                "header_type": "UInt32",
            },
        )
        multiblock_elem = ET.SubElement(vtk_file_elem, "vtkMultiBlockDataSet")
        for block_idx, (block_name, rel_file) in enumerate(block_entries):
            ET.SubElement(
                multiblock_elem,
                "DataSet",
                attrib={
                    "index": str(block_idx),
                    "name": block_name,
                    "file": rel_file,
                },
            )

        xml_tree = ET.ElementTree(vtk_file_elem)
        ET.indent(xml_tree, space="  ")
        xml_tree.write(target_path, encoding="utf-8", xml_declaration=True)
        return target_path

    def convert_hdf_to_vtk(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        pipeline: TrackProcessingPipeline | None = None,
        max_tracks_per_detector: int | None = None,
        max_files: int | None = None,
        as_scene_vtm: bool = False,
    ) -> Path:
        """Сквозным образом конвертирует файл `.hdf` или директорию ракурсов напрямую в VTK-треки (`.vtp` или `.vtm`).

        Args:
            input_path: Путь к входному файлу `.hdf` или директории с файлами угловых ракурсов.
            output_path: Целевой путь сохранения файла `.vtp` или `.vtm` (переопределяет путь из конструктора).
            pipeline: Конвейер сборки и аннотирования `TrackProcessingPipeline` (если `None` — создаётся по умолчанию).
            max_tracks_per_detector: Лимит числа треков на детектор в одном файле (если `pipeline is None`).
            max_files: Лимит числа обрабатываемых файлов при передаче директории.
            as_scene_vtm: Если `True` (или расширение `output_path == '.vtm'`), сохраняет полную 3D-сцену `.vtm`
                с треками, точками вспышек и геометрией гантри; иначе сохраняет чистый файл треков `.vtp`.

        Returns:
            Полный путь к созданному файлу `.vtp` или `.vtm`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения выходного файла VTK (output_path).")

        source_path = Path(input_path)
        active_pipeline = (
            pipeline
            if pipeline is not None
            else TrackProcessingPipeline(max_tracks_per_detector=max_tracks_per_detector)
        )

        if source_path.is_dir():
            tracks, metadata = active_pipeline.process_directory(source_path, max_files=max_files)
        elif source_path.is_file():
            tracks, metadata = active_pipeline.process_file(source_path)
        else:
            raise FileNotFoundError(f"Входной путь HDF5 не найден: {source_path}")

        if as_scene_vtm or target_path.suffix.lower() == ".vtm":
            return self.export_scene_vtm(tracks=tracks, metadata=metadata, output_path=target_path)
        return self.export_vtp(tracks=tracks, output_path=target_path)
