"""Экспорт стека проекций ОФЭКТ в медицинский стандарт DICOM Nuclear Medicine (NM Image Storage)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from hepunits import deg, keV, mm
from numpy.typing import NDArray
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from nmspect.spectrometry.models import EnergyWindowConfig


class DicomNMExporter:
    """Экспортирует стек планарных проекций ОФЭКТ в многокадровый клинический файл DICOM NM (`TOMO`)."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Инициализирует экспортер медицинского формата DICOM NM.

        Args:
            output_path: Опциональный путь к целевому файлу `.dcm` (может быть также передан в `export`).
        """
        self.output_path: Path | None = Path(output_path) if output_path is not None else None

    def export(
        self,
        projections: NDArray[np.floating],
        angles: Sequence[float] | NDArray[np.floating],
        pixel_size: float = 4.2 * mm,
        window_config: EnergyWindowConfig | None = None,
        orbit_radius: float = 250.0 * mm,
        metadata: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Создаёт и сохраняет многокадровый файл DICOM NM Image Storage.

        Args:
            projections: 3D стек проекций формы `(views, nx, ny)` или 2D срез `(nx, ny)`.
            angles: Вектор углов проекций в базовых единицах hepunits (радианах).
            pixel_size: Размер пикселя детектора в базовых единицах hepunits (mm).
            window_config: Конфигурация энергетического окна `EnergyWindowConfig`.
            orbit_radius: Радиус орбиты вращения детекторов в базовых единицах hepunits (mm).
            metadata: Дополнительные метаданные исследования (пациент, радионуклид, сканер).
            output_path: Целевой путь сохранения файла `.dcm` (переопределяет путь из конструктора).

        Returns:
            Полный путь к созданному файлу `.dcm`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения файла DICOM NM (output_path).")
        if pixel_size <= 0.0:
            raise ValueError(f"Размер пикселя должен быть строго положительным: {pixel_size}")
        if orbit_radius <= 0.0:
            raise ValueError(f"Радиус орбиты должен быть строго положительным: {orbit_radius}")

        proj_arr = np.asarray(projections, dtype=np.float64)
        if proj_arr.ndim == 2:
            proj_arr = proj_arr[np.newaxis, ...]
        elif proj_arr.ndim != 3 or proj_arr.size == 0:
            raise ValueError(f"Ожидался непустой 2D или 3D массив проекций, получена форма: {proj_arr.shape}")

        angles_arr = np.asarray(angles, dtype=np.float64).ravel()
        if angles_arr.size != proj_arr.shape[0]:
            raise ValueError(
                f"Число углов ({angles_arr.size}) не совпадает с числом проекций ({proj_arr.shape[0]})."
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        meta_dict = metadata if metadata is not None else {}

        # Перевод из координат детектора [views, X, Y] в клиническую систему DICOM [views, rows, cols]
        proj_dicom = np.rot90(proj_arr, k=-1, axes=(1, 2))[:, ::-1, :]
        num_frames, num_rows, num_cols = proj_dicom.shape

        clipped_proj = np.clip(proj_dicom, 0.0, None)
        max_val = float(np.max(clipped_proj)) if clipped_proj.size > 0 else 0.0
        if max_val > 65535.0:
            scale_factor = 65535.0 / max_val
            pixel_array = np.round(clipped_proj * scale_factor).astype(np.uint16)
            rescale_slope = max_val / 65535.0
        else:
            pixel_array = np.round(clipped_proj).astype(np.uint16)
            rescale_slope = 1.0

        converted_angles = np.mod(angles_arr / deg, 360.0)
        start_angle = float(converted_angles[0]) if converted_angles.size > 0 else 0.0
        if converted_angles.size > 1:
            wrapped_diffs = ((np.diff(converted_angles) + 180.0) % 360.0) - 180.0
            non_zero_diffs = wrapped_diffs[np.abs(wrapped_diffs) > 1e-6]
            signed_step = float(np.median(non_zero_diffs)) if non_zero_diffs.size > 0 else 0.0
        else:
            signed_step = 3.0
        angular_step = abs(signed_step) if abs(signed_step) > 1e-6 else 3.0
        converted_pixel_size = float(pixel_size / mm)
        resolved_orbit = float(meta_dict.get("orbit_radius", orbit_radius) / mm)

        timestamp = datetime.now()
        study_date = timestamp.strftime("%Y%m%d")
        study_time = timestamp.strftime("%H%M%S")

        sop_instance_uid = generate_uid()
        sop_class_uid = UID("1.2.840.10008.5.1.4.1.1.20")

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = UID("1.2.826.0.1.3680043.8.498.1")
        file_meta.ImplementationVersionName = "NMSPECT_0.2"

        dicom_ds = FileDataset(str(target_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient Module
        dicom_ds.PatientName = str(meta_dict.get("patient_name", "PHANTOM^NEMA IEC"))
        dicom_ds.PatientID = str(meta_dict.get("patient_id", "NMSPECT-SIM-001"))
        dicom_ds.PatientBirthDate = ""
        dicom_ds.PatientSex = "O"

        # General Study Module
        dicom_ds.StudyInstanceUID = str(meta_dict.get("study_uid", generate_uid()))
        dicom_ds.StudyDate = study_date
        dicom_ds.StudyTime = study_time
        dicom_ds.ReferringPhysicianName = ""
        dicom_ds.StudyID = "1"
        dicom_ds.AccessionNumber = ""
        dicom_ds.StudyDescription = str(meta_dict.get("study_desc", "SPECT Simulation Study"))

        # General Series Module
        dicom_ds.Modality = "NM"
        dicom_ds.SeriesInstanceUID = str(meta_dict.get("series_uid", generate_uid()))
        dicom_ds.SeriesNumber = 1
        dicom_ds.SeriesDate = study_date
        dicom_ds.SeriesTime = study_time

        series_desc = str(meta_dict.get("series_desc", f"SPECT Projections {num_frames} views"))
        if window_config is not None:
            if window_config.primary_peak is not None:
                peak_val = float(window_config.primary_peak / keV)
                series_desc = f"{series_desc} ({window_config.name}, {peak_val:.1f} keV)"
            else:
                series_desc = f"{series_desc} ({window_config.name})"
        dicom_ds.SeriesDescription = series_desc

        # General Equipment Module
        dicom_ds.Manufacturer = str(meta_dict.get("manufacturer", "GE MEDICAL SYSTEMS"))
        dicom_ds.InstitutionName = "Radiology Research Lab"
        dicom_ds.StationName = "SPECT_WS"
        dicom_ds.ManufacturerModelName = str(meta_dict.get("scanner_name", "Discovery NM/CT 670"))

        # Image Pixel Module
        dicom_ds.InstanceNumber = 1
        dicom_ds.ContentDate = study_date
        dicom_ds.ContentTime = study_time
        dicom_ds.ImageType = ["ORIGINAL", "PRIMARY", "TOMO", "EMISSION"]
        dicom_ds.SamplesPerPixel = 1
        dicom_ds.PhotometricInterpretation = "MONOCHROME2"
        dicom_ds.Rows = num_rows
        dicom_ds.Columns = num_cols
        dicom_ds.BitsAllocated = 16
        dicom_ds.BitsStored = 16
        dicom_ds.HighBit = 15
        dicom_ds.PixelRepresentation = 0
        dicom_ds.PixelSpacing = [converted_pixel_size, converted_pixel_size]
        dicom_ds.RescaleSlope = rescale_slope
        dicom_ds.RescaleIntercept = 0.0
        dicom_ds.add_new(0x00280106, "US", int(np.min(pixel_array)) if pixel_array.size > 0 else 0)
        dicom_ds.add_new(0x00280107, "US", int(np.max(pixel_array)) if pixel_array.size > 0 else 0)

        effective_max = max(max_val, 1.0)
        dicom_ds.WindowCenter = float(meta_dict.get("window_center", round(effective_max / 2.0, 2)))
        dicom_ds.WindowWidth = float(meta_dict.get("window_width", round(effective_max, 2)))

        total_counts = int(meta_dict.get("counts_accumulated", round(float(np.sum(clipped_proj)))))
        dicom_ds.CountsAccumulated = total_counts

        # Multi-Frame NM Module
        dicom_ds.NumberOfFrames = num_frames
        dicom_ds.FrameIncrementPointer = [0x00540050, 0x00540090]
        dicom_ds.EnergyWindowVector = [1] * num_frames
        dicom_ds.NumberOfEnergyWindows = 1
        half_frames = max(1, num_frames // 2)
        default_detectors = [1 if frame_idx < half_frames else 2 for frame_idx in range(num_frames)]
        detector_vector = [
            int(det_id) for det_id in meta_dict.get("detector_vector", default_detectors)
        ]
        num_detectors = max(detector_vector) if detector_vector else 1
        dicom_ds.DetectorVector = detector_vector
        dicom_ds.NumberOfDetectors = num_detectors
        dicom_ds.PhaseVector = [1] * num_frames
        dicom_ds.NumberOfPhases = 1
        dicom_ds.RotationVector = [1] * num_frames
        dicom_ds.NumberOfRotations = 1
        dicom_ds.RRIntervalVector = [1] * num_frames
        dicom_ds.NumberOfRRIntervals = 1
        dicom_ds.TimeSlotVector = [1] * num_frames
        dicom_ds.NumberOfTimeSlots = 1
        dicom_ds.AngularViewVector = list(range(1, num_frames + 1))
        dicom_ds.TimeSliceVector = [1] * num_frames

        # NM Detector Module
        detector_info_seq = DicomSequence()
        for _ in range(num_detectors):
            det_item = Dataset()
            det_item.CollimatorGridName = str(meta_dict.get("collimator_grid", "LEHR"))
            det_item.CollimatorType = "PARA"
            det_item.FieldOfViewShape = "RECTANGLE"
            det_item.FieldOfViewDimensions = [
                round(num_cols * converted_pixel_size),
                round(num_rows * converted_pixel_size),
            ]
            det_item.FocalDistance = 0
            det_item.DistanceSourceToDetector = resolved_orbit
            detector_info_seq.append(det_item)
        dicom_ds.DetectorInformationSequence = detector_info_seq

        # NM TOMO Acquisition Module
        gantry_meta = meta_dict.get("gantry", {})
        default_rot_dir = "CCW" if signed_step >= 0.0 else "CW"
        raw_rot_dir = meta_dict.get(
            "rotation_direction",
            gantry_meta.get("rotation_direction", default_rot_dir)
            if isinstance(gantry_meta, dict)
            else default_rot_dir,
        )
        rot_dir_str = str(raw_rot_dir).strip().upper()
        tomo_rot_seq = DicomSequence()
        rot_item = Dataset()
        rot_item.StartAngle = start_angle
        rot_item.AngularStep = angular_step
        rot_item.RotationDirection = rot_dir_str if rot_dir_str in ("CW", "CCW") else default_rot_dir
        rot_item.ScanArc = 360.0
        rot_item.TypeOfDetectorMotion = "STEP AND SHOOT"
        rot_item.RadialPosition = [resolved_orbit] * num_frames
        rot_item.DistanceSourceToDetector = resolved_orbit
        rot_item.NumberOfFramesInRotation = num_frames
        rot_item.TableHeight = 0.0
        rot_item.TableTraverse = 0.0
        angle_values = [float(angle_val) for angle_val in converted_angles]
        rot_item.add_new(0x00540091, "DS", angle_values)
        tomo_rot_seq.append(rot_item)
        dicom_ds.RotationInformationSequence = tomo_rot_seq
        dicom_ds.FramePrimaryAngleVector = angle_values

        # NM Energy Window Module
        ew_info_seq = DicomSequence()
        ew_item = Dataset()
        if window_config is not None:
            ew_item.EnergyWindowName = window_config.name[:16]
        else:
            ew_item.EnergyWindowName = str(meta_dict.get("energy_window_name", "Photopeak"))[:16]

        ew_range_seq = DicomSequence()
        if window_config is not None:
            main_range = Dataset()
            main_range.EnergyWindowLowerLimit = float(window_config.lower_bound / keV)
            main_range.EnergyWindowUpperLimit = float(window_config.upper_bound / keV)
            ew_range_seq.append(main_range)
            for assoc_win in window_config.associated_windows:
                sub_range = Dataset()
                sub_range.EnergyWindowLowerLimit = float(assoc_win.lower_bound / keV)
                sub_range.EnergyWindowUpperLimit = float(assoc_win.upper_bound / keV)
                ew_range_seq.append(sub_range)
        else:
            fallback_range = Dataset()
            fallback_range.EnergyWindowLowerLimit = float(meta_dict.get("ew_lower", 67.1 * keV) / keV)
            fallback_range.EnergyWindowUpperLimit = float(meta_dict.get("ew_upper", 82.1 * keV) / keV)
            ew_range_seq.append(fallback_range)

        ew_item.EnergyWindowRangeSequence = ew_range_seq
        ew_info_seq.append(ew_item)
        dicom_ds.EnergyWindowInformationSequence = ew_info_seq

        # NM Isotope Module
        radiopharm_seq = DicomSequence()
        radiopharm_item = Dataset()
        radiopharm_item.Radiopharmaceutical = str(meta_dict.get("radiopharm", "161Tb Chloride"))
        radiopharm_item.RadionuclideTotalDose = 1.0
        radiopharm_seq.append(radiopharm_item)
        dicom_ds.RadiopharmaceuticalInformationSequence = radiopharm_seq

        # SOP Common Module
        dicom_ds.SOPClassUID = sop_class_uid
        dicom_ds.SOPInstanceUID = sop_instance_uid
        dicom_ds.PixelData = pixel_array.tobytes()

        dicom_ds.save_as(str(target_path), enforce_file_format=True)
        return target_path
