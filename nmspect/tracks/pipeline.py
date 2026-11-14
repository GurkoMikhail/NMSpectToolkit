"""Сквозной конвейер сборки и аннотирования треков частиц из файлов симуляций HDF5."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from hepunits import keV, mm

from nmspect.io.hdf5_reader import parse_gantry_angle_from_filename
from nmspect.tracks.assembler import HdfTrackAssembler
from nmspect.tracks.collimator_annotator import CollimatorTrackAnnotator
from nmspect.tracks.models import TrackAnnotator, TrackBatch
from nmspect.tracks.process_counters import ProcessCounterAnnotator
from nmspect.tracks.scintillation_annotator import ScintillationTrackAnnotator
from nmspect.tracks.taxonomy import TaxonomyScheme, TopologyClassAnnotator


class TrackProcessingPipeline:
    """Координирует извлечение первозданных треков из HDF5 и последовательное навешивание свойств в `cell_data`."""

    def __init__(
        self,
        assembler: HdfTrackAssembler | None = None,
        annotators: Sequence[TrackAnnotator] | None = None,
        max_tracks_per_detector: int | None = None,
        energy_resolution: float = 0.099,
        reference_energy: float = 140.5 * keV,
        intrinsic_resolution: float = 3.8 * mm,
        random_seed: int | None = 42,
        rng: np.random.Generator | None = None,
        require_energy_deposit: bool = False,
        taxonomy_scheme: TaxonomyScheme | None = None,
    ) -> None:
        """Инициализирует сквозной конвейер обработки треков.

        Args:
            assembler: Сборщик первозданных треков из HDF5 (если `None` — создаётся по умолчанию).
            annotators: Последовательность модулей-аннотаторов `TrackAnnotator` (если `None` — создаётся
                стандартная цепочка из счётчиков процессов, сцинтилляционного отклика, коллиматора и таксономии).
            max_tracks_per_detector: Лимит числа треков на один детектор в рамках одного файла.
            energy_resolution: Относительное энергетическое разрешение детектора (FWHM) на опорной энергии.
            reference_energy: Опорная энергия в базовых единицах hepunits (`MeV`).
            intrinsic_resolution: Внутреннее пространственное разрешение кристалл+ФЭУ (FWHM) в `mm`.
            random_seed: Зерно генератора случайных чисел для воспроизводимости выборки и размытия.
            rng: Опциональный общий экземпляр `np.random.Generator` (исключает сброс зерна между ракурсами).
            require_energy_deposit: Если `True`, отбрасывает треки с нулевым энерговыделением в кристалле.
            taxonomy_scheme: Опциональная пользовательская схема таксономии `TaxonomyScheme`
                (по умолчанию — канонические 10 классов `DEFAULT_TRACK_TAXONOMY`).
        """
        shared_rng = rng if rng is not None else np.random.default_rng(random_seed)
        self.rng: np.random.Generator = shared_rng
        self.taxonomy_scheme: TaxonomyScheme | None = taxonomy_scheme
        self.assembler: HdfTrackAssembler = (
            assembler
            if assembler is not None
            else HdfTrackAssembler(
                max_tracks_per_detector=max_tracks_per_detector,
                random_seed=random_seed,
                rng=shared_rng,
                require_energy_deposit=require_energy_deposit,
            )
        )
        if annotators is not None:
            self.annotators: tuple[TrackAnnotator, ...] = tuple(annotators)
        else:
            self.annotators = (
                ProcessCounterAnnotator(),
                ScintillationTrackAnnotator(
                    energy_resolution=energy_resolution,
                    reference_energy=reference_energy,
                    intrinsic_resolution=intrinsic_resolution,
                    random_seed=random_seed,
                    rng=shared_rng,
                ),
                CollimatorTrackAnnotator(),
                TopologyClassAnnotator(scheme=taxonomy_scheme),
            )

    def annotate_batch(
        self,
        tracks: TrackBatch,
        metadata: dict[str, Any] | None = None,
    ) -> TrackBatch:
        """Последовательно применяет все зарегистрированные аннотаторы к пакету треков `TrackBatch`."""
        for annotator_item in self.annotators:
            annotator_item.annotate(tracks, metadata)
        return tracks

    def process_file(
        self,
        file_path: str | Path,
        projection_index: int = 0,
        gantry_angle: float | None = None,
    ) -> tuple[TrackBatch, dict[str, Any]]:
        """Собирает первозданные треки из одного файла HDF5 и вычисляет все вторичные свойства в `cell_data`."""
        tracks, metadata = self.assembler.assemble_file(
            file_path=file_path,
            projection_index=projection_index,
            gantry_angle=gantry_angle,
        )
        self.annotate_batch(tracks, metadata)
        return tracks, metadata

    def process_directory(
        self,
        input_dir: str | Path,
        max_files: int | None = None,
    ) -> tuple[TrackBatch, dict[str, Any]]:
        """Собирает и аннотирует треки из всех файлов `.hdf`/`.hdf5` в директории, сортируя их по углу гантри."""
        dir_path = Path(input_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Директория с файлами симуляции не найдена: {dir_path}")
        if max_files is not None and max_files <= 0:
            raise ValueError(f"Параметр max_files должен быть положительным целым числом: {max_files}.")

        hdf_candidates = sorted(
            file_item
            for file_item in dir_path.iterdir()
            if file_item.is_file() and file_item.suffix.lower() in (".hdf", ".hdf5", ".h5")
        )
        if not hdf_candidates:
            raise FileNotFoundError(f"В директории '{dir_path}' не найдено файлов HDF5 (.hdf, .hdf5, .h5).")

        def _angle_sort_key(candidate_path: Path) -> tuple[float, str]:
            parsed_val = parse_gantry_angle_from_filename(candidate_path.name)
            return (float(parsed_val) if parsed_val is not None else float("inf"), candidate_path.name)

        sorted_files = sorted(hdf_candidates, key=_angle_sort_key)
        if max_files is not None:
            sorted_files = sorted_files[:max_files]

        batches: list[TrackBatch] = []
        primary_metadata: dict[str, Any] = {}
        num_files = len(sorted_files)

        for view_idx, hdf_path in enumerate(sorted_files):
            parsed_angle = parse_gantry_angle_from_filename(hdf_path.name)
            view_batch, view_metadata = self.process_file(
                file_path=hdf_path,
                projection_index=view_idx,
                gantry_angle=float(parsed_angle) if parsed_angle is not None else None,
            )
            if "detector_id" in view_batch.cell_data and "projection_index" in view_batch.cell_data:
                det_offset = (view_batch.cell_data["detector_id"].astype(np.int32) - 1) * num_files
                view_batch.cell_data["projection_index"] = (
                    view_batch.cell_data["projection_index"] + det_offset
                ).astype(np.int32)
            if view_idx == 0:
                primary_metadata = view_metadata
            batches.append(view_batch)

        combined_batch = TrackBatch.concat(batches)
        return combined_batch, primary_metadata
