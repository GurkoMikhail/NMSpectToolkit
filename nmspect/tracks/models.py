"""Модели данных и протоколы для представления и аннотирования треков частиц в формате CSR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

EMISSION_PROCESS_CODE: Final[int] = -1
PHOTOELECTRIC_PROCESS_CODE: Final[int] = 0
COMPTON_PROCESS_CODE: Final[int] = 1
COHERENT_PROCESS_CODE: Final[int] = 2
PAIR_PRODUCTION_PROCESS_CODE: Final[int] = 3

EMISSION_VOLUME_ID: Final[int] = 0
CRYSTAL_NODE_FLAG: Final[int] = 1


@dataclass
class TrackBatch:
    """Пакет полилиний треков частиц в компактном CSR-представлении (Compressed Sparse Row).

    Attributes:
        points: Мировые координаты узлов треков формы `(num_points, 3)` в базовых единицах (`mm`).
        offsets: Монотонно возрастающий массив сдвигов начала/конца треков формы `(num_tracks + 1,)`.
        point_data: Словарь первозданных физических свойств узлов (длина каждого массива равна `num_points`).
        cell_data: Словарь паспортных и агрегированных свойств треков (длина каждого массива равна `num_tracks`).
    """

    points: NDArray[np.float32]
    offsets: NDArray[np.int64]
    point_data: dict[str, NDArray[Any]] = field(default_factory=dict)
    cell_data: dict[str, NDArray[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Проверяет структурные инварианты CSR-массивов, размерности координат и словарей атрибутов."""
        self.points = np.asarray(self.points, dtype=np.float32)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"Массив points должен иметь форму (num_points, 3), получено: {self.points.shape}."
            )

        if self.offsets.ndim != 1 or self.offsets.size < 1:
            raise ValueError(
                f"Массив offsets должен быть одномерным и содержать хотя бы 1 элемент, получено: {self.offsets.shape}."
            )

        if int(self.offsets[0]) != 0:
            raise ValueError(f"Первый элемент offsets[0] должен быть равен 0, получено: {int(self.offsets[0])}.")

        total_points = int(self.points.shape[0])
        if int(self.offsets[-1]) != total_points:
            raise ValueError(
                f"Последний элемент offsets[-1] ({int(self.offsets[-1])}) должен совпадать с числом точек ({total_points})."
            )

        if self.offsets.size > 1:
            lengths = np.diff(self.offsets)
            if bool(np.any(lengths < 2)):
                min_length = int(np.min(lengths))
                raise ValueError(
                    f"Каждый трек должен содержать не менее 2 узлов (offsets строго возрастает с шагом >= 2), "
                    f"минимальная длина: {min_length}."
                )

        total_tracks = int(self.offsets.size - 1)
        for field_name, field_array in self.point_data.items():
            arr_obj = np.asarray(field_array)
            if arr_obj.ndim < 1 or int(arr_obj.shape[0]) != total_points:
                raise ValueError(
                    f"Поле point_data['{field_name}'] имеет первую размерность "
                    f"{arr_obj.shape[0] if arr_obj.ndim >= 1 else 'scalar'}, ожидалось num_points={total_points}."
                )
            self.point_data[field_name] = arr_obj

        for field_name, field_array in self.cell_data.items():
            arr_obj = np.asarray(field_array)
            if arr_obj.ndim < 1 or int(arr_obj.shape[0]) != total_tracks:
                raise ValueError(
                    f"Поле cell_data['{field_name}'] имеет первую размерность "
                    f"{arr_obj.shape[0] if arr_obj.ndim >= 1 else 'scalar'}, ожидалось num_tracks={total_tracks}."
                )
            self.cell_data[field_name] = arr_obj

    @property
    def num_tracks(self) -> int:
        """Возвращает количество треков в пакете."""
        return int(self.offsets.size - 1)

    @property
    def num_points(self) -> int:
        """Возвращает суммарное количество узлов во всех треках пакета."""
        return int(self.points.shape[0])

    @property
    def track_lengths(self) -> NDArray[np.int64]:
        """Возвращает одномерный массив количества узлов в каждом треке."""
        return np.diff(self.offsets)

    def slice_tracks(self, track_indices: NDArray[np.integer[Any]] | Sequence[int]) -> TrackBatch:
        """Формирует новый пакет `TrackBatch` для выбранного подмножества треков полностью векторно.

        Args:
            track_indices: Индексы отбираемых треков (целочисленный массив или последовательность).

        Returns:
            Новый экземпляр `TrackBatch`, содержащий только выбранные треки и их узлы.
        """
        raw_indices = np.asarray(track_indices)
        if raw_indices.dtype == np.bool_:
            if raw_indices.ndim != 1 or int(raw_indices.size) != self.num_tracks:
                raise ValueError(
                    f"Булева маска треков должна иметь длину num_tracks={self.num_tracks}, "
                    f"получено: {raw_indices.shape}."
                )
            selected_tracks = np.flatnonzero(raw_indices).astype(np.int64)
        else:
            selected_tracks = np.asarray(track_indices, dtype=np.int64).ravel()
            if selected_tracks.size > 0 and (
                bool(np.any(selected_tracks < 0)) or bool(np.any(selected_tracks >= self.num_tracks))
            ):
                raise IndexError(
                    f"Индексы треков выходят за допустимый диапазон [0, {self.num_tracks})."
                )

        if selected_tracks.size == 0:
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_offsets = np.zeros(1, dtype=np.int64)
            empty_point_data = {
                field_name: field_array[:0].copy()
                for field_name, field_array in self.point_data.items()
            }
            empty_cell_data = {
                field_name: field_array[:0].copy()
                for field_name, field_array in self.cell_data.items()
            }
            return TrackBatch(
                points=empty_points,
                offsets=empty_offsets,
                point_data=empty_point_data,
                cell_data=empty_cell_data,
            )

        track_starts = self.offsets[selected_tracks]
        track_ends = self.offsets[selected_tracks + 1]
        selected_lengths = track_ends - track_starts

        new_offsets = np.empty(selected_tracks.size + 1, dtype=np.int64)
        new_offsets[0] = 0
        np.cumsum(selected_lengths, out=new_offsets[1:])
        total_sliced_points = int(new_offsets[-1])

        base_shifts = np.repeat(track_starts - new_offsets[:-1], selected_lengths)
        point_indices = np.arange(total_sliced_points, dtype=np.int64) + base_shifts

        sliced_points = self.points[point_indices]
        sliced_point_data = {
            field_name: field_array[point_indices]
            for field_name, field_array in self.point_data.items()
        }
        sliced_cell_data = {
            field_name: field_array[selected_tracks]
            for field_name, field_array in self.cell_data.items()
        }

        return TrackBatch(
            points=sliced_points,
            offsets=new_offsets,
            point_data=sliced_point_data,
            cell_data=sliced_cell_data,
        )

    @classmethod
    def concat(cls, batches: Sequence[TrackBatch]) -> TrackBatch:
        """Объединяет последовательность пакетов `TrackBatch` в единый пакет.

        Args:
            batches: Непустая последовательность пакетов `TrackBatch` с согласованными ключами.

        Returns:
            Объединённый экземпляр `TrackBatch`.
        """
        if len(batches) == 0:
            raise ValueError("Последовательность пакетов TrackBatch для объединения не может быть пустой.")

        first_batch = batches[0]
        expected_point_keys = set(first_batch.point_data.keys())
        expected_cell_keys = set(first_batch.cell_data.keys())

        for batch_idx, batch_item in enumerate(batches[1:], start=1):
            if set(batch_item.point_data.keys()) != expected_point_keys:
                raise ValueError(
                    f"Ключи point_data в пакете #{batch_idx} не совпадают с эталонными ключами первого пакета."
                )
            if set(batch_item.cell_data.keys()) != expected_cell_keys:
                raise ValueError(
                    f"Ключи cell_data в пакете #{batch_idx} не совпадают с эталонными ключами первого пакета."
                )

        all_lengths = np.concatenate([batch_item.track_lengths for batch_item in batches])
        merged_offsets = np.empty(all_lengths.size + 1, dtype=np.int64)
        merged_offsets[0] = 0
        if all_lengths.size > 0:
            np.cumsum(all_lengths, out=merged_offsets[1:])

        merged_points = np.concatenate([batch_item.points for batch_item in batches], axis=0).astype(
            np.float32, copy=False
        )
        merged_point_data = {
            field_name: np.concatenate([batch_item.point_data[field_name] for batch_item in batches], axis=0)
            for field_name in first_batch.point_data
        }
        merged_cell_data = {
            field_name: np.concatenate([batch_item.cell_data[field_name] for batch_item in batches], axis=0)
            for field_name in first_batch.cell_data
        }

        return cls(
            points=merged_points,
            offsets=merged_offsets,
            point_data=merged_point_data,
            cell_data=merged_cell_data,
        )


@runtime_checkable
class TrackAnnotator(Protocol):
    """Протокол модуля-аннотатора, вычисляющего вторичные свойства треков в `cell_data`."""

    def annotate(self, tracks: TrackBatch, metadata: dict[str, Any] | None = None) -> None:
        """Аннотирует переданный пакет треков, добавляя вычисленные поля в `tracks.cell_data`."""
        ...
