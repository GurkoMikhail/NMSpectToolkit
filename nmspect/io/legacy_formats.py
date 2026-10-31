"""Утилиты сохранения и конвертации проекционных массивов в форматы .npy, .spr и .png."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def save_projections_as_npy(
    projections: NDArray[np.floating],
    output_path: str | Path,
) -> Path:
    """Сохраняет 2D или 3D массив проекций в бинарный файл формата NumPy (`.npy`).

    Args:
        projections: Массив проекций формы `(height, width)` или `(views, height, width)`.
        output_path: Целевой путь сохранения файла `.npy`.

    Returns:
        Полный путь к сохранённому файлу `.npy`.
    """
    proj_array = np.asarray(projections, dtype=np.float64)
    if proj_array.ndim not in (2, 3) or proj_array.size == 0:
        raise ValueError(
            f"Массив проекций должен быть непустым 2D или 3D массивом, получена форма: {proj_array.shape}"
        )

    target_path = Path(output_path)
    if target_path.suffix.lower() != ".npy":
        target_path = target_path.with_suffix(".npy")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, proj_array)
    return target_path


def save_projections_as_spr(
    projections: NDArray[np.floating],
    output_dir: str | Path,
    file_prefix: str = "slice_",
) -> Path:
    """Сохраняет стек проекций по кадрам в табличные файлы `.spr` (разделитель — табуляция).

    Args:
        projections: Стек проекций формы `(views, height, width)` или 2D срез `(height, width)`.
        output_dir: Директория для записи покадровых файлов `.spr`.
        file_prefix: Префикс имён создаваемых файлов (по умолчанию `'slice_'`).

    Returns:
        Путь к директории с сохранёнными файлами `.spr`.
    """
    proj_array = np.asarray(projections, dtype=np.float64)
    if proj_array.ndim == 2:
        proj_array = proj_array[np.newaxis, ...]
    elif proj_array.ndim != 3 or proj_array.size == 0:
        raise ValueError(
            f"Ожидался непустой 2D или 3D массив проекций, получена форма: {proj_array.shape}"
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_matrix in enumerate(proj_array):
        slice_path = target_dir / f"{file_prefix}{frame_idx:03d}.spr"
        np.savetxt(slice_path, frame_matrix, fmt="%.6f", delimiter="\t")

    return target_dir


def save_spr_as_png(
    spr_dir: str | Path,
    output_dir: str | Path,
    image_shape: tuple[int, int] = (128, 128),
) -> list[Path]:
    """Конвертирует файлы проекционных срезов `.spr` из директории в серию 8-битных изображений `.png`.

    Args:
        spr_dir: Входная директория с файлами `.spr` (или `.dat`).
        output_dir: Выходная директория для сохранения изображений `.png`.
        image_shape: Ожидаемая форма 2D-среза `(height, width)` при чтении одномерных массивов.

    Returns:
        Отсортированный список путей к созданным файлам `.png`.
    """
    source_dir = Path(spr_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Входная директория с файлами .spr не найдена: {source_dir}")
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError(f"Некорректная форма изображения image_shape: {image_shape}")

    spr_files = sorted(source_dir.glob("*.spr"))
    if not spr_files:
        spr_files = sorted(source_dir.glob("*.dat"))

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for spr_file in spr_files:
        raw_slice = np.loadtxt(spr_file, dtype=np.float64)
        if raw_slice.ndim == 1:
            expected_size = image_shape[0] * image_shape[1]
            if raw_slice.size != expected_size:
                raise ValueError(
                    f"Размер плоского массива в '{spr_file.name}' ({raw_slice.size}) "
                    f"не соответствует image_shape={image_shape} ({expected_size})."
                )
            raw_slice = raw_slice.reshape(image_shape)
        elif raw_slice.ndim != 2:
            raise ValueError(f"Ожидался 2D массив в файле '{spr_file.name}', получена форма {raw_slice.shape}.")

        clipped_slice = np.clip(raw_slice, 0.0, None)
        max_value = float(np.max(clipped_slice)) if clipped_slice.size > 0 else 0.0
        if max_value > 0.0:
            normalized_uint8 = np.round((clipped_slice / max_value) * 255.0).astype(np.uint8)
        else:
            normalized_uint8 = np.zeros_like(clipped_slice, dtype=np.uint8)

        png_path = target_dir / f"{spr_file.stem}.png"
        Image.fromarray(normalized_uint8, mode="L").save(png_path)
        saved_paths.append(png_path)

    return saved_paths
