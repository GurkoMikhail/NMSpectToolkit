"""Инструменты инспекции, верификации дерева и визуализации научных файлов HDF5 NeXus (.nxs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from hepunits import deg, keV


def format_nexus_tree(file_path: str | Path) -> str:
    """Формирует текстовое представление иерархического дерева групп, датасетов и атрибутов файла `.nxs`.

    Args:
        file_path: Путь к файлу HDF5 NeXus (`.nxs`).

    Returns:
        Форматированная многострочная строка с деревом структуры файла.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл NeXus не найден: {resolved_path}")

    lines: list[str] = [f"Структура файла NeXus: {resolved_path.name}"]

    def _decode_attr(val: Any) -> Any:
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        if isinstance(val, np.ndarray) and val.size <= 6:
            return val.tolist()
        return val

    with h5py.File(resolved_path, "r") as h5_file:
        for root_key, root_val in h5_file.attrs.items():
            lines.append(f"  @{root_key} = {_decode_attr(root_val)}")

        def _visitor(name: str, obj: Any) -> None:
            depth = name.count("/")
            indent = "  " * (depth + 1)
            short_name = name.split("/")[-1]
            if isinstance(obj, h5py.Group):
                nx_class = _decode_attr(obj.attrs.get("NX_class", "group"))
                lines.append(f"{indent}[Group] {short_name} ({nx_class})")
            elif isinstance(obj, h5py.Dataset):
                units_str = _decode_attr(obj.attrs.get("units", ""))
                units_suffix = f" [{units_str}]" if units_str else ""
                if obj.shape == ():
                    scalar_val = _decode_attr(obj[()])
                    lines.append(f"{indent}[Dataset] {short_name}: {scalar_val}{units_suffix} ({obj.dtype})")
                else:
                    lines.append(f"{indent}[Dataset] {short_name}: shape={obj.shape}, dtype={obj.dtype}{units_suffix}")

        h5_file.visititems(_visitor)

    return "\n".join(lines)


def inspect_nexus_file(file_path: str | Path) -> dict[str, Any]:
    """Извлекает сводную диагностическую информацию и метрики качества из файла `.nxs`.

    Args:
        file_path: Путь к файлу HDF5 NeXus (`.nxs`).

    Returns:
        Словарь с метаданными, формой массива проекций, диапазоном углов и спектрометрическими метриками.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл NeXus не найден: {resolved_path}")

    summary: dict[str, Any] = {"file_path": str(resolved_path)}
    with h5py.File(resolved_path, "r") as h5_file:
        proj_path = "entry/data/data" if "entry/data/data" in h5_file else "entry/data_projections/data"
        if proj_path in h5_file:
            proj_ds = h5_file[proj_path]
            summary["projections_shape"] = tuple(int(dim) for dim in proj_ds.shape)
            summary["projections_dtype"] = str(proj_ds.dtype)
            summary["total_counts"] = float(np.sum(proj_ds[:]))

        angle_path = (
            "entry/data/gantry_angle"
            if "entry/data/gantry_angle" in h5_file
            else (
                "entry/data_projections/gantry_angle"
                if "entry/data_projections/gantry_angle" in h5_file
                else "entry/data/rotation_angle"
            )
        )
        if angle_path in h5_file:
            angles = np.asarray(h5_file[angle_path][:], dtype=np.float64)
            display_angles = angles / deg if np.max(np.abs(angles)) <= 2.5 * np.pi else angles
            summary["num_angles"] = int(angles.size)
            summary["angle_min"] = float(np.min(display_angles)) if angles.size > 0 else 0.0
            summary["angle_max"] = float(np.max(display_angles)) if angles.size > 0 else 0.0

        metrics_path = (
            "entry/spectrum/metrics"
            if "entry/spectrum/metrics" in h5_file
            else "entry/data_spectrum/metrics"
        )
        if metrics_path in h5_file:
            metrics_group = h5_file[metrics_path]
            metrics_dict: dict[str, float] = {}
            for key in metrics_group:
                metrics_dict[key] = float(np.asarray(metrics_group[key][()]))
            summary["metrics"] = metrics_dict
    return summary


def plot_nexus_spectrum(
    file_path: str | Path,
    output_image_path: str | Path | None = None,
    show_interactive: bool = False,
) -> Path | None:
    """Строит график трёхуровневого спектра (Total, Primary, Scatter, Continuum) из файла `.nxs`.

    Args:
        file_path: Путь к файлу `.nxs`.
        output_image_path: Опциональный путь для сохранения графика в PNG/PDF.
        show_interactive: Показать интерактивное окно Matplotlib.

    Returns:
        Путь к сохранённому изображению или None.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл NeXus не найден: {resolved_path}")

    with h5py.File(resolved_path, "r") as h5_file:
        if "entry/spectrum" not in h5_file:
            raise KeyError(f"В файле {resolved_path} отсутствует группа 'entry/spectrum'.")
        spec_grp = h5_file["entry/spectrum"]
        energy_raw = np.asarray(spec_grp["energy"][:], dtype=np.float64)
        units_attr = spec_grp["energy"].attrs.get("units", "keV")
        if isinstance(units_attr, bytes):
            units_attr = units_attr.decode("utf-8", errors="ignore")
        energy_axis = energy_raw if str(units_attr).lower() == "kev" else energy_raw / keV

        total_spec = np.asarray(spec_grp["total"][:], dtype=np.float64)
        primary_spec = (
            np.asarray(spec_grp["primary"][:], dtype=np.float64) if "primary" in spec_grp else None
        )
        scatter_spec = (
            np.asarray(spec_grp["scatter"][:], dtype=np.float64) if "scatter" in spec_grp else None
        )
        continuum_spec = (
            np.asarray(spec_grp["compton_continuum"][:], dtype=np.float64)
            if "compton_continuum" in spec_grp
            else None
        )

    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=140)
    axis.plot(energy_axis, total_spec, label="Полный спектр (Total)", color="#1f77b4", linewidth=1.6)
    if primary_spec is not None:
        axis.plot(energy_axis, primary_spec, label="Первичные фотоны (Primary)", color="#2ca02c", linewidth=1.3)
    if scatter_spec is not None:
        axis.plot(energy_axis, scatter_spec, label="Рассеянные фотоны (Scatter)", color="#d62728", linewidth=1.3)
    if continuum_spec is not None:
        axis.plot(
            energy_axis,
            continuum_spec,
            label="Комптоновский континуум",
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.2,
        )

    axis.set_title(f"Энергетический спектр ОФЭКТ — {resolved_path.name}")
    axis.set_xlabel("Энергия фотонов, кэВ")
    axis.set_ylabel("Число отсчётов")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()

    saved_path: Path | None = None
    if output_image_path is not None:
        saved_path = Path(output_image_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path)

    if show_interactive:
        plt.show()
    plt.close(fig)
    return saved_path


def plot_nexus_projections(
    file_path: str | Path,
    angle_indices: list[int] | None = None,
    output_image_path: str | Path | None = None,
    show_interactive: bool = False,
) -> Path | None:
    """Строит сетку 2D-планарных проекций под выбранными ракурсами из файла `.nxs`.

    Args:
        file_path: Путь к файлу `.nxs`.
        angle_indices: Список индексов ракурсов для отображения (по умолчанию 4 равномерных ракурса).
        output_image_path: Опциональный путь для сохранения изображения.
        show_interactive: Показать интерактивное окно Matplotlib.

    Returns:
        Путь к сохранённому изображению или None.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл NeXus не найден: {resolved_path}")

    with h5py.File(resolved_path, "r") as h5_file:
        projections = np.asarray(h5_file["entry/data/data"][:], dtype=np.float64)
        angles = (
            np.asarray(h5_file["entry/data/rotation_angle"][:], dtype=np.float64)
            if "entry/data/rotation_angle" in h5_file
            else np.arange(projections.shape[0], dtype=np.float64)
        )

    num_views = projections.shape[0]
    if angle_indices is None or len(angle_indices) == 0:
        step = max(1, num_views // 4)
        selected_indices = [idx for idx in range(0, num_views, step)][:4]
    else:
        selected_indices = [idx for idx in angle_indices if 0 <= idx < num_views]

    cols = len(selected_indices)
    fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols, 3.8), dpi=140, squeeze=False)
    for col_idx, view_idx in enumerate(selected_indices):
        ax_item = axes[0, col_idx]
        image_handle = ax_item.imshow(projections[view_idx], cmap="inferno", origin="lower")
        ax_item.set_title(f"Ракурс #{view_idx} ({angles[view_idx]:.1f}°)")
        ax_item.set_xlabel("Пиксель X")
        ax_item.set_ylabel("Пиксель Y")
        fig.colorbar(image_handle, ax=ax_item, fraction=0.046, pad=0.04)

    fig.suptitle(f"Планарные проекции ОФЭКТ — {resolved_path.name}")
    fig.tight_layout()

    saved_path: Path | None = None
    if output_image_path is not None:
        saved_path = Path(output_image_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path)

    if show_interactive:
        plt.show()
    plt.close(fig)
    return saved_path
